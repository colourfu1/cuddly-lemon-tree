#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO-World-X 提示词探索脚本

目标：
1. 每个 prompt 独立运行，避免不同同义词在同一次 vocabulary 中互相竞争。
2. 在 YOLO 格式人工验证集上评估每个 prompt。
3. 输出 AP50、mAP50:95、最佳 F1 阈值，以及“满足最低 Precision 时尽量保 Recall”的推荐阈值。
4. 自动生成 best_yoloworld_prompts.yaml，供第二阶段伪标签脚本直接读取。
5. 支持缓存 predictions.csv，后续可 --skip-inference 只重扫阈值，不重复推理。

GT 默认类别：
0 = fire
1 = smoke
可通过 prompts.yaml 中 class_id 修改。
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLOWorld


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser("YOLO-World-X prompt evaluation")
    p.add_argument("--images", type=Path, required=True, help="验证集 images 目录")
    p.add_argument("--gt-yolo", type=Path, required=True, help="验证集 YOLO labels 目录")
    p.add_argument("--prompts", type=Path, required=True, help="提示词 YAML")
    p.add_argument("--output", type=Path, required=True, help="输出目录")
    p.add_argument("--model", type=str, default="", help="模型；为空时使用 prompts.yaml 里的 model")
    p.add_argument("--predictions-csv", type=Path, default=None, help="可选：指定预测缓存 CSV")
    p.add_argument("--skip-inference", action="store_true", help="跳过模型推理，直接读取 predictions.csv")

    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--half", action="store_true")
    p.add_argument("--infer-conf", type=float, default=0.001,
                   help="推理缓存最低置信度。探索阶段要低，阈值后续离线扫描")
    p.add_argument("--nms-iou", type=float, default=0.70)
    p.add_argument("--max-det", type=int, default=300)

    p.add_argument("--eval-iou", type=float, default=0.50,
                   help="阈值选择时 TP/FP 的 IoU 判定阈值")
    p.add_argument("--thr-start", type=float, default=0.05)
    p.add_argument("--thr-end", type=float, default=0.90)
    p.add_argument("--thr-step", type=float, default=0.01)
    p.add_argument("--min-precision", type=float, default=0.70,
                   help="伪标签推荐阈值的最低 Precision 目标；<=0 表示关闭该约束")
    p.add_argument(
        "--select-metric",
        choices=["map50_95", "ap50", "f1", "precision", "recall"],
        default="map50_95",
        help="同一语义类中选择 best prompt 的主指标"
    )
    return p.parse_args()


def list_images(root: Path):
    images = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if not images:
        raise FileNotFoundError(f"没有在 {root} 找到图片")
    return images


def load_prompt_cfg(path: Path):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "classes" not in cfg:
        raise ValueError("prompts.yaml 缺少 classes")
    rows = []
    for class_key, info in cfg["classes"].items():
        class_id = int(info["class_id"])
        prompts = info.get("prompts", {})
        for prompt_id, prompt_text in prompts.items():
            rows.append({
                "class_key": str(class_key),
                "class_id": class_id,
                "prompt_id": str(prompt_id),
                "prompt_text": str(prompt_text),
            })
    if not rows:
        raise ValueError("prompts.yaml 中没有 prompt")
    return cfg, rows


def yolo_gt_to_xyxy(label_path: Path, image_w: int, image_h: int, target_class_id: int):
    boxes = []
    if not label_path.exists():
        return np.empty((0, 4), dtype=np.float32)

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        if cls_id != target_class_id:
            continue
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = (xc - bw / 2.0) * image_w
        y1 = (yc - bh / 2.0) * image_h
        x2 = (xc + bw / 2.0) * image_w
        y2 = (yc + bh / 2.0) * image_h
        boxes.append([x1, y1, x2, y2])

    if not boxes:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def box_iou_one_to_many(box, boxes):
    """计算一个 box 与 N 个 boxes 的 IoU。"""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-9
    return inter / union


def build_gt_cache(images, image_root: Path, gt_root: Path, prompt_rows):
    """只读一次图片尺寸，并缓存 fire/smoke GT。"""
    class_ids = sorted({(r["class_key"], r["class_id"]) for r in prompt_rows})
    gt = {class_key: {} for class_key, _ in class_ids}
    image_shapes = {}

    for img_path in images:
        rel = img_path.relative_to(image_root).as_posix()
        im = cv2.imread(str(img_path))
        if im is None:
            raise RuntimeError(f"无法读取图片: {img_path}")
        h, w = im.shape[:2]
        image_shapes[rel] = (h, w)
        label_path = gt_root / Path(rel).with_suffix(".txt")

        for class_key, class_id in class_ids:
            gt[class_key][rel] = yolo_gt_to_xyxy(label_path, w, h, class_id)

    return gt, image_shapes


def run_inference(args, images, image_root, model_name, prompt_rows, pred_csv):
    model = YOLOWorld(model_name)
    pred_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "image", "prompt_id", "class_key", "class_id", "prompt_text",
        "confidence", "x1", "y1", "x2", "y2"
    ]

    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        total = len(prompt_rows)
        for idx, row in enumerate(prompt_rows, start=1):
            prompt = row["prompt_text"]
            print(f"[{idx:02d}/{total:02d}] {row['prompt_id']} [{row['class_key']}] -> {prompt}")

            # 关键：每次只设置一个文本类别。
            # 这样得到的是“这个 prompt 自身”的检测能力，而不是 prompt 间竞争后的结果。
            model.set_classes([prompt])

            results = model.predict(
                source=[str(p) for p in images],
                stream=True,
                batch=args.batch,
                imgsz=args.imgsz,
                conf=args.infer_conf,
                iou=args.nms_iou,
                max_det=args.max_det,
                device=args.device,
                half=args.half,
                verbose=False,
            )

            for img_path, result in zip(images, results):
                rel = img_path.relative_to(image_root).as_posix()
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                boxes = result.boxes.xyxy.detach().cpu().numpy()
                confs = result.boxes.conf.detach().cpu().numpy()

                for box, conf in zip(boxes, confs):
                    writer.writerow({
                        "image": rel,
                        "prompt_id": row["prompt_id"],
                        "class_key": row["class_key"],
                        "class_id": row["class_id"],
                        "prompt_text": prompt,
                        "confidence": float(conf),
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                    })


def load_predictions(pred_csv: Path):
    preds_by_prompt = defaultdict(list)
    meta = {}

    with pred_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pid = r["prompt_id"]
            item = {
                "image": r["image"],
                "confidence": float(r["confidence"]),
                "box": np.asarray(
                    [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])],
                    dtype=np.float32,
                ),
            }
            preds_by_prompt[pid].append(item)
            meta[pid] = {
                "prompt_id": pid,
                "class_key": r["class_key"],
                "class_id": int(r["class_id"]),
                "prompt_text": r["prompt_text"],
            }

    return preds_by_prompt, meta


def evaluate_threshold(preds, gt_by_image, conf_thr, iou_thr):
    """按图片逐一贪心匹配，得到指定置信度阈值下的 P/R/F1。"""
    pred_by_image = defaultdict(list)
    for p in preds:
        if p["confidence"] >= conf_thr:
            pred_by_image[p["image"]].append(p)

    tp = fp = 0
    matched_ious = []
    total_gt = sum(len(v) for v in gt_by_image.values())

    for image, image_preds in pred_by_image.items():
        image_preds = sorted(image_preds, key=lambda x: x["confidence"], reverse=True)
        gts = gt_by_image.get(image, np.empty((0, 4), dtype=np.float32))
        used = np.zeros(len(gts), dtype=bool)

        for p in image_preds:
            if len(gts) == 0:
                fp += 1
                continue

            ious = box_iou_one_to_many(p["box"], gts)
            ious[used] = -1.0
            j = int(np.argmax(ious))
            best_iou = float(ious[j])

            if best_iou >= iou_thr:
                used[j] = True
                tp += 1
                matched_ious.append(best_iou)
            else:
                fp += 1

    fn = total_gt - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {
        "threshold": float(conf_thr),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_matched_iou": mean_iou,
    }


def compute_ap(preds, gt_by_image, iou_thr):
    """101-point 插值 AP；每个 prompt 对应一个语义类。"""
    total_gt = sum(len(v) for v in gt_by_image.values())
    if total_gt == 0:
        return 0.0

    preds = sorted(preds, key=lambda x: x["confidence"], reverse=True)
    used = {k: np.zeros(len(v), dtype=bool) for k, v in gt_by_image.items()}

    tp_flags = []
    fp_flags = []

    for p in preds:
        gts = gt_by_image.get(p["image"], np.empty((0, 4), dtype=np.float32))
        if len(gts) == 0:
            tp_flags.append(0.0)
            fp_flags.append(1.0)
            continue

        ious = box_iou_one_to_many(p["box"], gts)
        u = used[p["image"]]
        ious[u] = -1.0
        j = int(np.argmax(ious))
        best_iou = float(ious[j])

        if best_iou >= iou_thr:
            u[j] = True
            tp_flags.append(1.0)
            fp_flags.append(0.0)
        else:
            tp_flags.append(0.0)
            fp_flags.append(1.0)

    if not tp_flags:
        return 0.0

    tp_cum = np.cumsum(np.asarray(tp_flags))
    fp_cum = np.cumsum(np.asarray(fp_flags))
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    recall_points = np.linspace(0.0, 1.0, 101)
    interp = []
    for r in recall_points:
        mask = recalls >= r
        interp.append(float(np.max(precisions[mask])) if np.any(mask) else 0.0)
    return float(np.mean(interp))


def pick_threshold(curve, min_precision):
    # 先得到纯 F1 最优阈值，便于和常规检测指标比较
    best_f1 = max(
        curve,
        key=lambda x: (x["f1"], x["precision"], x["recall"], x["threshold"])
    )

    # 伪标签阶段更看重 precision：
    # 若给了最低 precision 约束，则在满足约束的阈值里尽量保留 recall。
    if min_precision and min_precision > 0:
        valid = [x for x in curve if x["precision"] >= min_precision and (x["tp"] + x["fp"]) > 0]
        if valid:
            selected = max(
                valid,
                key=lambda x: (x["recall"], x["f1"], x["precision"], x["threshold"])
            )
            constraint_met = True
        else:
            selected = best_f1
            constraint_met = False
    else:
        selected = best_f1
        constraint_met = True

    return best_f1, selected, constraint_met


def metric_for_prompt(row, metric_name):
    if metric_name == "map50_95":
        return row["map50_95"]
    if metric_name == "ap50":
        return row["ap50"]
    if metric_name == "f1":
        return row["selected_f1"]
    if metric_name == "precision":
        return row["selected_precision"]
    if metric_name == "recall":
        return row["selected_recall"]
    raise ValueError(metric_name)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cfg, prompt_rows = load_prompt_cfg(args.prompts)
    model_name = args.model or cfg.get("model", "yolov8x-worldv2.pt")
    images = list_images(args.images)

    pred_csv = args.predictions_csv or (args.output / "predictions.csv")

    print(f"[Model] {model_name}")
    print(f"[Images] {len(images)}")
    print(f"[Prompts] {len(prompt_rows)}")
    for r in prompt_rows:
        print(f"  - {r['prompt_id']}: [{r['class_key']}] {r['prompt_text']}")

    gt, _ = build_gt_cache(images, args.images, args.gt_yolo, prompt_rows)

    if args.skip_inference:
        if not pred_csv.exists():
            raise FileNotFoundError(f"--skip-inference 但预测缓存不存在: {pred_csv}")
        print(f"[Cache] 读取 {pred_csv}")
    else:
        print("[Inference] 开始逐 prompt 推理")
        run_inference(args, images, args.images, model_name, prompt_rows, pred_csv)
        print(f"[Inference] 缓存已写入 {pred_csv}")

    preds_by_prompt, pred_meta = load_predictions(pred_csv)

    # 即使某 prompt 一张框都没出，也要保留在评测结果中
    row_meta = {r["prompt_id"]: r for r in prompt_rows}
    thresholds = np.arange(
        args.thr_start,
        args.thr_end + args.thr_step * 0.5,
        args.thr_step,
        dtype=np.float64,
    )

    metrics_rows = []
    curves_rows = []

    for pid, m in row_meta.items():
        class_key = m["class_key"]
        preds = preds_by_prompt.get(pid, [])
        gt_by_image = gt[class_key]

        curve = []
        for thr in thresholds:
            stat = evaluate_threshold(preds, gt_by_image, float(thr), args.eval_iou)
            curve.append(stat)
            curves_rows.append({
                "prompt_id": pid,
                "class_key": class_key,
                "prompt_text": m["prompt_text"],
                **stat,
            })

        best_f1, selected, constraint_met = pick_threshold(curve, args.min_precision)

        ap50 = compute_ap(preds, gt_by_image, 0.50)
        aps = [compute_ap(preds, gt_by_image, iou) for iou in np.arange(0.50, 0.96, 0.05)]
        map50_95 = float(np.mean(aps))

        metrics_rows.append({
            "prompt_id": pid,
            "class_key": class_key,
            "class_id": m["class_id"],
            "prompt_text": m["prompt_text"],
            "num_cached_boxes": len(preds),
            "ap50": ap50,
            "map50_95": map50_95,

            "best_f1_threshold": best_f1["threshold"],
            "best_f1_precision": best_f1["precision"],
            "best_f1_recall": best_f1["recall"],
            "best_f1": best_f1["f1"],

            "selected_threshold": selected["threshold"],
            "selected_precision": selected["precision"],
            "selected_recall": selected["recall"],
            "selected_f1": selected["f1"],
            "selected_tp": selected["tp"],
            "selected_fp": selected["fp"],
            "selected_fn": selected["fn"],
            "selected_mean_matched_iou": selected["mean_matched_iou"],
            "precision_constraint_met": int(constraint_met),
        })

    metrics_csv = args.output / "single_prompt_metrics.csv"
    if metrics_rows:
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)

    curves_csv = args.output / "threshold_curves.csv"
    if curves_rows:
        with curves_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(curves_rows[0].keys()))
            writer.writeheader()
            writer.writerows(curves_rows)

    # 每个语义类选择一个 best single prompt。
    # 这一步故意不做 prompt ensemble，便于后续把增益归因给“第二个异构 Teacher”本身。
    best_by_class = {}
    for class_key in cfg["classes"].keys():
        candidates = [r for r in metrics_rows if r["class_key"] == class_key]
        candidates.sort(
            key=lambda r: (
                metric_for_prompt(r, args.select_metric),
                r["selected_precision"],
                r["selected_recall"],
                r["selected_f1"],
            ),
            reverse=True,
        )
        best_by_class[class_key] = candidates[0]

    best_cfg = {
        "model_id": model_name,
        "selection": {
            "select_metric": args.select_metric,
            "eval_iou": args.eval_iou,
            "min_precision": args.min_precision,
            "imgsz": args.imgsz,
        },
        "classes": {},
    }

    print("\n========== YOLO-World-X 提示词结果 ==========")
    for class_key, best in best_by_class.items():
        best_cfg["classes"][class_key] = {
            "class_id": int(best["class_id"]),
            "prompt_id": best["prompt_id"],
            "prompt": best["prompt_text"],
            "threshold": float(best["selected_threshold"]),
            "metrics": {
                "ap50": float(best["ap50"]),
                "map50_95": float(best["map50_95"]),
                "precision": float(best["selected_precision"]),
                "recall": float(best["selected_recall"]),
                "f1": float(best["selected_f1"]),
                "mean_matched_iou": float(best["selected_mean_matched_iou"]),
                "precision_constraint_met": bool(best["precision_constraint_met"]),
            },
        }

        print(
            f"[{class_key}] best single: {best['prompt_id']} = '{best['prompt_text']}' | "
            f"mAP50:95={best['map50_95']:.4f} AP50={best['ap50']:.4f} | "
            f"thr={best['selected_threshold']:.2f} "
            f"P={best['selected_precision']:.4f} R={best['selected_recall']:.4f} "
            f"F1={best['selected_f1']:.4f}"
        )
        if not bool(best["precision_constraint_met"]):
            print(
                f"  ! 没有阈值达到 min_precision={args.min_precision:.2f}，"
                "best_yoloworld_prompts.yaml 已回退到该 prompt 的最佳 F1 阈值。"
            )

    best_yaml = args.output / "best_yoloworld_prompts.yaml"
    best_yaml.write_text(
        yaml.safe_dump(best_cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print("\n输出：")
    print(f"  - {pred_csv}")
    print(f"  - {metrics_csv}")
    print(f"  - {curves_csv}")
    print(f"  - {best_yaml}")


if __name__ == "__main__":
    main()
