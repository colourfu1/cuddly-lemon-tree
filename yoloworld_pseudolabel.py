#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO-World-X 冷启动伪标签生成脚本

输入：
- 未标注图片目录
- 第一阶段 yoloworld_prompt_eval.py 生成的 best_yoloworld_prompts.yaml

输出：
output/
├── predictions_with_conf.csv      # 每个最终框的详细置信度与坐标
├── confidence_summary.csv         # 按类别的置信度统计
├── image_summary.csv              # 每张图的框数量/最高置信度等
├── audit_manifest.csv             # 抽检图清单与抽检原因
├── visualizations/                # 带框抽检图
└── yolo/
    ├── images/                    # 默认软链接原图，避免复制大数据
    ├── labels/                    # 标准 YOLO 5 列伪标签，可直接训练
    └── labels_with_conf/          # 6 列：class xc yc w h conf，仅用于审计

设计原则：
1. 最终每个语义类只使用提示词探索阶段选出的 best single prompt。
2. fire / smoke 使用各自独立阈值。
3. 标准训练标签不写 confidence，额外 sidecar 与 CSV 保留 confidence。
4. 对每张无目标图片也创建空 txt，避免训练时被误认为“标签缺失”。
5. 抽检可视化优先覆盖阈值边缘样本、随机正样本、无检测样本。
"""

import argparse
import csv
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLOWorld


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser("YOLO-World-X pseudo label generator")
    p.add_argument("--images", type=Path, required=True, help="待生成伪标签的图片目录")
    p.add_argument("--config", type=Path, required=True,
                   help="yoloworld_prompt_eval.py 生成的 best_yoloworld_prompts.yaml")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=str, default="",
                   help="可覆盖 config 里的 model_id；一般不需要")

    p.add_argument("--imgsz", type=int, default=0,
                   help="0=沿用 config selection.imgsz；也可手工覆盖")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--half", action="store_true")
    p.add_argument("--nms-iou", type=float, default=0.70)
    p.add_argument("--max-det", type=int, default=300)

    p.add_argument("--image-mode", choices=["symlink", "copy", "none"], default="symlink",
                   help="输出 yolo/images 的方式；AutoDL/Linux 推荐 symlink")
    p.add_argument("--vis-k", type=int, default=48, help="抽检可视化图片数量")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--line-width", type=int, default=2)
    return p.parse_args()


def list_images(root: Path):
    images = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if not images:
        raise FileNotFoundError(f"没有在 {root} 找到图片")
    return images


def load_config(path: Path):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "classes" not in cfg:
        raise ValueError("config 缺少 classes")

    class_rows = []
    for class_key, info in cfg["classes"].items():
        for required in ["class_id", "prompt", "threshold"]:
            if required not in info:
                raise ValueError(f"{class_key} 缺少字段 {required}")
        class_rows.append({
            "class_key": str(class_key),
            "class_id": int(info["class_id"]),
            "prompt_id": str(info.get("prompt_id", class_key)),
            "prompt": str(info["prompt"]),
            "threshold": float(info["threshold"]),
        })

    # 为了让 Ultralytics 的 cls index 与我们的 vocabulary 顺序严格一致，
    # 这里固定按 class_id 排序。最终 class_id 仍使用配置中的 class_id。
    class_rows.sort(key=lambda x: x["class_id"])
    return cfg, class_rows


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def prepare_image_link(src: Path, dst: Path, mode: str):
    """在输出数据集里创建图片软链接/复制；none 时不处理。"""
    if mode == "none":
        return
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(str(src.resolve()), str(dst))


def clip_xyxy(box, w, h):
    x1, y1, x2, y2 = map(float, box)
    x1 = min(max(x1, 0.0), float(w))
    y1 = min(max(y1, 0.0), float(h))
    x2 = min(max(x2, 0.0), float(w))
    y2 = min(max(y2, 0.0), float(h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def xyxy_to_yolo(box, w, h):
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return (
        min(max(xc, 0.0), 1.0),
        min(max(yc, 0.0), 1.0),
        min(max(bw, 0.0), 1.0),
        min(max(bh, 0.0), 1.0),
    )


def write_yolo_labels(label_path: Path, conf_label_path: Path, detections):
    ensure_parent(label_path)
    ensure_parent(conf_label_path)

    lines = []
    conf_lines = []
    for d in detections:
        xc, yc, bw, bh = d["xywhn"]
        lines.append(f"{d['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        conf_lines.append(
            f"{d['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {d['confidence']:.6f}"
        )

    # detections 为空时也写空文件，这是有效的 YOLO negative image 标签。
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    conf_label_path.write_text(
        "\n".join(conf_lines) + ("\n" if conf_lines else ""),
        encoding="utf-8",
    )


def draw_detections(img, detections, line_width=2):
    out = img.copy()
    h, w = out.shape[:2]
    font_scale = max(0.45, min(0.9, max(h, w) / 1800.0))

    # 不强行指定固定颜色，按类别生成稳定的 BGR 值，便于 fire/smoke 区分。
    palette = {
        "fire": (0, 165, 255),
        "smoke": (180, 180, 180),
    }

    for d in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
        color = palette.get(d["class_key"], (80, 220, 80))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_width)

        text = f"{d['class_key']} {d['confidence']:.3f}"
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, line_width)
        )
        ty = max(th + 4, y1)
        cv2.rectangle(
            out,
            (x1, ty - th - 6),
            (min(w - 1, x1 + tw + 6), ty + baseline),
            color,
            -1,
        )
        cv2.putText(
            out,
            text,
            (x1 + 3, ty - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            max(1, line_width),
            cv2.LINE_AA,
        )
    return out


def write_confidence_summary(path: Path, rows, class_rows):
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["class_key"]].append(float(r["confidence"]))

    fields = [
        "class_key", "class_id", "prompt", "threshold", "num_boxes",
        "mean_conf", "std_conf", "min_conf", "p10", "p25",
        "median", "p75", "p90", "max_conf"
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for c in class_rows:
            values = np.asarray(by_class.get(c["class_key"], []), dtype=np.float32)
            if len(values):
                stats = {
                    "mean_conf": float(np.mean(values)),
                    "std_conf": float(np.std(values)),
                    "min_conf": float(np.min(values)),
                    "p10": float(np.quantile(values, 0.10)),
                    "p25": float(np.quantile(values, 0.25)),
                    "median": float(np.quantile(values, 0.50)),
                    "p75": float(np.quantile(values, 0.75)),
                    "p90": float(np.quantile(values, 0.90)),
                    "max_conf": float(np.max(values)),
                }
            else:
                stats = {k: 0.0 for k in [
                    "mean_conf", "std_conf", "min_conf", "p10", "p25",
                    "median", "p75", "p90", "max_conf"
                ]}

            writer.writerow({
                "class_key": c["class_key"],
                "class_id": c["class_id"],
                "prompt": c["prompt"],
                "threshold": c["threshold"],
                "num_boxes": int(len(values)),
                **stats,
            })


def choose_audit_images(image_records, vis_k, seed):
    """
    抽检策略：
    - 约 1/2：阈值边缘正样本（最值得人工检查）
    - 约 1/4：随机正样本
    - 剩余：无检测图片（检查漏检）
    """
    if vis_k <= 0:
        return []

    rng = random.Random(seed)
    positives = [r for r in image_records if r["num_boxes"] > 0]
    negatives = [r for r in image_records if r["num_boxes"] == 0]

    positives_sorted = sorted(
        positives,
        key=lambda r: r["min_margin"] if r["min_margin"] is not None else 999.0
    )

    n_border = min(len(positives_sorted), vis_k // 2)
    border = positives_sorted[:n_border]
    used = {r["image"] for r in border}

    remain_pos = [r for r in positives if r["image"] not in used]
    rng.shuffle(remain_pos)
    n_random_pos = min(len(remain_pos), max(0, (vis_k - len(border)) // 2))
    random_pos = remain_pos[:n_random_pos]
    used.update(r["image"] for r in random_pos)

    rng.shuffle(negatives)
    slots = vis_k - len(border) - len(random_pos)
    neg = negatives[:max(0, slots)]
    used.update(r["image"] for r in neg)

    # 如果负样本不够，再用剩余正样本补齐
    if len(border) + len(random_pos) + len(neg) < vis_k:
        extra = [r for r in remain_pos[n_random_pos:] if r["image"] not in used]
        slots = vis_k - len(border) - len(random_pos) - len(neg)
        random_pos.extend(extra[:slots])

    result = []
    for r in border:
        result.append((r, "borderline_positive"))
    for r in random_pos:
        result.append((r, "random_positive"))
    for r in neg:
        result.append((r, "no_detection"))
    return result[:vis_k]


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cfg, class_rows = load_config(args.config)
    model_name = args.model or cfg.get("model_id", "yolov8x-worldv2.pt")
    config_imgsz = int(cfg.get("selection", {}).get("imgsz", 1280))
    imgsz = args.imgsz if args.imgsz > 0 else config_imgsz

    images = list_images(args.images)
    print(f"[Model] {model_name}")
    print(f"[Images] {len(images)}")
    print(f"[imgsz] {imgsz}")

    # vocabulary 顺序与 YOLO-World 输出 cls index 一一对应
    vocabulary = [c["prompt"] for c in class_rows]
    for vocab_idx, c in enumerate(class_rows):
        c["vocab_idx"] = vocab_idx
        print(
            f"  - cls={c['class_id']} [{c['class_key']}] prompt='{c['prompt']}' "
            f"threshold={c['threshold']:.3f}"
        )

    min_threshold = min(c["threshold"] for c in class_rows)
    # 推理阶段只需把全局 conf 设为各类别阈值中的最小值；
    # 之后再按类别分别过滤，避免因全局阈值过高提前丢框。
    inference_conf = max(0.001, min_threshold)

    model = YOLOWorld(model_name)
    model.set_classes(vocabulary)

    yolo_root = args.output / "yolo"
    out_images = yolo_root / "images"
    out_labels = yolo_root / "labels"
    out_conf_labels = yolo_root / "labels_with_conf"
    vis_dir = args.output / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    box_rows = []
    image_records = []
    dets_by_image = {}

    results = model.predict(
        source=[str(p) for p in images],
        stream=True,
        batch=args.batch,
        imgsz=imgsz,
        conf=inference_conf,
        iou=args.nms_iou,
        max_det=args.max_det,
        device=args.device,
        half=args.half,
        verbose=False,
    )

    for idx, (img_path, result) in enumerate(zip(images, results), start=1):
        rel_path = img_path.relative_to(args.images)
        rel = rel_path.as_posix()

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"无法读取图片: {img_path}")
        h, w = img.shape[:2]

        final_dets = []

        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            clses = result.boxes.cls.detach().cpu().numpy().astype(int)

            for box, conf, vocab_idx in zip(xyxy, confs, clses):
                if vocab_idx < 0 or vocab_idx >= len(class_rows):
                    continue

                c = class_rows[vocab_idx]
                conf = float(conf)

                # 关键：每个最终类别使用自己的验证集最优阈值。
                if conf < c["threshold"]:
                    continue

                clipped = clip_xyxy(box, w, h)
                if clipped is None:
                    continue

                xywhn = xyxy_to_yolo(clipped, w, h)
                det = {
                    "image": rel,
                    "class_id": c["class_id"],
                    "class_key": c["class_key"],
                    "prompt_id": c["prompt_id"],
                    "prompt": c["prompt"],
                    "threshold": c["threshold"],
                    "confidence": conf,
                    "xyxy": clipped,
                    "xywhn": xywhn,
                    "margin_to_threshold": conf - c["threshold"],
                }
                final_dets.append(det)

        # 排序只是为了让 txt / CSV 更稳定、可复现
        final_dets.sort(key=lambda d: (d["class_id"], -d["confidence"]))
        dets_by_image[rel] = final_dets

        label_path = out_labels / rel_path.with_suffix(".txt")
        conf_label_path = out_conf_labels / rel_path.with_suffix(".txt")
        write_yolo_labels(label_path, conf_label_path, final_dets)

        if args.image_mode != "none":
            dst_img = out_images / rel_path
            prepare_image_link(img_path, dst_img, args.image_mode)

        counts = defaultdict(int)
        max_conf = 0.0
        min_margin = None
        for d in final_dets:
            counts[d["class_key"]] += 1
            max_conf = max(max_conf, d["confidence"])
            margin = d["margin_to_threshold"]
            min_margin = margin if min_margin is None else min(min_margin, margin)

            x1, y1, x2, y2 = d["xyxy"]
            xc, yc, bw, bh = d["xywhn"]
            box_rows.append({
                "image": rel,
                "class_id": d["class_id"],
                "class_key": d["class_key"],
                "prompt_id": d["prompt_id"],
                "prompt": d["prompt"],
                "threshold": d["threshold"],
                "confidence": d["confidence"],
                "margin_to_threshold": d["margin_to_threshold"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "xc_norm": xc,
                "yc_norm": yc,
                "w_norm": bw,
                "h_norm": bh,
            })

        rec = {
            "image": rel,
            "num_boxes": len(final_dets),
            "max_conf": max_conf,
            "min_margin": min_margin,
        }
        for c in class_rows:
            rec[f"{c['class_key']}_boxes"] = counts[c["class_key"]]
        image_records.append(rec)

        if idx % 100 == 0 or idx == len(images):
            print(f"[Progress] {idx}/{len(images)}")

    # 详细框 CSV
    pred_csv = args.output / "predictions_with_conf.csv"
    pred_fields = [
        "image", "class_id", "class_key", "prompt_id", "prompt",
        "threshold", "confidence", "margin_to_threshold",
        "x1", "y1", "x2", "y2",
        "xc_norm", "yc_norm", "w_norm", "h_norm"
    ]
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fields)
        writer.writeheader()
        writer.writerows(box_rows)

    # 每张图片汇总
    image_csv = args.output / "image_summary.csv"
    image_fields = ["image", "num_boxes"] + [
        f"{c['class_key']}_boxes" for c in class_rows
    ] + ["max_conf", "min_margin"]
    with image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=image_fields)
        writer.writeheader()
        for r in image_records:
            writer.writerow({k: r.get(k, "") for k in image_fields})

    # 置信度分布汇总
    conf_csv = args.output / "confidence_summary.csv"
    write_confidence_summary(conf_csv, box_rows, class_rows)

    # 抽检可视化
    audit = choose_audit_images(image_records, args.vis_k, args.seed)
    audit_csv = args.output / "audit_manifest.csv"
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "reason", "num_boxes", "max_conf", "min_margin", "visualization"]
        )
        writer.writeheader()

        for rec, reason in audit:
            src = args.images / Path(rec["image"])
            img = cv2.imread(str(src))
            rendered = draw_detections(
                img,
                dets_by_image.get(rec["image"], []),
                line_width=args.line_width,
            )

            # 子目录路径打平，避免同名文件互相覆盖
            safe_name = rec["image"].replace("/", "__").replace("\\", "__")
            out_path = vis_dir / f"{Path(safe_name).stem}__{reason}.jpg"
            cv2.imwrite(str(out_path), rendered)

            writer.writerow({
                "image": rec["image"],
                "reason": reason,
                "num_boxes": rec["num_boxes"],
                "max_conf": rec["max_conf"],
                "min_margin": rec["min_margin"] if rec["min_margin"] is not None else "",
                "visualization": out_path.name,
            })

    total_boxes = len(box_rows)
    total_positive = sum(1 for r in image_records if r["num_boxes"] > 0)
    print("\n========== 伪标签生成完成 ==========")
    print(f"图片数: {len(images)}")
    print(f"有检测图片: {total_positive}")
    print(f"最终伪标签框: {total_boxes}")
    for c in class_rows:
        n = sum(1 for r in box_rows if r["class_key"] == c["class_key"])
        print(
            f"[{c['class_key']}] boxes={n} "
            f"prompt='{c['prompt']}' threshold={c['threshold']:.3f}"
        )

    print("\n输出目录:")
    print(f"  - {pred_csv}")
    print(f"  - {conf_csv}")
    print(f"  - {image_csv}")
    print(f"  - {audit_csv}")
    print(f"  - {vis_dir}")
    print(f"  - {out_labels}              # 标准 YOLO 伪标签")
    print(f"  - {out_conf_labels}    # 带 confidence 的审计标签")
    if args.image_mode != "none":
        print(f"  - {out_images}              # {args.image_mode} 原图")


if __name__ == "__main__":
    main()
