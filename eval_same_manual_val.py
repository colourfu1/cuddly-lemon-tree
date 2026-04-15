#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ultralytics import YOLO
from pathlib import Path
import json

DATA_YAML = "/root/autodl-tmp/datasets_routeA/datap_routeA.yaml"
DEVICE = 0
IMGSZ = 640
BATCH = 16

MODELS = {
    "yolo1_teacher": "/root/autodl-tmp/ultralytics-main/ultralytics/runs/fair_compare/SAM3labels1/weights/best.pt",
    "routeA_yolo26s": "/root/autodl-tmp/ultralytics-main/ultralytics/runs/fair_compare/routeA_yolo26s_fair/weights/best.pt",
}

OUT_JSON = "/root/autodl-tmp/pseudo_pipeline/eval_same_manual_val.json"


def eval_one(name, weight_path):
    print(f"\n{'=' * 80}")
    print(f"Evaluating: {name}")
    print(f"weights: {weight_path}")
    print(f"data   : {DATA_YAML}")
    print(f"{'=' * 80}")

    model = YOLO(weight_path)
    metrics = model.val(
        data=DATA_YAML,
        split="val",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        plots=False,
        save_json=False,
        verbose=True,
    )

    result = {
        "model_name": name,
        "weight_path": weight_path,
        "all": {},
        "per_class": {}
    }

    # overall
    try:
        result["all"] = {
            "precision": float(metrics.results_dict.get("metrics/precision(B)", -1)),
            "recall": float(metrics.results_dict.get("metrics/recall(B)", -1)),
            "mAP50": float(metrics.results_dict.get("metrics/mAP50(B)", -1)),
            "mAP50-95": float(metrics.results_dict.get("metrics/mAP50-95(B)", -1)),
        }
    except Exception:
        pass

    # per class
    names = getattr(metrics, "names", {0: "fire", 1: "smoke"})
    for cls_id, cls_name in names.items():
        try:
            p, r, ap50, ap = metrics.class_result(int(cls_id))
            result["per_class"][str(cls_name)] = {
                "precision": float(p),
                "recall": float(r),
                "mAP50": float(ap50),
                "mAP50-95": float(ap),
            }
        except Exception as e:
            result["per_class"][str(cls_name)] = {"error": str(e)}

    print("\n[SUMMARY]")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    all_results = {}
    for name, weight in MODELS.items():
        if not Path(weight).exists():
            print(f"[WARN] weight not found: {weight}")
            continue
        all_results[name] = eval_one(name, weight)

    out_path = Path(OUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()