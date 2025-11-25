# -*- coding: utf- -*-
"""
完整本地版推理脚本（无 torch.hub / ultralytics 依赖）
YOLOv5 使用本地 v7.0 代码 + 自定义 best.pt
在 YOLO 没有检测框时，会调用 VLM-FO1 自带的 UPN 检测器生成 proposals。
无论是否有检测框、是否检测到 fire/smoke，都会生成详细描述。
"""

import os
import re
import sys

# ================== 配置 YOLOv5 本地路径并校验 ==================
YOLOV5_LOCAL_DIR = "/root/autodl-tmp/yolov5/yolov5-master"  # 必须指向包含 models/ 的那一层

if not os.path.exists(YOLOV5_LOCAL_DIR):
    raise FileNotFoundError(f"❌ 路径不存在: {YOLOV5_LOCAL_DIR}")

models_path = os.path.join(YOLOV5_LOCAL_DIR, "models")
if not os.path.exists(models_path):
    raise FileNotFoundError(f"❌ 找不到 models 文件夹！请检查: {models_path}")

sys.path.insert(0, YOLOV5_LOCAL_DIR)
print(f"✅ 已成功添加 YOLOv5 路径: {YOLOV5_LOCAL_DIR}")

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# 再次添加（你原来的做法），指向 YOLOv5 主目录
YOLOV5_LOCAL_DIR = "/root/autodl-tmp/yolov5"
sys.path.insert(0, YOLOV5_LOCAL_DIR)

# ================== 导入本地 YOLOv5 模块 ==================
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# ================== 导入 VLM-FO1 模块 ==================
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import prepare_inputs, draw_bboxes_and_save, extract_predictions_to_bboxes
from vlm_fo1.task_templates import OD_template

# ================== 导入 UPN 检测器封装 ==================
# 参见官方文档：UPNWrapper 提供 inference() 和 filter() 接口([DeepWiki](https://deepwiki.com/om-ai-lab/VLM-FO1/2.2-quick-start-guide))
try:
    from detect_tools.upn.inference_wrapper import UPNWrapper
    has_upn = True
except ImportError:
    print("⚠️ 未找到 detect_tools.upn.inference_wrapper.UPNWrapper，UPN 检测将被禁用，只使用 YOLO/整图 bbox。")
    has_upn = False

# ================== 配置路径 ==================
image_folder = "/root/autodl-tmp/datasets_input/images"       # 输入图像文件夹路径
output_folder = "/root/autodl-tmp/datasets_output"            # 输出结果文件夹路径
model_path = '/root/autodl-tmp/resources/resources/VLM-FO1_Qwen2.5-VL-3B-v01'  # VLM-FO1 模型路径
wheel_path = '/root/autodl-tmp/resources/resources/flash_attn-2.8.0+cu124torch2.6-cp311-cp311-linux_x86_64.whl'
yolov5_weights_path = "/root/autodl-tmp/exp8best/weights/best.pt"  # 你的 YOLOv5 best.pt
upn_ckpt_path = "/root/autodl-tmp/resources/resources/upn_large.pth"  # UPN checkpoint，按需修改

os.makedirs(output_folder, exist_ok=True)

# ================== 安装 flash-attn（可选） ==================
if os.path.exists(wheel_path):
    os.system(f"pip install -q {wheel_path}")

# ================== 加载 VLM-FO1 模型 ==================
print("🔥 加载 VLM-FO1 模型...")
tokenizer, model, image_processors = load_pretrained_model(
    model_path,
    load_8bit=False,
    load_4bit=False,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# ================== 加载本地 YOLOv5 模型 ==================
print("🔥 加载本地 YOLOv5 模型 (v7.0)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
yolov5_model = attempt_load(yolov5_weights_path, device=device)
yolov5_model.eval()

# ================== （可选）加载 UPN 检测器 ==================
upn_detector = None
if has_upn and os.path.exists(upn_ckpt_path):
    try:
        print("🔥 加载 UPN 检测器，用于 YOLO 无检测框时的自动检测...")
        # 官方文档说明：UPNWrapper.__init__(ckpt_path) 会从 checkpoint 初始化模型([DeepWiki](https://deepwiki.com/om-ai-lab/VLM-FO1/2.2-quick-start-guide))
        upn_detector = UPNWrapper(ckpt_path=upn_ckpt_path)
    except Exception as e:
        print(f"⚠️ 加载 UPN 检测器失败，将不会使用 UPN。错误: {e}")
        upn_detector = None
else:
    if has_upn:
        print(f"⚠️ 未找到 UPN checkpoint 文件: {upn_ckpt_path}，将不会使用 UPN。")

# 类别标签（必须与你训练时一致）
fire_label = 0
smoke_label = 1

# ================== 工具函数 ==================
def parse_count_from_text(text: str) -> int:
    """从 VLM 的自然语言回答中提取数量（支持英文数字和阿拉伯数字）"""
    text_lower = text.lower().strip()

    # 明确无目标
    if any(word in text_lower for word in
           ["no", "none", "not", "zero", "not visible", "not detected", "no fire", "no smoke"]):
        return 0

    # 英文数字映射（覆盖常见值）
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "multiple": 2, "several": 3, "many": 5, "a lot": 5
    }

    for word, num in word_to_num.items():
        if word in text_lower:
            return num

    # 尝试提取阿拉伯数字
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[0])

    # 默认：如果提到对象但没给数量，至少算 1
    if any(obj in text_lower for obj in ["fire", "smoke", "flame", "burn", "smoke"]):
        return 1

    return 0


def get_yolov5_bboxes(model, img_path, conf_thresh=0.3):
    """使用本地 YOLOv5 推理，返回 [x1, y1, x2, y2] 列表 和 原始图像 shape"""
    img0 = cv2.imread(img_path)
    if img0 is None:
        return [], None

    # 预处理
    img = letterbox(img0, 640, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 推理 + NMS
    with torch.no_grad():
        pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thresh, 0.45, classes=None, agnostic=False)

    bboxes = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                if float(conf) > conf_thresh:
                    x1, y1, x2, y2 = map(int, xyxy)
                    bboxes.append([x1, y1, x2, y2])
    return bboxes, img0.shape


def detect_and_count(object_name, bbox_list, img_path):
    """
    使用 VLM-FO1 在给定 bbox_list 上做检测和计数。
    返回:
        - bboxes: List[List[float]] 格式为 [[x, y, w, h], ...]（VLM 输出中属于该 object_name 的 bbox）
        - count: int
    """
    # ---------- 检测 ----------
    detect_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_path}},
                {"type": "text", "text": OD_template.format(object_name)},
            ],
            "bbox_list": bbox_list,
        }
    ]

    # ---------- 计数 ----------
    count_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_path}},
                {"type": "text", "text": f"How many {object_name} are there in this image?"},
            ],
            "bbox_list": bbox_list,
        }
    ]

    detect_kwargs = prepare_inputs(
        model_path, model, image_processors, tokenizer, detect_messages,
        max_tokens=4096, top_p=0.05, temperature=0.0, do_sample=False
    )
    with torch.inference_mode():
        detect_output_ids = model.generate(**detect_kwargs)
        detect_outputs = tokenizer.decode(
            detect_output_ids[0, detect_kwargs['inputs'].shape[1]:]
        ).strip()

    label_to_bboxes = extract_predictions_to_bboxes(detect_outputs, bbox_list)
    bboxes = label_to_bboxes.get(object_name.lower(), [])

    count_kwargs = prepare_inputs(
        model_path, model, image_processors, tokenizer, count_messages,
        max_tokens=4096, top_p=0.05, temperature=0.0, do_sample=False
    )
    with torch.inference_mode():
        count_output_ids = model.generate(**count_kwargs)
        count_outputs = tokenizer.decode(
            count_output_ids[0, count_kwargs['inputs'].shape[1]:]
        ).strip()

    count = parse_count_from_text(count_outputs)

    return bboxes, count


def vlm_xywh_to_xyxy(bbox_xywh):
    if len(bbox_xywh) < 4:
        return None
    x, y, w, h = bbox_xywh[:4]
    if w <= 0 or h <= 0:
        return None
    return int(x), int(y), int(x + w), int(y + h)


def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def fuse_results(yolov5_dets, vlm_fire_bboxes, vlm_smoke_bboxes, fire_count, smoke_count, iou_threshold=0.3):
    """
    YOLO + VLM 融合：
    - 当 YOLO 有框时，用 IoU 判断是否与 VLM 框重合，重合则提升置信度；
    - 当 YOLO 无框但 VLM 有结果时，直接使用 VLM 框。
    """
    fused_bboxes = []

    def process_vlm_bboxes(vlm_dets):
        processed = []
        for bbox_xywh in vlm_dets:
            xyxy = vlm_xywh_to_xyxy(bbox_xywh)
            if xyxy is None or xyxy[0] >= xyxy[2] or xyxy[1] >= xyxy[3]:
                continue
            processed.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], 0.9))
        return processed

    vlm_fire_xyxy = process_vlm_bboxes(vlm_fire_bboxes) if fire_count > 0 else []
    vlm_smoke_xyxy = process_vlm_bboxes(vlm_smoke_bboxes) if smoke_count > 0 else []

    # 情况1：YOLOv5 有检测
    for x1, y1, x2, y2, conf, cls in yolov5_dets:
        if conf <= 0.15:
            continue
        cls_id = int(cls)
        yolo_box = (x1, y1, x2, y2)
        label_name = None
        should_promote = False

        if cls_id == fire_label and fire_count > 0:
            label_name = "Fire"
            for xb in vlm_fire_xyxy:
                if compute_iou(yolo_box, xb[:4]) > iou_threshold:
                    should_promote = True
                    break
        elif cls_id == smoke_label and smoke_count > 0:
            label_name = "Smoke"
            for xb in vlm_smoke_xyxy:
                if compute_iou(yolo_box, xb[:4]) > iou_threshold:
                    should_promote = True
                    break
        else:
            continue

        if label_name is None:
            continue

        final_conf = 0.9 if should_promote else conf
        fused_bboxes.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": final_conf, "label": label_name
        })

    # 情况2：回退到 VLM（无 YOLO 框或 YOLO 框未通过过滤）
    if not fused_bboxes and (fire_count > 0 or smoke_count > 0):
        for x1, y1, x2, y2, conf in vlm_fire_xyxy:
            fused_bboxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "label": "Fire"})
        for x1, y1, x2, y2, conf in vlm_smoke_xyxy:
            fused_bboxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "label": "Smoke"})

    return fused_bboxes


def get_detailed_description(object_name, img_path):
    """
    使用 VLM-FO1 生成详细描述。
    为避免 bbox_list 缺失导致崩溃，这里总是用整图 bbox 作为 fallback。
    无论是否存在该 object_name（fire/smoke），都会尝试描述。
    """
    img_temp = Image.open(img_path)
    w, h = img_temp.size
    bbox_list = [[0, 0, w, h]]  # [x, y, width, height]

    description_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_path}},
                {"type": "text", "text": f"Describe the {object_name} in this image."},
            ],
            "bbox_list": bbox_list,
        }
    ]
    kwargs = prepare_inputs(
        model_path, model, image_processors, tokenizer, description_messages,
        max_tokens=4096, top_p=0.9, temperature=0.7, do_sample=True
    )
    with torch.inference_mode():
        output_ids = model.generate(**kwargs)
        outputs = tokenizer.decode(
            output_ids[0, kwargs['inputs'].shape[1]:]
        ).strip()
    return outputs


# ================== 主流程 ==================
for img_filename in os.listdir(image_folder):
    if not img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        continue

    img_path = os.path.join(image_folder, img_filename)
    print(f"\n==============================")
    print(f"处理图像: {img_path}")

    # --- Step 1: YOLOv5 推理 ---
    yolov5_bboxes_coords, orig_shape = get_yolov5_bboxes(yolov5_model, img_path, conf_thresh=0.3)
    if orig_shape is None:
        print(f"⚠️ 无法读取图像，跳过: {img_path}")
        continue

    # --- Step 2: 构建 bbox_list（优先 YOLO，其次 UPN，最后整图）---
    bbox_list = []

    if yolov5_bboxes_coords:
        # 有 YOLO 检测框：正常使用 [x1,y1,x2,y2] → [x,y,w,h]
        print(f"✅ YOLO 检测到 {len(yolov5_bboxes_coords)} 个框，用于 FO1 推理。")
        bbox_list = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in yolov5_bboxes_coords]

    elif upn_detector is not None:
        # 没有 YOLO 框，尝试使用 UPN 自动检测（FO1 官方推荐的 Path2）([DeepWiki](https://deepwiki.com/om-ai-lab/VLM-FO1/2.2-quick-start-guide))
        try:
            print("🔍 YOLO 无检测框，使用 UPN 检测器生成 proposals ...")
            pil_img = Image.open(img_path).convert("RGB")
            # prompt_type 使用 fine_grained_prompt，适合精细目标检测和计数([DeepWiki](https://deepwiki.com/om-ai-lab/VLM-FO1/2.2-quick-start-guide))
            upn_raw = upn_detector.inference(pil_img, prompt_type="fine_grained_prompt")
            upn_filtered = upn_detector.filter(upn_raw, min_score=0.4, nms_value=0.8)

            # 官方文档说明：filter 输出包含 boxes（[x1,y1,x2,y2]）和 scores 等字段([DeepWiki](https://deepwiki.com/om-ai-lab/VLM-FO1/2.2-quick-start-guide))
            upn_boxes_xyxy = upn_filtered.get("boxes", []) if isinstance(upn_filtered, dict) else []
            print(f"✅ UPN 检测到 {len(upn_boxes_xyxy)} 个 proposals。")

            if len(upn_boxes_xyxy) > 0:
                bbox_list = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                             for x1, y1, x2, y2 in upn_boxes_xyxy]

        except Exception as e:
            print(f"⚠️ UPN 检测过程中出错，将退回整图 bbox。错误: {e}")
            bbox_list = []

    if not bbox_list:
        # YOLO 与 UPN 都没有给出有效 bbox，用整图 fallback
        h, w = orig_shape[:2]
        print("⚠️ YOLO 和 UPN 均无有效检测框，使用整图作为单一 bbox。")
        bbox_list = [[0, 0, w, h]]

    # --- Step 3: 调用 VLM-FO1 进行 fire / smoke 检测与计数 ---
    try:
        vlm_fire_bboxes, fire_count = detect_and_count("fire", bbox_list, img_path)
        vlm_smoke_bboxes, smoke_count = detect_and_count("smoke", bbox_list, img_path)
    except Exception as e:
        print(f"❌ VLM 推理出错（跳过此图）: {e}")
        continue

    # --- Step 4: 构建 yolov5_dets（仅当 YOLO 有框时）---
    yolov5_dets = []
    for (x1, y1, x2, y2) in yolov5_bboxes_coords:
        # 这里可以根据需要替换成真实的 conf 和 cls
        yolov5_dets.append([x1, y1, x2, y2, 0.5, fire_label])  # 简化：全部当作 fire

    # --- Step 5: 融合 YOLO + VLM 结果 ---
    fused_bboxes = fuse_results(
        yolov5_dets,
        vlm_fire_bboxes,
        vlm_smoke_bboxes,
        fire_count,
        smoke_count,
        iou_threshold=0.3
    )

    # --- Step 6: 绘图并保存 ---
    image = Image.open(img_path).convert("RGB")
    img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for box in fused_bboxes:
        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])
        conf = float(box["conf"])
        label = box["label"]

        color = (0, 255, 0) if label == "Fire" else (0, 0, 255)
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img_np,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    output_filename = os.path.splitext(img_filename)[0] + "_result.jpg"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, img_np)

    # --- Step 7: 生成详细描述（❗无论是否有检测框 / 是否检测到 fire/smoke 都会执行）---
    try:
        detailed_description_fire = get_detailed_description("fire", img_path)
    except Exception as e:
        detailed_description_fire = f"[Fire 描述失败: {e}]"

    try:
        detailed_description_smoke = get_detailed_description("smoke", img_path)
    except Exception as e:
        detailed_description_smoke = f"[Smoke 描述失败: {e}]"

    print(f"Fire Description: {detailed_description_fire}")
    print(f"Smoke Description: {detailed_description_smoke}")

    description_filename = os.path.splitext(img_filename)[0] + "_description.txt"
    description_path = os.path.join(output_folder, description_filename)
    with open(description_path, "w", encoding="utf-8") as f:
        f.write(f"Fire Description: {detailed_description_fire}\n")
        f.write(f"Smoke Description: {detailed_description_smoke}\n")

    print(f"火的数量: {fire_count}")
    print(f"烟的数量: {smoke_count}")
    print(f"可视化结果已保存到: {output_path}")
    print(f"描述文件已保存到: {description_path}")

    if len(fused_bboxes) == 0 and fire_count == 0 and smoke_count == 0:
        print("未检测到任何 Fire 或 Smoke。")