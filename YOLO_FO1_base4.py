"""
完整本地版推理脚本（无 torch.hub / ultralytics 依赖）
功能：融合 YOLOv5、UPN 和 VLM-FO1，实现火灾/烟雾的高级检测、计数和描述。
流程：YOLOv5 -> UPN (Fallback) -> VLM-FO1 (Detection/Counting) -> Fusion -> Description。
"""
import os
import re
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# ================== 配置 YOLOv5 本地路径并校验 ==================
# 必须指向包含 models/ 的那一层目录
YOLOV5_LOCAL_DIR = "/root/autodl-tmp/yolov5/yolov5-master"

if not os.path.exists(YOLOV5_LOCAL_DIR):
    raise FileNotFoundError(f"❌ 路径不存在: {YOLOV5_LOCAL_DIR}")

models_path = os.path.join(YOLOV5_LOCAL_DIR, "models")
if not os.path.exists(models_path):
    raise FileNotFoundError(f"❌ 找不到 models 文件夹！请检查: {models_path}")

# 将 YOLOv5 路径添加到系统路径，以便导入其模块
sys.path.insert(0, YOLOV5_LOCAL_DIR)
print(f"✅ 已成功添加 YOLOv5 路径: {YOLOV5_LOCAL_DIR}")

# ================== 导入本地 YOLOv5 模块 ==================
from models.experimental import attempt_load                 # 用于加载本地的模型权重
from utils.general import non_max_suppression, scale_boxes  # 用于非极大值抑制和坐标缩放
from utils.augmentations import letterbox                   # 用于图像预处理（Letterbox）

# ================== 导入 VLM-FO1 模块 ==================
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import prepare_inputs, extract_predictions_to_bboxes
from vlm_fo1.task_templates import OD_template

# ================== 导入 UPN 检测器封装 ==================
try:
    from detect_tools.upn.inference_wrapper import UPNWrapper # 尝试导入 UPNWrapper
    has_upn = True
except ImportError:
    # 如果导入失败，禁用 UPN 功能
    print("⚠️ 未找到 detect_tools.upn.inference_wrapper.UPNWrapper，UPN 检测将被禁用，只使用 YOLO/整图 bbox。")
    has_upn = False

# ================== 配置路径 ==================
image_folder = "/root/autodl-tmp/datasets_input/images"    # 输入图像文件夹路径
output_folder = "/root/autodl-tmp/datasets_output"      # 输出结果文件夹路径
model_path = '/root/autodl-tmp/resources/resources/VLM-FO1_Qwen2.5-VL-3B-v01' # VLM-FO1 模型路径
wheel_path = '/root/autodl-tmp/resources/resources/flash_attn-2.8.0+cu124torch2.6-cp311-cp311-linux_x86_64.whl'
yolov5_weights_path = "/root/autodl-tmp/exp8best/weights/best.pt" # YOLOv5 权重路径
upn_ckpt_path = "/root/autodl-tmp/resources/resources/upn_large.pth" # UPN checkpoint 路径

os.makedirs(output_folder, exist_ok=True) # 创建输出文件夹

# ================== 安装 flash-attn（可选） ==================
if os.path.exists(wheel_path):
    os.system(f"pip install -q {wheel_path}") # 静默安装 flash-attn wheel 文件

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
device = "cuda" if torch.cuda.is_available() else "cpu" # 确定运行设备
yolov5_model = attempt_load(yolov5_weights_path, device=device)
yolov5_model.eval() # 设置模型为评估模式

# ================== （可选）加载 UPN 检测器 ==================
upn_detector = None
if has_upn and os.path.exists(upn_ckpt_path):
    try:
        print("🔥 加载 UPN 检测器，用于 YOLO 无检测框时的自动检测...")
        upn_detector = UPNWrapper(ckpt_path=upn_ckpt_path) # 初始化 UPN 检测器
    except Exception as e:
        print(f"⚠️ 加载 UPN 检测器失败，将不会使用 UPN。错误: {e}")
        upn_detector = None
else:
    if has_upn:
        print(f"⚠️ 未找到 UPN checkpoint 文件: {upn_ckpt_path}，将不会使用 UPN。")

# 类别标签（必须与 YOLOv5 训练时一致）
fire_label = 0
smoke_label = 1

# ================== 工具函数 ==================
def parse_count_from_text(text: str) -> int:
    """从 VLM 的自然语言回答中提取数量（支持英文数字和阿拉伯数字）"""
    text_lower = text.lower().strip()

    # 明确无目标的情况
    if any(word in text_lower for word in
           ["no", "none", "not", "zero", "not visible", "not detected", "no fire", "no smoke", "not any", "not present", "absence", "missing",
            "没有", "零个", "未发现"]):
        return 0

    # 英文数字映射（覆盖常见值）
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,"eleven": 11, "twelve": 12,
        "a single": 1,"dozens": 12,"hundreds": 100,
        "a couple": 2,"multiple": 2, "several": 3, "many": 5, "a lot": 5
    }

    for word, num in word_to_num.items():
        if word in text_lower:
            return num

    # 尝试提取阿拉伯数字
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[0])

    # 默认：如果提到对象但没给数量，至少算 1
    if any(obj in text_lower for obj in ["fire", "smoke", "flame", "burn", "smoke", "烟", "火"]):
        return 1

    return 0


def get_yolov5_bboxes(model, img_path, conf_thresh=0.3):
    """
    使用本地 YOLOv5 推理。
    返回: [x1, y1, x2, y2, conf, cls] 列表 和 原始图像 shape
    """
    img0 = cv2.imread(img_path)
    if img0 is None:
        return [], None

    # 预处理：Letterbox 缩放，调整维度 (HWC -> CHW)，归一化
    img = letterbox(img0, 640, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0) # 增加 Batch 维度

    # 推理 + NMS
    with torch.no_grad():
        pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thresh, 0.45, classes=None, agnostic=False)

    full_detections = [] # 存储完整的 YOLO 输出：[x1, y1, x2, y2, conf, cls]
    for det in pred:
        if len(det):
            # 将坐标从 640x640 缩放到原始图像尺寸
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                if float(conf) > conf_thresh:
                    x1, y1, x2, y2 = map(int, xyxy)
                    # 保存完整信息
                    full_detections.append([x1, y1, x2, y2, float(conf), int(cls)])

    return full_detections, img0.shape # 返回完整的检测结果和图像尺寸


def detect_and_count(object_name, bbox_list, img_path):
    """
    使用 VLM-FO1 在给定的 Proposals (bbox_list) 上进行目标检测和计数。
    返回: VLM 检测到的目标框 (xywh 格式) 和计数。
    """
    # ---------- 1. 目标检测任务 ----------
    detect_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_path}},
                {"type": "text", "text": OD_template.format(object_name)}, # 使用目标检测模板
            ],
            "bbox_list": bbox_list, # 传入 Proposals (xywh)
        }
    ]

    # ---------- 2. 目标计数任务 ----------
    count_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_path}},
                {"type": "text", "text": f"How many {object_name} are there in this image?"}, # 计数 Prompt
            ],
            "bbox_list": bbox_list,
        }
    ]

    # 执行检测推理
    detect_kwargs = prepare_inputs(model_path, model, image_processors, tokenizer, detect_messages, max_tokens=4096, top_p=0.05, temperature=0.0, do_sample=False)
    with torch.inference_mode():
        detect_output_ids = model.generate(**detect_kwargs)
        detect_outputs = tokenizer.decode(detect_output_ids[0, detect_kwargs['inputs'].shape[1]:]).strip()

    # 从 VLM 的输出中提取目标框
    label_to_bboxes = extract_predictions_to_bboxes(detect_outputs, bbox_list)
    bboxes = label_to_bboxes.get(object_name.lower(), []) # VLM 识别出的目标框，格式为 xywh

    # 执行计数推理
    count_kwargs = prepare_inputs(model_path, model, image_processors, tokenizer, count_messages, max_tokens=4096, top_p=0.05, temperature=0.0, do_sample=False)
    with torch.inference_mode():
        count_output_ids = model.generate(**count_kwargs)
        count_outputs = tokenizer.decode(count_output_ids[0, count_kwargs['inputs'].shape[1]:]).strip()

    # 从文本中解析数量
    count = parse_count_from_text(count_outputs)

    return bboxes, count


def vlm_xywh_to_xyxy(bbox_xywh):
    """将 VLM 输出的 [x, y, w, h] 格式转换为 [x1, y1, x2, y2]"""
    if len(bbox_xywh) < 4:
        return None
    x, y, w, h = bbox_xywh[:4]
    if w <= 0 or h <= 0:
        return None
    return int(x), int(y), int(x + w), int(y + h)


def compute_iou(box1, box2):
    """计算两个边界框的 IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    # 计算交集坐标
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    # 计算交集面积
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    # 计算并集面积
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def fuse_results(yolov5_dets, vlm_fire_bboxes, vlm_smoke_bboxes, fire_count, smoke_count, iou_threshold=0.3):
    """
    YOLO + VLM 结果融合逻辑。
    yolov5_dets 格式: [x1, y1, x2, y2, conf, cls]
    vlm_bboxes 格式: [x, y, w, h]
    """
    fused_bboxes = []

    def process_vlm_bboxes(vlm_dets):
        """将 VLM 的 xywh 框转换为 xyxy，并赋予默认高置信度 (0.9)"""
        processed = []
        for bbox_xywh in vlm_dets:
            xyxy = vlm_xywh_to_xyxy(bbox_xywh)
            if xyxy is None or xyxy[0] >= xyxy[2] or xyxy[1] >= xyxy[3]:
                continue
            processed.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], 0.9)) # 默认置信度 0.9
        return processed

    vlm_fire_xyxy = process_vlm_bboxes(vlm_fire_bboxes) if fire_count > 0 else []
    vlm_smoke_xyxy = process_vlm_bboxes(vlm_smoke_bboxes) if smoke_count > 0 else []

    # 情况1：融合 YOLOv5 检测结果 (以 YOLO 结果为主)
    for x1, y1, x2, y2, conf, cls in yolov5_dets:
        if conf <= 0.15: # 过滤低置信度的 YOLO 框
            continue
        cls_id = int(cls)
        yolo_box = (x1, y1, x2, y2)
        label_name = None
        should_promote = False # 标记是否应被 VLM 确认而提升置信度

        # 检查是否与 VLM 的 Fire 框重合
        if cls_id == fire_label and fire_count > 0:
            label_name = "Fire"
            for xb in vlm_fire_xyxy:
                if compute_iou(yolo_box, xb[:4]) > iou_threshold:
                    should_promote = True
                    break
        # 检查是否与 VLM 的 Smoke 框重合
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

        # 如果 VLM 确认，将置信度提升到 0.9
        final_conf = 0.9 if should_promote else conf
        fused_bboxes.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": final_conf, "label": label_name
        })

    # 情况2：回退到 VLM 结果 (如果 YOLO 未检测到或检测结果被过滤完)
    if not fused_bboxes and (fire_count > 0 or smoke_count > 0):
        for x1, y1, x2, y2, conf in vlm_fire_xyxy:
            # 将 VLM 识别出的 Fire 框加入结果 (使用 VLM 默认的 0.9 置信度)
            fused_bboxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "label": "Fire"})
        for x1, y1, x2, y2, conf in vlm_smoke_xyxy:
            # 将 VLM 识别出的 Smoke 框加入结果
            fused_bboxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "label": "Smoke"})

    return fused_bboxes


def get_detailed_description(object_name, img_path):
    """
    使用 VLM-FO1 生成详细描述。使用整图 bbox 作为 Proposal。
    无论是否检测到目标，都会尝试描述。
    """
    img_temp = Image.open(img_path)
    w, h = img_temp.size
    bbox_list = [[0, 0, w, h]] # 使用整图作为唯一的 Proposal [x, y, width, height]

    description_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_path}},
                {"type": "text", "text": f"Describe the {object_name} in this image."}, # 描述 Prompt
            ],
            "bbox_list": bbox_list,
        }
    ]
    # 使用 Top-p 和 Temperature 进行采样，以获得更具创造性的描述
    kwargs = prepare_inputs(model_path, model, image_processors, tokenizer, description_messages, max_tokens=4096, top_p=0.9, temperature=0.7, do_sample=True)
    with torch.inference_mode():
        output_ids = model.generate(**kwargs)
        outputs = tokenizer.decode(output_ids[0, kwargs['inputs'].shape[1]:]).strip()
    return outputs


# ================== 主流程 ==================
def main():
    for img_filename in os.listdir(image_folder):
        # 检查文件是否为图片
        if not img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue

        img_path = os.path.join(image_folder, img_filename)
        print(f"\n==============================")
        print(f"处理图像: {img_path}")

        # --- Step 1: YOLOv5 推理 ---
        # yolov5_full_dets 格式: [x1, y1, x2, y2, conf, cls]
        yolov5_full_dets, orig_shape = get_yolov5_bboxes(yolov5_model, img_path, conf_thresh=0.3)
        if orig_shape is None:
            print(f"⚠️ 无法读取图像，跳过: {img_path}")
            continue

        # --- Step 2: 构建 bbox_list (VLM 的 Proposals 输入) ---
        bbox_list = []

        if yolov5_full_dets:
            # 策略 1: 有 YOLO 检测框 → 使用 YOLO 框作为 Proposal
            print(f"✅ YOLO 检测到 {len(yolov5_full_dets)} 个框，用于 FO1 推理。")
            # 将 YOLO 的 xyxy 格式转换为 VLM 所需的 xywh 格式
            bbox_list = [[d[0], d[1], d[2] - d[0], d[3] - d[1]] for d in yolov5_full_dets]

        elif upn_detector is not None:
            # 策略 2: YOLO 无框且 UPN 可用 → 使用 UPN 生成 Proposal
            try:
                print("🔍 YOLO 无检测框，使用 UPN 检测器生成 proposals ...")
                pil_img = Image.open(img_path).convert("RGB")
                # UPN 推理并过滤
                upn_raw = upn_detector.inference(pil_img, prompt_type="fine_grained_prompt")
                upn_filtered = upn_detector.filter(upn_raw, min_score=0.4, nms_value=0.8)

                upn_boxes_xyxy = upn_filtered.get("boxes", []) if isinstance(upn_filtered, dict) else []
                print(f"✅ UPN 检测到 {len(upn_boxes_xyxy)} 个 proposals。")

                if len(upn_boxes_xyxy) > 0:
                    # 将 UPN 的 xyxy 格式转换为 VLM 所需的 xywh 格式
                    bbox_list = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                                 for x1, y1, x2, y2 in upn_boxes_xyxy]

            except Exception as e:
                print(f"⚠️ UPN 检测过程中出错，将退回整图 bbox。错误: {e}")
                bbox_list = []

        if not bbox_list:
            # 策略 3: 无有效 Proposals → 使用整图作为 Proposal
            h, w = orig_shape[:2]
            print("⚠️ YOLO 和 UPN 均无有效检测框，使用整图作为单一 bbox。")
            bbox_list = [[0, 0, w, h]] # [x, y, width, height]

        # --- Step 3: 调用 VLM-FO1 进行 fire / smoke 检测与计数 ---
        try:
            # vlm_fire_bboxes 格式为 VLM 输出的 xywh
            vlm_fire_bboxes, fire_count = detect_and_count("fire", bbox_list, img_path)
            vlm_smoke_bboxes, smoke_count = detect_and_count("smoke", bbox_list, img_path)
        except Exception as e:
            print(f"❌ VLM 推理出错（跳过此图）: {e}")
            continue

        # --- Step 4: 准备 YOLO 结果用于融合 ---
        # 使用 Step 1 中获取的完整 YOLO 检测结果
        yolov5_dets = yolov5_full_dets

        # --- Step 5: 融合 YOLO + VLM 结果 ---
        # fused_bboxes 格式: 字典列表 [{"x1": x, "y1": y, "x2": x, "y2": y, "conf": c, "label": l}, ...]
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
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # PIL Image 转 OpenCV 格式

        for box in fused_bboxes:
            # 提取融合后的边界框信息
            x1 = int(box["x1"])
            y1 = int(box["y1"])
            x2 = int(box["x2"])
            y2 = int(box["y2"])
            conf = float(box["conf"])
            label = box["label"]

            # 绘制边界框和标签
            color = (0, 255, 0) if label == "Fire" else (0, 0, 255) # Fire 用绿色，Smoke 用蓝色
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
        cv2.imwrite(output_path, img_np) # 保存绘制结果

        # --- Step 7: 生成详细描述 ---
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

        # 保存描述到文本文件
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

if __name__ == '__main__':
    main()
