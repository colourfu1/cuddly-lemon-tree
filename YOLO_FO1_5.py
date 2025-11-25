# -*- coding: utf-8 -*-
"""
完整本地版推理脚本（YOLO + VLM-FO1，英文日志版）
- YOLO 有框 → 使用真实类别（fire=0, smoke=1），将框区域送入 VLM 计数
  - 若 VLM 计数 > 1，则置信度提升至 0.9
- YOLO 无框 → 3x3 网格 → VLM 检测 fire/smoke
- 若检测到目标 → 绘制对应颜色框（火焰: 绿, 烟雾: 红）
  - YOLO 模式：使用调整后的置信度
  - 网格模式：VLM 结果，置信度固定为 0.5
- 始终生成「fire」「smoke」的英文详细描述（基于整图），并直接包含在日志中
- 终端日志为英文
- 每张图的可视化结果保存，所有日志汇总到 total_processing_log.txt
"""

import os
import re
import sys
import signal

# ================== YOLOv5 路径 ==================
YOLOV5_LOCAL_DIR = "/root/autodl-tmp/yolov5/yolov5-master"
if not os.path.exists(YOLOV5_LOCAL_DIR):
    raise FileNotFoundError(f"❌ YOLOv5 路径不存在: {YOLOV5_LOCAL_DIR}")

models_path = os.path.join(YOLOV5_LOCAL_DIR, "models")
if not os.path.exists(models_path):
    raise FileNotFoundError(f"❌ 找不到 YOLOv5 的 models 文件夹，请检查路径: {models_path}")

sys.path.insert(0, YOLOV5_LOCAL_DIR)
print(f"✅ 已成功添加 YOLOv5 路径: {YOLOV5_LOCAL_DIR}")

import torch
import cv2
import numpy as np
from PIL import Image

# 有些仓库结构是 /yolov5/yolov5-master，这里也把上一级加一下
YOLOV5_PARENT_DIR = "/root/autodl-tmp/yolov5"
sys.path.insert(0, YOLOV5_PARENT_DIR)

# ================== 导入 YOLOv5 相关模块 ==================
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# ================== 导入 VLM-FO1 相关模块 ==================
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import prepare_inputs, extract_predictions_to_bboxes

# ================== 基本配置 ==================
image_folder = "/root/autodl-tmp/datasets_input/images"  # 输入图像目录
output_folder = "/root/autodl-tmp/datasets_output"       # 输出结果目录
model_path = "/root/autodl-tmp/resources/resources/VLM-FO1_Qwen2.5-VL-3B-v01"
wheel_path = "/root/autodl-tmp/resources/resources/flash_attn-2.8.0+cu124torch2.6-cp311-cp311-linux_x86_64.whl"
yolov5_weights_path = "/root/autodl-tmp/exp8best/weights/best.pt"

os.makedirs(output_folder, exist_ok=True)

# 如果存在 flash_attn 轮子，就静默安装一下（可选）
if os.path.exists(wheel_path):
    print(f"🔧 检测到 flash_attn whl 包，正在安装：{wheel_path}")
    os.system(f"pip install -q {wheel_path}")

# ================== 加载模型 ==================
print("🔥 正在加载 VLM-FO1 模型...")
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer, model, image_processors = load_pretrained_model(
    model_path,
    load_8bit=False,
    load_4bit=False,
    device=device,
)

print("🔥 正在加载 YOLOv5 模型权重...")
yolov5_model = attempt_load(yolov5_weights_path, device=device)
yolov5_model.eval()

# 类别 ID（必须与训练时一致）
fire_label = 0   # 对应火焰
smoke_label = 1  # 对应烟雾

# ================== 工具函数 ==================
def parse_count_from_text(text: str) -> int:
    """从 VLM 的文本输出中解析数量，尽量鲁棒一点"""
    text_lower = text.lower().strip()

    # 明确的否定情况
    if any(w in text_lower for w in ["no ", "no.", "none", "not ", "zero", "not detected", "no fire", "no smoke"]):
        return 0

    # 英文数量词映射
    word_to_num = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "multiple": 2,
        "several": 3,
        "many": 5,
        "a lot": 5,
    }
    for word, num in word_to_num.items():
        if word in text_lower:
            return num

    # 提取阿拉伯数字
    numbers = re.findall(r"\d+", text)
    if numbers:
        try:
            return int(numbers[0])
        except ValueError:
            pass

    # 如果提到了 fire/smoke/flame 等关键词但没给数量，默认 1
    if any(obj in text_lower for obj in ["fire", "smoke", "flame", "burn","burning"]):
        return 1

    # 实在解析不出来就认为 0
    return 0

def get_yolov5_bboxes(model, img_path, conf_thresh=0.3):
    """对单张图片使用 YOLOv5 推理，返回 [x1, y1, x2, y2, conf, cls_id] 列表和原图 shape"""
    img0 = cv2.imread(img_path)
    if img0 is None:
        return [], None

    # letterbox 预处理
    img = letterbox(img0, 640, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]

    # NMS
    pred = non_max_suppression(pred, conf_thresh, 0.45, classes=None, agnostic=False)

    bboxes = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                if float(conf) > conf_thresh:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cls_id = int(cls)
                    bboxes.append([x1, y1, x2, y2, float(conf), cls_id])

    return bboxes, img0.shape

def detect_and_count(object_name_en: str, bbox_list, img_path):
    """
    使用 VLM-FO1 进行目标检测 + 计数
    object_name_en: "fire" 或 "smoke"（英文，用于模板和解析）
    bbox_list: [[x, y, w, h], ...]，给 VLM 的候选区域
    """
    # 检测消息（带 bbox）
    # ✅ 提前格式化 f-string
    detect_prompt = f"请在图像中检测所有的 {object_name_en}。请列出它们在以下区域中的位置（x,y,w,h）：{bbox_list}。如果没有，请返回空列表 []。"
    detect_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_path}},
                {"type": "text", "text": detect_prompt},
            ],
            "bbox_list": bbox_list,
        }
    ]

    # 计数消息（同样使用英文，保持和 parse_count 一致）
    count_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_path}},
                {
                    "type": "text",
                    "text": f"图像中总共有多少个 {object_name_en}？请直接回答一个数字。",
                },
            ],
            "bbox_list": bbox_list, # 保持一致性，虽然计数可能不依赖bbox
        }
    ]

    # 检测推理
    detect_kwargs = prepare_inputs(
        model_path,
        model,
        image_processors,
        tokenizer,
        detect_messages,
        max_tokens=4096,
        top_p=0.05,
        temperature=0.0,
        do_sample=False,
    )
    # --- 超时处理 ---
    def timeout_handler(signum, frame):
        raise TimeoutError("VLM detection 推理超时")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 设置 30 秒超时
    try:
        with torch.inference_mode():
            detect_output_ids = model.generate(**detect_kwargs)
            detect_outputs = tokenizer.decode(
                detect_output_ids[0, detect_kwargs["inputs"].shape[1]:]
            ).strip()
    except TimeoutError:
        print("❌ VLM detection 推理超时，跳过本次检测")
        detect_outputs = "[]"
    finally:
        signal.alarm(0)  # 取消超时
        signal.signal(signal.SIGALRM, old_handler) # 恢复信号处理
    # --- 超时处理结束 ---

    label_to_bboxes = extract_predictions_to_bboxes(detect_outputs, bbox_list)
    # VLM 返回的 label 一般用小写英文
    bboxes = label_to_bboxes.get(object_name_en.lower(), [])

    # 计数推理
    count_kwargs = prepare_inputs(
        model_path,
        model,
        image_processors,
        tokenizer,
        count_messages,
        max_tokens=512, # 计数不需要太长
        top_p=0.95,
        temperature=0.7,
        do_sample=True,
    )
    # --- 超时处理 ---
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(15)  # 设置 15 秒超时
    try:
        with torch.inference_mode():
            count_output_ids = model.generate(**count_kwargs)
            count_outputs = tokenizer.decode(
                count_output_ids[0, count_kwargs["inputs"].shape[1]:]
            ).strip()
    except TimeoutError:
        print("❌ VLM count 推理超时，跳过本次计数")
        count_outputs = "0"
    finally:
        signal.alarm(0)  # 取消超时
        signal.signal(signal.SIGALRM, old_handler) # 恢复信号处理
    # --- 超时处理结束 ---

    count = parse_count_from_text(count_outputs)
    return bboxes, count

def detect_count_in_single_bbox(object_name_en: str, x1, y1, x2, y2, img_path):
    """
    使用 VLM-FO1 对单个 YOLO 框区域进行计数
    object_name_en: "fire" 或 "smoke"
    x1, y1, x2, y2: YOLO 框的坐标
    """
    # 计数消息（只针对单个 bbox）
    count_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_path}},
                {
                    "type": "text",
                    "text": f"在区域 [{x1}, {y1}, {x2 - x1}, {y2 - y1}] 内，总共有多少个 {object_name_en}？请直接回答一个数字。",
                },
            ],
            "bbox_list": [[x1, y1, x2 - x1, y2 - y1]], # 只包含这一个框
        }
    ]

    # 计数推理
    count_kwargs = prepare_inputs(
        model_path,
        model,
        image_processors,
        tokenizer,
        count_messages,
        max_tokens=512,
        top_p=0.95,
        temperature=0.7,
        do_sample=True,
    )
    # --- 超时处理 ---
    def timeout_handler(signum, frame):
        raise TimeoutError("VLM single bbox count 推理超时")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(15)  # 设置 15 秒超时
    try:
        with torch.inference_mode():
            count_output_ids = model.generate(**count_kwargs)
            count_outputs = tokenizer.decode(
                count_output_ids[0, count_kwargs["inputs"].shape[1]:]
            ).strip()
    except TimeoutError:
        print(f"❌ VLM 对区域 [{x1}, {y1}, {x2}, {y2}] 的计数推理超时")
        count_outputs = "0"
    finally:
        signal.alarm(0)  # 取消超时
        signal.signal(signal.SIGALRM, old_handler) # 恢复信号处理
    # --- 超时处理结束 ---

    count = parse_count_from_text(count_outputs)
    return count


def vlm_xywh_to_xyxy(bbox_xywh):
    """VLM 的 [x, y, w, h] 转为 [x1, y1, x2, y2]"""
    if len(bbox_xywh) < 4:
        return None
    x, y, w, h = bbox_xywh[:4]
    if w <= 0 or h <= 0:
        return None
    return int(x), int(y), int(x + w), int(y + h)


def generate_grid_bboxes(w, h, grid_size=3):
    """生成 grid_size x grid_size 网格 bbox（xywh 格式）"""
    bboxes = []
    cell_w, cell_h = w // grid_size, h // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            x = j * cell_w
            y = i * cell_h
            w_ = cell_w if j < grid_size - 1 else w - x
            h_ = cell_h if i < grid_size - 1 else h - y
            bboxes.append([x, y, w_, h_])
    return bboxes


def get_detailed_description(object_name_en: str, img_path: str) -> str: # ✅ 移除了 bbox_list_for_desc 参数
    """
    使用 VLM 生成英文详细描述（基于整图）
    object_name_en: "fire" 或 "smoke"（英文，用于提示）
    """
    # 获取图片尺寸，用于构建整图 bbox
    img_temp = Image.open(img_path)
    w, h = img_temp.size
    # 使用整图作为描述区域
    whole_image_bbox_list = [[0, 0, w, h]]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_path}},
                {
                    "type": "text",
                    "text": f"Please describe the {object_name_en} in this image in detail. If it does not exist, clearly state that there is no {object_name_en}.",
                },
            ],
            # ✅ 使用整图 bbox
            "bbox_list": whole_image_bbox_list,
        }
    ]

    kwargs = prepare_inputs(
        model_path,
        model,
        image_processors,
        tokenizer,
        messages,
        max_tokens=4096,
        top_p=0.9,
        temperature=0.7,
        do_sample=True,
    )

    # --- 超时处理 ---
    def timeout_handler(signum, frame):
        raise TimeoutError("VLM description 推理超时")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 设置 30 秒超时
    try:
        with torch.inference_mode():
            output_ids = model.generate(**kwargs)
            outputs = tokenizer.decode(
                output_ids[0, kwargs["inputs"].shape[1]:]
            ).strip()
    except TimeoutError:
        print("❌ VLM description 推理超时，返回默认描述")
        outputs = f"[{object_name_en}描述超时]"
    finally:
        signal.alarm(0)  # 取消超时
        signal.signal(signal.SIGALRM, old_handler) # 恢复信号处理
    # --- 超时处理结束 ---

    return outputs


# ================== 主流程 ==================
def main():
    all_logs = []

    if not os.path.isdir(image_folder):
        raise NotADirectoryError(f"❌ 图像输入目录不存在: {image_folder}")

    # 按文件名排序处理
    img_files = sorted(os.listdir(image_folder))

    if not img_files:
        print(f"⚠️ 输入目录中未找到任何图片文件: {image_folder}")
        return

    for img_filename in img_files:
        if not img_filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            continue

        img_path = os.path.join(image_folder, img_filename)
        print("\n==============================")
        print(f"开始处理图像: {img_path}")

        # Step 1: YOLO 推理
        yolov5_results, orig_shape = get_yolov5_bboxes(
            yolov5_model, img_path, conf_thresh=0.2
        )
        if orig_shape is None:
            print(f"⚠️ 无法读取图像，已跳过: {img_path}")
            continue

        # Step 2: 构建给 VLM 的 bbox_list
        if yolov5_results:
            print(f"✅ YOLO 检测到 {len(yolov5_results)} 个候选框")
            bbox_list = [
                [x1, y1, (x2 - x1), (y2 - y1)]
                for x1, y1, x2, y2, _, _ in yolov5_results
            ]
        else:
            img_temp = Image.open(img_path)
            w, h = img_temp.size
            grid_size = 3
            bbox_list = generate_grid_bboxes(w, h, grid_size=grid_size)
            print(
                f"🔍 YOLO 未检测到任何框，将图像划分为 {grid_size}x{grid_size} 网格（共 {len(bbox_list)} 个区域）提交给 VLM"
            )

        # Step 3: 使用 VLM 检测火焰 & 烟雾 (仅在 YOLO 无框时)
        if not yolov5_results:
            try:
                vlm_fire_bboxes, fire_count = detect_and_count("fire", bbox_list, img_path)
                vlm_smoke_bboxes, smoke_count = detect_and_count("smoke", bbox_list, img_path)
            except Exception as e:
                print(f"❌ VLM 推理出错，该图片跳过。错误信息: {e}")
                continue
        else:
            # YOLO 有框时，不进行全局检测，只进行单框计数
            vlm_fire_bboxes, vlm_smoke_bboxes = [], []
            # 计数基于 YOLO 框的数量
            fire_count = sum(1 for _, _, _, _, _, cls_id in yolov5_results if cls_id == fire_label)
            smoke_count = sum(1 for _, _, _, _, _, cls_id in yolov5_results if cls_id == smoke_label)

        # Step 4: 融合为绘图用的框
        fused_bboxes = []

        if yolov5_results:
            # 使用 YOLO 的类别和置信度，对外显示英文
            for x1, y1, x2, y2, conf, cls_id in yolov5_results:
                if cls_id == fire_label:
                    label_en = "Fire"
                    object_name_en = "fire"
                elif cls_id == smoke_label:
                    label_en = "Smoke"
                    object_name_en = "smoke"
                else:
                    label_en = None
                    object_name_en = None

                if label_en:
                    # ✅ 新增：对单个 YOLO 框进行 VLM 计数验证
                    vlm_count_in_bbox = detect_count_in_single_bbox(object_name_en, x1, y1, x2, y2, img_path)
                    # 如果 VLM 计数 > 1，则提升置信度
                    final_conf = 0.9 if vlm_count_in_bbox > 1 else conf
                    print(f"  - YOLO 框 [{x1}, {y1}, {x2}, {y2}] ({label_en})，VLM 计数: {vlm_count_in_bbox}，置信度: {conf:.2f} -> {final_conf:.2f}")

                    fused_bboxes.append(
                        {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "conf": final_conf,
                            "label": label_en,
                        }
                    )
        elif fire_count > 0 or smoke_count > 0:
            # 网格模式：仅使用 VLM 检测结果，置信度统一为 0.5
            for bbox in vlm_fire_bboxes:
                xyxy = vlm_xywh_to_xyxy(bbox)
                if xyxy:
                    fused_bboxes.append(
                        {
                            "x1": xyxy[0],
                            "y1": xyxy[1],
                            "x2": xyxy[2],
                            "y2": xyxy[3],
                            "conf": 0.5,
                            "label": "Fire",
                        }
                    )
            for bbox in vlm_smoke_bboxes:
                xyxy = vlm_xywh_to_xyxy(bbox)
                if xyxy:
                    fused_bboxes.append(
                        {
                            "x1": xyxy[0],
                            "y1": xyxy[1],
                            "x2": xyxy[2],
                            "y2": xyxy[3],
                            "conf": 0.5,
                            "label": "Smoke",
                        }
                    )

        # Step 5: 绘制框并保存结果图
        image = Image.open(img_path).convert("RGB")
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        for box in fused_bboxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            conf, label_en = float(box["conf"]), box["label"]

            # 火焰: 绿色, 烟雾: 红色
            color = (0, 255, 0) if label_en == "Fire" else (0, 0, 255)

            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img_np,
                f"{label_en} {conf:.2f}",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        output_path = os.path.join(
            output_folder, os.path.splitext(img_filename)[0] + "_result.jpg"
        )
        cv2.imwrite(output_path, img_np)

        # Step 6: 生成火焰/烟雾的英文详细描述（基于整图）
        # ✅ 调用方式简化，不再传入 bbox_list
        try:
            fire_desc = get_detailed_description("fire", img_path)
        except Exception as e:
            fire_desc = f"[Fire description failed: {e}]"

        try:
            smoke_desc = get_detailed_description("smoke", img_path)
        except Exception as e:
            smoke_desc = f"[Smoke description failed: {e}]"

        # Step 7: 构建 & 输出日志（包含详细描述）
        # ✅ 移除 desc_path 定义
        # desc_path = os.path.join(output_folder, os.path.splitext(img_filename)[0] + "_description.txt")

        log_lines = [
            "==============================",
            f"Processing image path: {img_path}",
            f"Fire count (detection result): {fire_count}",
            f"Smoke count (detection result): {smoke_count}",
            # ✅ 直接在日志中包含详细描述
            f"Fire Description: {fire_desc}",
            f"Smoke Description: {smoke_desc}",
            f"Detection visualization result saved to: {output_path}",
            # ✅ 移除 "文字描述文件已保存至..." 这一行
            # f"Text description file saved to: {desc_path}",
        ]

        if not fused_bboxes and fire_count == 0 and smoke_count == 0:
            log_lines.append("No fire or smoke detected in this image.")

        current_log = "\n".join(log_lines)
        all_logs.append(current_log)
        print(current_log)

        # Step 8: 移除保存描述文件的逻辑
        # with open(desc_path, "w", encoding="utf-8") as f:
        #     f.write(f"火焰详细描述:\n{fire_desc}\n\n")
        #     f.write(f"烟雾详细描述:\n{smoke_desc}\n")

    # ================== 汇总日志 ==================
    print("\n\n" + "=" * 70)
    print("✅ 所有图像处理完成！完整汇总日志如下：")
    print("=" * 70)
    full_log = "\n\n".join(all_logs)
    print(full_log)

    total_log_path = os.path.join(output_folder, "total_processing_log.txt")
    with open(total_log_path, "w", encoding="utf-8") as f:
        f.write(full_log)
    print(f"\n📄 完整汇总日志已保存至: {total_log_path}")


if __name__ == "__main__":
    main()