#使用 YOLOv5 进行目标检测。使用 VLM-FO1 对火和烟进行详细的区域描述和计数。

# 结果融合：
# 如果 YOLOv5 检测到火或烟，并且对应的火或烟数量大于 0，则增加这些检测框的置信度到 0.9，并输出详细描述。
# 如果 YOLOv5 没有检测出任何检测框，但 VLM-FO1 检测到火或烟，则使用带有边界框的 VLM-FO1 的图片。

import os
import torch
from PIL import Image
import cv2
import numpy as np
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import (
    prepare_inputs,
    draw_bboxes_and_save,
    extract_predictions_to_bboxes,
)
from detect_tools.upn import UPNWrapper
from vlm_fo1.task_templates import OD_template
import re

# 配置路径
image_folder = "demo/images"  # 输入图像文件夹路径
output_folder = "demo/results"  # 输出结果文件夹路径
upn_ckpt_path = "./resources/upn_large.pth"
model_path = './resources/VLM-FO1_Qwen2.5-VL-3B-v01'  # 模型路径
yolov5_weights_path = "./yolov5/weights/yolov5s.pt"  # YOLOv5 模型权重路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 加载 UPN 模型和 VLM-FO1 模型及分词器
upn_model = UPNWrapper(upn_ckpt_path)
tokenizer, model, image_processors = load_pretrained_model(model_path)
yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolov5_weights_path)

# YOLOv5类别标签（假设火的标签是0，烟的标签是1）
fire_label = 0
smoke_label = 1

# 函数：获取 YOLOv5 的边界框
def get_yolov5_bboxes(results):
    bboxes = []
    for *box, conf, cls in results.xyxy[0]:
        if conf > 0.3:  # 考虑置信度大于0.3的检测结果
            x1, y1, x2, y2 = map(int, box)
            bboxes.append([x1, y1, x2, y2])
    return bboxes

# 检测和计数函数
def detect_and_count(object_name, bbox_list, img_path):
    # 检测任务
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
    # 计数任务
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
    # 执行检测
    detect_kwargs = prepare_inputs(
        model_path, model, image_processors, tokenizer, detect_messages,
        max_tokens=4096, top_p=0.05, temperature=0.0, do_sample=False
    )
    with torch.inference_mode():
        detect_output_ids = model.generate(**detect_kwargs)
        detect_outputs = tokenizer.decode(
            detect_output_ids[0, detect_kwargs['inputs'].shape[1]:]
        ).strip()
    # 提取边界框
    bboxes = extract_predictions_to_bboxes(detect_outputs, bbox_list)
    # 执行计数
    count_kwargs = prepare_inputs(
        model_path, model, image_processors, tokenizer, count_messages,
        max_tokens=4096, top_p=0.05, temperature=0.0, do_sample=False
    )
    with torch.inference_mode():
        count_output_ids = model.generate(**count_kwargs)
        count_outputs = tokenizer.decode(
            count_output_ids[0, count_kwargs['inputs'].shape[1]:]
        ).strip()
    # 提取数字
    ans = re.sub(r'<region\d+>', '', count_outputs)
    numbers = re.findall(r'(?<!region)\d+', ans)
    count = int(numbers[0]) if numbers else 0
    return bboxes, count

# 融合结果函数
def fuse_results(yolov5_results, vlm_fire_bboxes, vlm_smoke_bboxes, fire_count, smoke_count):
    fused_bboxes = []
    # 检查YOLOv5是否有火和烟的检测框
    for *box, conf, cls in yolov5_results.xyxy[0]:
        if conf > 0.15:  # 只考虑置信度大于0.15的检测结果
            label = int(cls)
            if (label == fire_label and fire_count > 0) or (label == smoke_label and smoke_count > 0):
                fused_bboxes.append([*box, 0.9])  # 增加置信度到0.9
            else:
                fused_bboxes.append([*box, conf])
    # 如果YOLOv5没有检测出任何检测框，但VLM-FO1检测到火和烟，则使用VLM-FO1的边界框
    if len(fused_bboxes) == 0 and (fire_count > 0 or smoke_count > 0):
        for obj_type, bboxes in {"fire": vlm_fire_bboxes, "smoke": vlm_smoke_bboxes}.items():
            for bbox in bboxes:
                fused_bboxes.append([*bbox, 0.9])  # 增加置信度到0.9
    return fused_bboxes

# 获取详细描述的函数
def get_detailed_description(object_name, img_path):
    description_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_path}},
                {"type": "text", "text": f"Describe the {object_name} in this image."},
            ],
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

# 遍历文件夹中的所有图像
for img_filename in os.listdir(image_folder):
    if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(image_folder, img_filename)
        print(f"处理图像: {img_path}")
        # 加载图像
        image = Image.open(img_path)
        # 获取 YOLOv5 的边界框
        yolov5_results = yolov5_model(image)
        yolov5_bboxes = get_yolov5_bboxes(yolov5_results)

        # 生成边界框列表
        bbox_list = []
        for bbox in yolov5_bboxes:
            bbox_list.append([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])

        # 获取 VLM-FO1 的检测和计数结果
        vlm_fire_bboxes, fire_count = detect_and_count("fire", bbox_list)
        vlm_smoke_bboxes, smoke_count = detect_and_count("smoke", bbox_list)

        # 融合边界框结果
        fused_bboxes = fuse_results(yolov5_results, vlm_fire_bboxes, vlm_smoke_bboxes, fire_count, smoke_count)

        # 绘制融合后的边界框
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for bbox in fused_bboxes:
            x1, y1, x2, y2, conf = map(int, [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1], bbox[4]])
            label = "Fire" if (x2 - x1) * (y2 - y1) > 0 else "Smoke"
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0) if label == "Fire" else (0, 0, 255), 2)
            cv2.putText(img_np, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if label == "Fire" else (0, 0, 255), 2)

        # 构建输出文件路径
        output_filename = os.path.splitext(img_filename)[0] + "_result.jpg"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, img_np)

        # 获取详细描述
        if fire_count > 0 or smoke_count > 0:
            detailed_description_fire = get_detailed_description("fire", img_path) if fire_count > 0 else ""
            detailed_description_smoke = get_detailed_description("smoke", img_path) if smoke_count > 0 else ""

            # 打印详细描述
            if detailed_description_fire:
                print(f"Fire Description: {detailed_description_fire}")
            if detailed_description_smoke:
                print(f"Smoke Description: {detailed_description_smoke}")

            # 保存详细描述到文件
            description_filename = os.path.splitext(img_filename)[0] + "_description.txt"
            description_path = os.path.join(output_folder, description_filename)
            with open(description_path, "w") as f:
                if detailed_description_fire:
                    f.write(f"Fire Description: {detailed_description_fire}\n")
                if detailed_description_smoke:
                    f.write(f"Smoke Description: {detailed_description_smoke}\n")

        print(f"火的数量: {fire_count}")
        print(f"烟的数量: {smoke_count}")
        print(f"结果已保存到: {output_path}")

        # 如果既没有YOLOv5检测到的边界框，也没有VLM-FO1检测到的边界框
        if len(fused_bboxes) == 0 and fire_count == 0 and smoke_count == 0:
            print("未检测到任何物体。")
