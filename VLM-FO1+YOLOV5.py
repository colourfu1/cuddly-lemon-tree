# 在VLM-FO1.py代码基础上(计数)，我想让他和yolov5的检测结果进行一个加权判断融合成最终结果，
#逻辑如下：如果yolov5有烟或者火的检测框，并且对应的火和烟的数量大于0，则增加yolov5检测框的置信度到0.9.
#如果yolov5没有检测出任何检测框，但是对应的烟和火数量大于0，则采用可视化结果中的图片

#输出成果物
#1.融合后的图像：这些图像将保存在 output_folder 文件夹中，图像文件名格式为 {原图像文件名}_fused_result.jpg
#2.控制台输出：  A.处理图像的路径  B.检测到的火和烟的数量。 C.保存融合后结果图像的路径。


import os
import torch
from PIL import Image
from detect_tools.upn import UPNWrapper
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import prepare_inputs, draw_bboxes_and_save, extract_predictions_to_bboxes
from vlm_fo1.task_templates import OD_template
import re
import cv2
import numpy as np

# 路径配置
upn_ckpt_path = "./resources/upn_large.pth"
model_path = './resources/VLM-FO1_Qwen2.5-VL-3B-v01'
image_folder = "demo/images"  # 包含多张图片的文件夹路径
output_folder = "demo/results"  # 结果保存文件夹
yolov5_weights_path = "./yolov5/weights/yolov5s.pt"  # YOLOv5模型权重路径

# 初始化模型
upn_model = UPNWrapper(upn_ckpt_path)
tokenizer, model, image_processors = load_pretrained_model(model_path)

# 加载YOLOv5模型
yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolov5_weights_path)

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

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
                fused_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], 0.9])  # 增加置信度到0.9

    return fused_bboxes

# YOLOv5类别标签（假设火的标签是0，烟的标签是1）
fire_label = 0
smoke_label = 1

# 遍历文件夹中的所有图像
for img_filename in os.listdir(image_folder):
    if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(image_folder, img_filename)
        print(f"处理图像: {img_path}")

        # 加载图像
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)

        # 生成边界框提议
        fine_grained_proposals = upn_model.inference(img_pil)
        fine_grained_filtered_proposals = upn_model.filter(
            fine_grained_proposals, min_score=0.3
        )
        bbox_list = fine_grained_filtered_proposals['original_xyxy_boxes'][0][:100]

        # 分别检测火和烟
        fire_bboxes, fire_count = detect_and_count("fire", bbox_list, img_path)
        smoke_bboxes, smoke_count = detect_and_count("smoke", bbox_list, img_path)

        # 合并边界框字典
        all_vlm_bboxes = {}
        if fire_bboxes:
            all_vlm_bboxes["fire"] = fire_bboxes.get("fire", [])
        if smoke_bboxes:
            all_vlm_bboxes["smoke"] = smoke_bboxes.get("smoke", [])

        # 使用YOLOv5进行检测
        yolov5_results = yolov5_model(img_path)

        # 融合结果
        fused_bboxes = fuse_results(yolov5_results, fire_bboxes, smoke_bboxes, fire_count, smoke_count)

        # 将融合后的边界框绘制到图像上
        for bbox in fused_bboxes:
            x1, y1, x2, y2, conf = map(int, bbox)
            label = "fire" if (x2 - x1) * (y2 - y1) < 500 else "smoke"
            color = (0, 255, 0) if label == "fire" else (0, 0, 255)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_np, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 构建输出文件路径
        output_filename = os.path.splitext(img_filename)[0] + "_fused_result.jpg"
        output_path = os.path.join(output_folder, output_filename)

        # 保存融合后的图像
        cv2.imwrite(output_path, img_np)

        # 输出结果
        print(f"火的数量: {fire_count}")
        print(f"烟的数量: {smoke_count}")
        print(f"融合后结果已保存到: {output_path}")
