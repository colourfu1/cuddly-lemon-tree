#利用了计数功能


import os
import torch
from PIL import Image
from detect_tools.upn import UPNWrapper
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import prepare_inputs, draw_bboxes_and_save, extract_predictions_to_bboxes
from vlm_fo1.task_templates import OD_template
import re

# 路径配置
upn_ckpt_path = "./resources/upn_large.pth"
model_path = './resources/VLM-FO1_Qwen2.5-VL-3B-v01'
image_folder = "demo/images"  # 包含多张图片的文件夹路径
output_folder = "demo/results"  # 结果保存文件夹

# 初始化模型
upn_model = UPNWrapper(upn_ckpt_path)
tokenizer, model, image_processors = load_pretrained_model(model_path)

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

# 遍历文件夹中的所有图像
for img_filename in os.listdir(image_folder):
    if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(image_folder, img_filename)
        print(f"处理图像: {img_path}")

        # 加载图像
        img_pil = Image.open(img_path).convert("RGB")

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
        all_bboxes = {}
        if fire_bboxes:
            all_bboxes["fire"] = fire_bboxes.get("fire", [])
        if smoke_bboxes:
            all_bboxes["smoke"] = smoke_bboxes.get("smoke", [])

        # 构建输出文件路径
        output_filename = os.path.splitext(img_filename)[0] + "_result.jpg"
        output_path = os.path.join(output_folder, output_filename)

        # 绘制并保存结果
        draw_bboxes_and_save(
            image=img_pil,
            fo1_bboxes=all_bboxes,
            output_path=output_path
        )

        # 输出结果
        print(f"火的数量: {fire_count}")
        print(f"烟的数量: {smoke_count}")
        print(f"可视化结果已保存到: {output_path}")
