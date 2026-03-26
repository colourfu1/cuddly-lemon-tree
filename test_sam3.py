import os
from pathlib import Path
from ultralytics.models.sam import SAM3SemanticPredictor


def batch_predict_sam3(input_dir, output_dir, text_prompts, model_path="/root/autodl-tmp/ultralytics-main/ultralytics/weights/sam3.pt"):
    """
    使用 SAM3 对文件夹中的所有图片进行文本引导的语义分割
    """
    # 1. 初始化预测器配置
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=model_path,
        half=True,  # 使用 FP16 加速（如果 GPU 支持）
        save=True,  # 自动保存渲染后的图片
        project=output_dir,  # 保存的项目根目录
        name="sam3_results",  # 保存的子目录名
        exist_ok=True  # 如果目录存在不报错
    )

    # 初始化预测器
    predictor = SAM3SemanticPredictor(overrides=overrides)

    # 2. 检查并获取所有图片文件
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"错误: 找不到输入文件夹 {input_dir}")
        return

    # 支持常见的图片格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions]

    if not image_files:
        print(f"在 {input_dir} 中没有找到图片文件。")
        return

    print(f"找到 {len(image_files)} 张图片，开始处理...")

    # 3. 循环处理每一张图片
    for img_file in image_files:
        img_path = str(img_file)
        print(f"正在处理: {img_file.name}")

        try:
            # 设置当前图片（计算图像特征向量）
            predictor.set_image(img_path)

            # 使用文本提示进行推理
            # 注意：SAM3SemanticPredictor 会根据之前 set_image 的内容进行推理
            results = predictor(text=text_prompts)

            # 如果你想手动处理结果（比如获取掩码数组）：
            # for result in results:
            #     masks = result.masks  # 分割掩码
            #     names = result.names  # 类别名称

        except Exception as e:
            print(f"处理图片 {img_file.name} 时出错: {e}")

    print(f"所有任务完成！结果保存在: {os.path.join(output_dir, 'sam3_results')}")


if __name__ == "__main__":
    # --- 用户配置区 ---
    # 图片所在的文件夹路径
    INPUT_FOLDER = "/root/autodl-tmp/datasets/images/test"
    # 结果保存的文件夹路径
    OUTPUT_FOLDER = "/root/autodl-tmp/ultralytics-main/ultralytics/runs/yoloe_batch"
    # 你想要检测的物体（文本提示）
    PROMPTS = ["fire", "smoke"]
    # 模型权重文件（如果本地没有，Ultralytics会自动下载）
    MODEL_CHECKPOINT = "/root/autodl-tmp/ultralytics-main/ultralytics/weights/sam3.pt"
    # ----------------

    # 执行批量预测
    batch_predict_sam3(INPUT_FOLDER, OUTPUT_FOLDER, PROMPTS, MODEL_CHECKPOINT)