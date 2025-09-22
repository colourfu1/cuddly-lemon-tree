import torch
from pathlib import Path
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from transformers import AutoModelForVision2Seq, AutoProcessor

# 设置设备（CUDA 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载 YOLOv5 模型
yolo_weights_path = "/path/to/your/best.pt"  # 替换为你的 best.pt 文件路径
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights_path)
yolo_model.to(device)

# 2. 加载 SAM 模型并加载预训练权重
model_type_sam = "vit_l"
checkpoint_path_sam = "/path/to/sam_vit_l_0b3195.pth"  # 替换为你的文件路径
sam = sam_model_registry[model_type_sam](checkpoint=checkpoint_path_sam)
sam.to(device)
predictor = SamPredictor(sam)

# 3. 加载 qw-vl 模型和处理器
model_name_qw_vl = "your-qw-vl-model-name"  # 替换为你的 qw-vl 模型名称或路径
processor_qw_vl = AutoProcessor.from_pretrained(model_name_qw_vl)
model_qw_vl = AutoModelForVision2Seq.from_pretrained(model_name_qw_vl).to(device)

def process_frame(frame):
    # 1. 使用 YOLOv5 进行目标检测
    results = yolo_model(frame)
    detections = results.xyxy[0].cpu().numpy()

    # 处理每个检测框
    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)

        # 2. 裁剪检测框内的图像区域
        cropped_img = frame[y1:y2, x1:x2]
        if cropped_img.size == 0:
            continue

        # 3. 使用 SAM 进行分割
        predictor.set_image(cropped_img)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([0, 0, x2 - x1, y2 - y1]),
            multimask_output=False
        )

        if masks.shape[0] == 0:
            continue

        mask = masks[0]

        # 4. 将分割掩码应用于裁剪后的图像
        segmented_img = cropped_img.copy()
        segmented_img[~mask] = 0

        # 5. 调整图像大小以适应 qw-vl 模型的输入要求
        input_size_qw_vl = processor_qw_vl.size['height']
        resized_segmented_img = cv2.resize(segmented_img, (input_size_qw_vl, input_size_qw_vl))

        # 6. 使用 qw-vl 模型进行推理
        inputs_qw_vl = processor_qw_vl(resized_segmented_img, return_tensors="pt").to(device)
        generated_ids = model_qw_vl.generate(**inputs_qw_vl)
        scene_description = processor_qw_vl.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 7. 绘制分割区域和场景描述
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, scene_description, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# 示例：处理视频帧
cap = cv2.VideoCapture(0)  # 使用默认摄像头，可以替换为视频文件路径
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame)
    cv2.imshow('Processed Frame', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
