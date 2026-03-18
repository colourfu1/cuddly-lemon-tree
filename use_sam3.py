# 最简单的使用方式
from ultralytics import YOLO

# 加载模型
model = YOLO('sam3_b.pt')

# 直接使用开放词汇检测
results = model.predict(
    source='your_image.jpg',
    prompt={'text': ['fire', 'smoke']},
    conf=0.3,
    iou=0.5,
    retina_masks=True
)

# 处理结果
for result in results:
    result.show()  # 显示结果
    if result.masks:
        print(f"检测到 {len(result.masks)} 个掩码")