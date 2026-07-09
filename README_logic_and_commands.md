# 🔥 多模型协同伪标签生成管线 (Pipeline) 新人上手指南

> **文档版本**：v1.0 | **适用对象**：新入职算法工程师 / 数据标注开发  
> **核心思想**：粗标(GDINO) -> 提纯(课程学习YOLO) -> 路由分发 -> 补漏(SAM3) -> 终审(VLM-FO1) -> 合并  
> **目标**：利用多模型协同，零人工干预生成极高质量的 YOLO 训练标签。

---

## 🗺️ 一、 核心数据流向图

整个管线分为 8 个标准步骤，数据流向如下：

```text
[原始图像] 
   │
   ▼
① Grounding DINO (零样本粗标，生成带质量评估的 Sidecar JSON)
   │
   ▼
② 课程学习数据集构建 (按质量划分 Stage1/2/3，无验证集泄露)
   │
   ▼
③ YOLO 课程训练 (串行训练，得到 YOLO Teacher 强模型)
   │
   ▼
④ YOLO 推理与路由 (对比 GDINO 标签，分发: Keep / Send_SAM / Send_VLM)
   │
   ├──────────────────────┐
   ▼                      ▼
⑤ SAM3 补漏 (针对 Send_SAM)   ⑥ VLM-FO1 终审 (针对所有争议/补漏候选框)
   │                      │
   └──────────┬───────────┘
              ▼
         ⑦ 标签合并 (VLM 过滤 YOLO，SAM 兜底，生成最终干净标签)
              │
              ▼
         [最终高质量 YOLO 数据集] -> ⑧ 训练最终业务模型


二、 新人必读：全局路径替换 (极其重要)
代码和配置文件中写死了大量如 /root/autodl-tmp/... 的绝对路径。在执行任何命令前，请务必使用 IDE 的全局搜索功能（Ctrl+Shift+F），将以下路径替换为您本地的实际路径！
原始占位路径替换为自己的路径，或者和我路径保持一致
说明
/root/autodl-tmp/datasets/images/train
/your/local/path/images
原始图像目录
/root/autodl-tmp/grounding-dino-base
/your/local/path/gdino_weights
GDINO 模型权重目录
/root/autodl-tmp/ultralytics-main/...
/your/local/path/ultralytics
YOLO 框架及权重目录
/root/autodl-tmp/VLM-FO1-main
/your/local/path/VLM-FO1
VLM-FO1 代码及权重目录

 三、 标准执行流程 (SOP)
Step 1: Grounding DINO 零样本粗标
目的：利用 GDINO 对全量图像进行推理，生成包含多 Prompt 支持度、TTA 稳定性等质量指标的 Sidecar JSON。
输入：原始图像目录。
输出：sidecar/*.json。
# 1. 先修改配置文件中的 image_dir, output_dir, model_dir
# 文件：gdino_best_box_v2_config_example_repair0615-5.json

# 2. 执行推理
python grounding_dino_best_box_selector_v2_repair.py \
    --config gdino_best_box_v2_config_example_repair0615-5.json \
    --verbose

Step 2: 构建课程学习数据集 (无泄露版)
目的：读取 Step1 的 JSON，计算质量分（course_q），划分 Train/Val，并按阈值生成 Stage1/2/3 的 YOLO 格式数据集。
输入：原始图像、Step1 的 sidecar 目录。
输出：stage1/, stage2/, stage3/, holdout_val/ 等 YOLO 数据集。
# 1. 修改配置文件中的 paths 节点
# 文件：curriculum_config_no_leak_example0611.json

# 2. 执行构建
python build_gdino_curriculum_no_leak0611.py \
    --config curriculum_config_no_leak_example0611.json

Step 3: YOLO 课程学习串行训练
目的：使用 Ultralytics API，依次用 Stage1 -> Stage2 -> Stage3 的数据集训练 YOLO，后一阶段继承前一阶段的 best.pt。
输入：Step2 生成的数据集。
输出：最终的 YOLO Teacher 权重 (best.pt)。
# 复用 Step2 的配置文件，读取其中的 train_example 节点
python train_ultralytics_curriculum_aligned.py \
    --config curriculum_config_no_leak_example0611.json

Step 4: YOLO 推理与对比路由 (Routing)
目的：用 Step3 训练好的 YOLO 进行推理，并与 GDINO 的标签进行 IoU 对比。根据匹配度将图片路由为：keep_yolo, send_sam, send_vlm_direct。
输入：YOLO 权重、图像目录、GDINO 生成的 YOLO 标签。
输出：pred_labels/, report/vlm_consensus_candidates.csv。
# ⚠️ 注意替换 --weights, --images, --ref-labels, --ref-meta-dir, --out 的路径
python /root/autodl-tmp/pseudo_label_pipeline_v2/v2-plus/yolo_label_compare_routing_complete_vlm_consensus_dino_risk_support_source_replacement.py \
  --weights /root/autodl-tmp/GDbetter0606/result/YOLOteacher1_0615/train_runs/yolo26s_curriculum/stage3/weights/best.pt \
  --images /root/autodl-tmp/datasets/images/train \
  --ref-labels /root/autodl-tmp/GDbetter0606/result/YOLOteacher1_0615/stage3/train/labels \
  --ref-meta-dir /root/autodl-tmp/GDbetter0606/result/step1_GD0615/sidecar \
  --out /root/autodl-tmp/GDbetter0606/result/YOLOteacher1_0615/Step2_YOLOcompare \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.70 \
  --match-iou 0.30 \
  --keep-thr 0.85 \
  --sam-thr 0.20 \
  --review-iou-thr 0.50 \
  --audit-risk-thr 0.55 \
  --audit-sample-rate 0.00 \
  --export-vlm-consensus-candidates \
  --export-routes


Step 5: SAM3 选择性补漏
目的：针对 Step4 中路由为 send_sam 的图片，使用 SAM3 进行分割，提取掩码并转换为 YOLO 候选框。
输入：Step4 的 compare_manifest.csv、图像目录。
输出：SAM 预测标签、report/vlm_after_sam_candidates.csv。

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python /root/autodl-tmp/GDbetter0606/sam3_selective_generate_labels_vlm_consensus_source_replacement.py \
  --images-dir /root/autodl-tmp/datasets/images/train \
  --output-dir /root/autodl-tmp/GDbetter0606/result/step3_SAM0615-2 \
  --manifest-csv /root/autodl-tmp/GDbetter0606/result/Step2_YOLOcompare0615/report/compare_manifest.csv \
  --include-routes send_sam \
  --route-column stage1_route \
  --key-column sample_key \
  --image-path-column image_path \
  --model-path /root/autodl-tmp/ultralytics-main/ultralytics/weights/sam3.pt \
  --predictor-imgsz 512 \
  --half \
  --max-raw-masks-per-class 32 \
  --export-vlm-after-sam-candidates


Step 6: VLM-FO1 统一终审
目的：将 Step4 和 Step5 导出的所有“存疑”候选框裁剪出来，送入 VLM-FO1（视觉大模型）进行真伪判断。
输入：Step4 和 Step5 生成的 candidates CSV。
输出：vlm_box_verdicts.csv (包含每个框的最终判决)。
# ⚠️ 确保已下载 VLM-FO1 权重及代码仓库，并修改 --model-path 和 --code-dir
python /root/autodl-tmp/GDbetter0606/vlm_box_review_unified_fo1_source_replacement.py\
  --candidate-csvs \
    /root/autodl-tmp/GDbetter0606/result/Step2_YOLOcompare0615/report/vlm_consensus_candidates.csv \
    /root/autodl-tmp/GDbetter0606/result/step3_SAM0615-2/report/vlm_after_sam_candidates.csv \
  --output-dir /root/autodl-tmp/GDbetter0606/result/step4-VLM0615 \
  --mode fo1 \
  --decision-policy filter \
  --reject-score-thr 0.35 \
  --model-path /root/autodl-tmp/VLM-FO1-main/resources/VLM-FO1_Qwen2.5-VL-3B-v01 \
  --code-dir /root/autodl-tmp/VLM-FO1-main \
  --use-autocast \
  --continue-on-error
  --save-crops \   这个参数可写可不写，写上就会保存可视化图片，但是会变得慢很多


Step 7: 最终标签合并
目的：根据 VLM 的判决清洗标签。YOLO 框被拒绝则剔除；若 YOLO 框全被拒绝，则用 SAM3 且被 VLM 保留的框进行兜底。
输入：YOLO 预测标签、VLM 判决 CSV、SAM 候选 CSV。
输出：labels/ (最终可直接用于训练的高质量干净标签)。

python /root/autodl-tmp/GDbetter0606/merge_vlm_filtered_labels_source_replacement.py \
  --base-label-root /root/autodl-tmp/GDbetter0606/result/Step2_YOLOcompare0615/pred_labels \
  --verdict-csv /root/autodl-tmp/GDbetter0606/result/step4-VLM0615/vlm_box_verdicts.csv \
  --after-sam-candidates /root/autodl-tmp/GDbetter0606/result/step3_SAM0615-2/report/vlm_after_sam_candidates.csv \
  --output-dir /root/autodl-tmp/GDbetter0606/result/step5_final_labels0615


Step 8: 训练最终业务模型
目的：使用 Step7 生成的完美伪标签，训练最终部署的 YOLO 模型。
8-1 划分数据集
python /root/autodl-tmp/GDbetter0606/utilis/split_merge_dataset.py
8-2
最终训练
 python /root/autodl-tmp/GDbetter0606/test_yolo26s.py  --model /root/autodl-tmp/ultralytics-main/ultralytics/weights/yolo26s.pt   --data /root/autodl-tmp/GDbetter0606/result/step6_final_yolo_dataset_0615/data.yaml   --epochs 100 --batch 16 --imgsz 640 --device 0   --project /root/autodl-tmp/GDbetter0606/runs/step7-tainYOLO0615   --use-default   --seed 42 --name finaltrainyolo0615


