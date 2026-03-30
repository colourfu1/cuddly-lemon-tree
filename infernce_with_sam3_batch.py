import torch
import os
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import (
    extract_predictions_to_indexes,
    prepare_inputs,
    draw_bboxes_and_save,
    extract_predictions_to_bboxes,
)
from vlm_fo1.task_templates import OD_template

# Paths to required files
sam3_model_path = "./resources/sam3/sam3.pt"  # SAM3 model checkpoint
model_path = 'omlab/VLM-FO1_Qwen2.5-VL-3B-v01'
# model_path = './resources/VLM-FO1_Qwen2.5-VL-3B-v01'  # VLM FO1 model path
label_prompt = "fire,smoke"
confidence_threshold = 0.5

# Initialize UPN object detector
sam3_model = build_sam3_image_model(checkpoint_path=sam3_model_path, device="cuda")
sam3_processor = Sam3Processor(sam3_model, confidence_threshold=confidence_threshold, device="cuda")

# Load vision-language model and tokenizer
tokenizer, model, image_processors = load_pretrained_model(model_path)

# == Batch processing configuration ==
INPUT_DIR = "demo/input_images"  # Input folder with images
OUTPUT_DIR = "demo/output_results"  # Output folder for results
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process all images in the input directory
for img_file in os.listdir(INPUT_DIR):
    try:
        # 1. Filter by image file extensions
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue

        # 2. Construct file paths
        input_path = os.path.join(INPUT_DIR, img_file)
        output_basename = os.path.splitext(img_file)[0]
        output_path = os.path.join(OUTPUT_DIR, f"vlm_result_{output_basename}.jpg")

        # 3. Load and preprocess image
        img_pil = Image.open(input_path).convert("RGB")

        # 4. Run SAM3 to get fine-grained object proposals
        inference_state = sam3_processor.set_image(img_pil)
        sam3_processor.reset_all_prompts(inference_state)
        output = sam3_processor.set_text_prompt(state=inference_state,
                                                prompt=label_prompt)

        # 5. Get the masks, bounding boxes, and scores
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

        # 6. Sort by scores from high to low
        sorted_indices = torch.argsort(scores, descending=True)
        masks = masks[sorted_indices][:100, :]
        boxes = boxes[sorted_indices][:100, :]
        scores = scores[sorted_indices][:100]

        # 7. Prepare chat messages with vision input and bounding boxes
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": input_path},
                    },
                    {
                        "type": "text",
                        "text": OD_template.format(label_prompt),
                    },
                ],
                "bbox_list": boxes.tolist(),
            }
        ]

        # 8. Prepare input for model generation
        generation_kwargs = prepare_inputs(
            model_path, model, image_processors, tokenizer, messages,
            max_tokens=4096, top_p=0.05, temperature=0.0, do_sample=False
        )

        # 9. Run inference and decode output
        with torch.inference_mode():
            output_ids = model.generate(**generation_kwargs)
            outputs = tokenizer.decode(output_ids[0, generation_kwargs['inputs'].shape[1]:]).strip()
            print(f"Processing {img_file}:")
            print("========output======\n", outputs)

        # 10. Convert output prediction (indexes) to bounding box coordinates
        bbox_indexes = extract_predictions_to_indexes(outputs)
        res = {}
        res_masks = []

        for label, index in bbox_indexes.items():
            if label not in res:
                res[label] = []
            for i in index:
                res[label].append(boxes[i].tolist())
                res_masks.append(masks[i].tolist())

        # 11. Draw detected bounding boxes and save visualization
        draw_bboxes_and_save(
            image=img_pil,
            fo1_bboxes=res,
            output_path=output_path
        )

        print(f"Successfully processed: {img_file} -> Saved to {output_path}")

    except Exception as e:
        print(f"Error processing {img_file}: {str(e)}")
        continue

# == Resource cleanup ==
torch.cuda.empty_cache()