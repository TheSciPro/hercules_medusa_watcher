from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import torch
import json
import re

# Set device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load ColPali retriever
rag = RAGMultiModalModel.from_pretrained("vidore/colpali")

# Load Qwen2VL model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", 
    torch_dtype=torch.bfloat16,
    device_map=device
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Index image frames
image_folder = "/data/shared/users/antara/rag/video/output/frames"
rag.index(
    input_path=image_folder,
    index_name="video_ocr_full",
    store_collection_with_index=False,
    overwrite=True
)

# Load summary.json
summary_path = "/data/shared/users/antara/rag/video/output/summary.json"
with open(summary_path) as f:
    summary_data = json.load(f)

# Results collection
results = []

# Loop through frames for OCR + mapping
for image_file in sorted(os.listdir(image_folder)):
    if not image_file.endswith(".jpg"):
        continue

    # Extract frame number
    match = re.search(r"frame_(\d+)s\.jpg", image_file)
    if not match:
        continue
    frame_id = int(match.group(1))

    # Find matching step
    matching_step = next((step for step in summary_data if step["step_id"] == frame_id), None)
    if not matching_step:
        print(f"⚠️ No matching step found for frame {image_file}")
        continue

    # Load image
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path)

    # Prompt message
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "You are acting as a strict OCR engine. Read and transcribe **all visible text and UI elements** exactly as they appear in this frame. Do not infer or summarize. List each element you detect. At the end, give a one-line caption describing the purpose of the screen."}
        ]
    }]

    # Generate response
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)
        trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        output = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]

    print(f"🖼️ {image_file}: {output}")

    # Attach OCR to step
    matching_step["execution_result"]["ocr_output"] = output

    # Save to results list too
    results.append({
        "frame": image_file,
        "step_id": frame_id,
        "ocr_output": output
    })

# Save merged summary
summary_output_path = "/data/shared/users/antara/rag/video/output/full_ocr/summary_with_ocr.json"
with open(summary_output_path, "w") as f:
    json.dump(summary_data, f, indent=2)

# Save separate OCR output
ocr_output_path = "/data/shared/users/antara/rag/video/output/full_ocr/ocr_caption_results.json"
with open(ocr_output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Merged summary saved to: {summary_output_path}")
print(f"✅ OCR-only results saved to: {ocr_output_path}")
