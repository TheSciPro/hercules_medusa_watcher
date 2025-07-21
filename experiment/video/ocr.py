from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import torch
import json

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
    index_name="video_ocr",
    store_collection_with_index=False,
    overwrite=True
)

# ✅ Define local results list
results = []

# Loop through frames for captioning
for image_file in sorted(os.listdir(image_folder)):
    if not image_file.endswith(".jpg"):
        continue
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "You are acting as a strict OCR engine. Read and transcribe **all visible text and UI elements** exactly as they appear in this frame. Do not infer or summarize. List each element you detect. At the end, give a one-line caption describing the purpose of the screen."}
        ]
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)
        trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        output = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]

    print(f"🖼️ {image_file}: {output}")
    results.append({"frame": image_file, "description": output})

# Save the result as a JSON file
output_path = "/data/shared/users/antara/rag/video/output/ocr_caption_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ OCR and caption results saved to: {output_path}")
