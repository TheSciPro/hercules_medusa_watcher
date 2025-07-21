import os
import json
import re
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ==== Setup Paths ====
summary_path = "/data/shared/users/antara/rag/video/output/summary.json"
input_verification_path = "/data/shared/users/antara/rag/video/output/comparison/step_verification_llm.json"
frame_folder = "/data/shared/users/antara/rag/video/output/frames"
output_path = "/data/shared/users/antara/rag/agentic/output/video_agent/visual_verification_report.json"

# ==== Load Qwen2-VL ====
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map=device
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# ==== Load Data ====
with open(summary_path) as f:
    steps = json.load(f)

with open(input_verification_path) as f:
    verification_data = json.load(f)

# ==== Helper: Qwen2-VL Visual Step Verifier ====
def verify_step_with_frame(step_text, image: Image.Image) -> dict:
    prompt = (
        f"You are a strict visual verifier.\n"
        f"Does this image clearly confirm that the following step was completed?\n\n"
        f"Step: {step_text}\n\n"
        f"Respond only if the evidence is visually obvious.\n\n"
        f"Reply in JSON like: {{\"match\": true/false, \"reason\": \"...\"}}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)
        response = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    try:
        match = re.search(r'\{.*?\}', response, flags=re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
        return parsed
    except:
        return {"match": False, "reason": "Failed to parse model response"}

# ==== Run Verification ====
final_verification = []
for item in verification_data:
    step_id = item["step_id"]
    step_desc = item["description"]
    matched = False
    confirmed_frames = []

    for frame_ref in item.get("frame_refs", []):
        frame_path = os.path.join(frame_folder, frame_ref["frame_no"])
        if not os.path.exists(frame_path):
            continue

        image = Image.open(frame_path)
        result = verify_step_with_frame(step_desc, image)
        print(f"🧪 Step {step_id} Frame {frame_ref['frame_no']} → Match: {result['match']}")

        if result["match"] is True:
            matched = True
            confirmed_frames.append({
                "frame": frame_ref["frame_no"],
                "reason": result.get("reason", "")
            })

    status = "matched" if matched else "unverified"
    final_verification.append({
        "step_id": step_id,
        "description": step_desc,
        "status": status,
        "confirmed_frames": confirmed_frames
    })

# ==== Save Output ====
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(final_verification, f, indent=2)

print(f"\n✅ Strict visual verification completed. Output: {output_path}")
