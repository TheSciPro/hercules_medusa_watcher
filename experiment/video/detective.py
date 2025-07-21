from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import json
import os

# --- Load LLM ---
model_path = "/data/shared/users/antara/scrap-metal/model/qwen7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# --- Load data ---
with open("/data/shared/users/antara/rag/video/output/summary.json") as f:
    steps = json.load(f)

with open("/data/shared/users/antara/rag/video/output/ocr_caption_results.json") as f:
    frames = json.load(f)

# --- Prompt template ---
def build_prompt(step_text, frame_text):
    return f"""
Does the following frame description indicate that the action described in the step occurred?

Step:
\"\"\"{step_text}\"\"\"

Frame OCR & Caption:
\"\"\"{frame_text}\"\"\"

Respond strictly with JSON:
{{
  "match": true | false,
  "reason": "short reason"
}}
"""

# --- Call LLM per (step, frame) pair ---
def check_llm_match(step_text, frame_text):
    prompt = build_prompt(step_text, frame_text)
    output = llm(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]

    try:
        json_part = output.split("{", 1)[-1].rsplit("}", 1)[0]
        final_json = json.loads("{" + json_part + "}")
        return final_json["match"], final_json["reason"]
    except Exception as e:
        return False, f"Parsing failed: {str(e)}"

# --- Verification ---
verification = []
for step in steps:
    step_text = f"{step['description']} {step['plan']} {step['next_step']}"
    matches = []

    for frame in frames:
        matched, reason = check_llm_match(step_text, frame["description"])
        if matched:
            matches.append({
                "frame": frame["frame"],
                "description": frame["description"],
                "reason": reason
            })

    verification.append({
        "step_id": step["step_id"],
        "status": "matched" if matches else "missing",
        "matched_frames": [m["frame"] for m in matches],
        "matched_descriptions": [m["description"] for m in matches],
        "reasons": [m["reason"] for m in matches]
    })

# --- Save output ---
output_path = "/data/shared/users/antara/rag/video/output/comparison/step_verification_llm.json"
with open(output_path, "w") as f:
    json.dump(verification, f, indent=2)

print(f"✅ Step verification report saved to: {output_path}")
