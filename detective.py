import json
import os
import re
import ast
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# --- Load LLM ---
model_path = "local_path../model/qwen7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# --- Load Data ---
with open("/data/shared/users/antara/rag/video/output/summary.json") as f:
    steps = json.load(f)

with open("/data/shared/users/antara/rag/video/output/ocr_caption_results.json") as f:
    frames = json.load(f)

# --- Prompt Template ---
def build_prompt(step_no, frame_no, step_text, frame_text):
    return f"""
You are an action verification model.

Determine if the frame confirms that the described step has been completed.

Step Number: {step_no}
Frame Number: {frame_no}

Step Description:
\"\"\"{step_text}\"\"\"

Frame OCR and Caption:
\"\"\"{frame_text}\"\"\"

Respond in **JSON only** (no explanation outside JSON):

{{
  "match": true | false,
  "reason": "short reason why"
}}
"""

# --- Check Match Function ---
def check_llm_match(step_no, frame_no, step_text, frame_text, debug_f):
    prompt = build_prompt(step_no, frame_no, step_text, frame_text)
    print(f"\n====================")
    print(f"🔎 Checking Step {step_no} with Frame {frame_no}")
    print("📝 Prompt:\n", prompt.strip())

    result = llm(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]
    print("🧠 Model Output:\n", result.strip())
    debug_f.write(f"\n[STEP {step_no} | FRAME {frame_no}]\nPROMPT:\n{prompt.strip()}\n\nOUTPUT:\n{result.strip()}\n\n")

    try:
        json_strs = re.findall(r'\{.*?\}', result, flags=re.DOTALL)
        for js in json_strs:
            try:
                parsed = json.loads(js)  # stricter than ast.literal_eval
                if isinstance(parsed, dict) and "match" in parsed:
                    return parsed["match"], parsed.get("reason", "")
            except json.JSONDecodeError:
                continue
        return False, "No valid JSON match found"
    except Exception as e:
        return False, f"Parsing failed: {str(e)}"

# --- Verification Pipeline ---
output_path = "/output/comparison/step_verification_llm.json"
debug_log_path = output_path.replace(".json", "_debug.txt")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

verification = []
with open(debug_log_path, "w") as debug_f:
    for idx, step in enumerate(steps, start=1):
        step_text = step["description"].strip()
        matches = []

        for frame in frames:
            frame_no = frame["frame"]
            matched, reason = check_llm_match(idx, frame_no, step_text, frame["description"], debug_f)

            if matched:
                matches.append({
                    "step_no": idx,
                    "frame_no": frame_no,
                    "description": frame["description"],
                    "reason": reason
                })

        verification.append({
            "step_id": step["step_id"],
            "step_no": idx,
            "description": step_text, 
            "status": "matched" if matches else "missing",
            "matched_frames": [m["frame_no"] for m in matches],
            "matched_descriptions": [m["description"] for m in matches],
            "reasons": [m["reason"] for m in matches],
            "frame_refs": matches
        })

# --- Save Output ---
with open(output_path, "w") as f:
    json.dump(verification, f, indent=2)

print(f"\n✅ Step verification report saved to: {output_path}")
print(f"🪵 Debug log saved to: {debug_log_path}")
