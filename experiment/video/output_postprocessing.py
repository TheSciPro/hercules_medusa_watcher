import json

# Load the list of steps from JSON
with open("/data/shared/users/antara/rag/video/output/comparison/step_verification_llm.json", "r") as f:
    steps = json.load(f)

# Initialize categories
report = {
    "matched": [],
    "missing": []
}

# Process each step entry
for step in steps:
    step_id = step.get("step_id")
    step_no = step.get("step_no")
    status = step.get("status")

    if status == "matched":
        report["matched"].append({
            "step_id": step_id,
            "step_no": step_no,
            "matched_frames": step.get("matched_frames", []),
            "matched_descriptions": step.get("matched_descriptions", [])
        })
    elif status == "missing":
        report["missing"].append({
            "step_id": step_id,
            "step_no": step_no,
            "description": step.get("description", "")
        })

# Save the output
with open("/data/shared/users/antara/rag/video/output/comparison/llm_verification_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("✅ Report generated: llm_verification_report.json")
