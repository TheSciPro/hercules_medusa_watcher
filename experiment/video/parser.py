import json
import csv
from pathlib import Path

INPUT_FILE = Path("logs/agent_inner_logs.json")
SUMMARY_JSON = Path("output/summary.json")
CSV_REPORT = Path("output/report.csv")


def load_log():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("planner_agent", [])  # ✅ Only return planner_agent steps


def parse_steps(data):
    steps = []
    current_step = {}
    step_id = 1

    for entry in data:
        if not isinstance(entry, dict):
            continue  # Skip malformed entries

        role = entry.get("role", "")
        content = entry.get("content", {})

        if role == "assistant" and isinstance(content, dict):
            if content.get("plan"):
                current_step = {
                    "step_id": step_id,
                    "step_type": "ASSERT" if content.get("is_assert", False) else "NORMAL",
                    "description": content.get("next_step_summary", ""),
                    "plan": content.get("plan", ""),
                    "next_step": content.get("next_step", ""),
                    "is_assert": content.get("is_assert", False),
                    "assert_summary": content.get("assert_summary", ""),
                    "is_passed": content.get("is_passed", False),
                    "terminate": content.get("terminate", False),
                    "final_response": content.get("final_response", ""),
                    "execution_result": {}
                }
                step_id += 1

        elif role == "user" and current_step:
            if isinstance(content, dict):
                current_step["execution_result"] = {
                    "status": content.get("previous_step_status", ""),
                    "output": content.get("current_output", ""),
                    "verification": content.get("Verification_Status", ""),
                    "details": content.get("Verification_Details", ""),
                    "validation": content.get("Task_Completion_Validation", "")
                }
            elif isinstance(content, str):
                parsed = {}
                for line in content.split("\n"):
                    if ":" in line:
                        key, val = line.split(":", 1)
                        parsed[key.strip()] = val.strip()

                current_step["execution_result"] = {
                    "status": parsed.get("previous_step_status", ""),
                    "output": parsed.get("current_output", ""),
                    "verification": parsed.get("Verification_Status", ""),
                    "details": parsed.get("Verification_Details", ""),
                    "validation": parsed.get("Task_Completion_Validation", "")
                }

            steps.append(current_step)
            current_step = {}

    return steps


def save_summary(parsed_steps):
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(parsed_steps, f, indent=2)


def export_to_csv(parsed_steps):
    CSV_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_REPORT, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "step_id", "step_type", "description", "is_assert", "is_passed",
            "assert_summary", "status", "verification", "details", "validation"
        ])
        writer.writeheader()

        for step in parsed_steps:
            exec_result = step.get("execution_result", {})
            writer.writerow({
                "step_id": step.get("step_id", ""),
                "step_type": step.get("step_type", ""),
                "description": step.get("description", ""),
                "is_assert": step.get("is_assert", False),
                "is_passed": step.get("is_passed", False),
                "assert_summary": step.get("assert_summary", ""),
                "status": exec_result.get("status", ""),
                "verification": exec_result.get("verification", ""),
                "details": exec_result.get("details", ""),
                "validation": exec_result.get("validation", "")
            })


def main():
    log_entries = load_log()

    if not isinstance(log_entries, list):
        raise ValueError("Expected a list of log entries under 'planner_agent' in the input JSON.")

    parsed_steps = parse_steps(log_entries)
    save_summary(parsed_steps)
    export_to_csv(parsed_steps)
    print(f"✅ Parsed {len(parsed_steps)} steps. Output written to:\n- {SUMMARY_JSON}\n- {CSV_REPORT}")


if __name__ == "__main__":
    main()
