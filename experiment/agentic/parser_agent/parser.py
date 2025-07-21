import json
from pathlib import Path
from typing import List, Dict, Any
from openai import AzureOpenAI

# Azure OpenAI config
client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="",
    api_key="",
)

SUMMARY_PATH = Path("/data/shared/users/antara/rag/agentic/output/summary.json")
REFLECTION_PATH = Path("/data/shared/users/antara/rag/agentic/output/plan_review.txt")


class PlannerAgent:
    def __init__(self, input_file: Path):
        self.input_file = input_file
        self.logs = self._load_log()

    def _load_log(self) -> List[Dict[str, Any]]:
        with open(self.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("planner_agent", [])

    def parse_steps(self) -> List[Dict[str, Any]]:
        steps = []
        current_step = {}
        step_id = 1

        for entry in self.logs:
            if not isinstance(entry, dict):
                continue

            role = entry.get("role", "")
            content = entry.get("content", {})

            if role == "assistant" and isinstance(content, dict) and content.get("plan"):
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
                parsed = self._parse_user_content(content)
                current_step["execution_result"] = parsed
                steps.append(current_step)
                current_step = {}

        return steps

    @staticmethod
    def _parse_user_content(content) -> Dict[str, str]:
        if isinstance(content, dict):
            return {
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
            return {
                "status": parsed.get("previous_step_status", ""),
                "output": parsed.get("current_output", ""),
                "verification": parsed.get("Verification_Status", ""),
                "details": parsed.get("Verification_Details", ""),
                "validation": parsed.get("Task_Completion_Validation", "")
            }
        return {}

    def reflect_on_plan(self, steps: List[Dict[str, Any]]) -> str:
        plan_text = "\n".join(
            [f"{step['step_id']}. {step['description']}" for step in steps if step.get("description")]
        )

        prompt = f"""You are an AI plan reviewer.
Here is a sequence of steps from an AI agent's planning log:

{plan_text}

Please reflect on this plan and answer:
1. Are there any missing or redundant steps?
2. Are there inconsistencies or unclear transitions?
3. Suggest any fixes or reordering for better coherence.

Return your thoughts followed by a revised step list if needed.
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in validating AI reasoning and plans."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    def revise_steps_if_suggested(self, original_steps: List[Dict[str, Any]], reflection: str) -> List[Dict[str, Any]]:
        if "Revised Step List" in reflection or "Revised Plan" in reflection:
            # Extract revised step list using simple parsing
            revised = []
            in_list = False
            for line in reflection.splitlines():
                if line.strip().startswith("1.") or in_list:
                    in_list = True
                    if line.strip() == "":
                        continue
                    parts = line.strip().split(".", 1)
                    if len(parts) == 2:
                        step_text = parts[1].strip()
                        revised.append(step_text)

            return [
                {
                    "step_id": i + 1,
                    "step_type": "NORMAL",
                    "description": desc,
                    "plan": desc,
                    "next_step": "",
                    "is_assert": False,
                    "assert_summary": "",
                    "is_passed": False,
                    "terminate": False,
                    "final_response": "",
                    "execution_result": {}
                }
                for i, desc in enumerate(revised)
            ]
        return original_steps

    def save_summary(self, steps: List[Dict[str, Any]], output_file: Path):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(steps, f, indent=2)

    def save_reflection(self, content: str, output_file: Path):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")


# ---- Callable interface ----

def run_planner_agent(input_file="/data/shared/users/antara/rag/agentic/logs/agent_inner_logs.json",
                      output_file="/data/shared/users/antara/rag/agentic/output/summary.json",
                      reflection_output="/data/shared/users/antara/rag/agentic/output/plan_review.txt"):

    agent = PlannerAgent(Path(input_file))
    steps = agent.parse_steps()

    # 🔍 Reflect
    reflection = agent.reflect_on_plan(steps)
    agent.save_reflection(reflection, Path(reflection_output))
    print("\n🧠 Plan Reflection:\n" + reflection + "\n")

    # ✨ Auto-revise if needed
    revised_steps = agent.revise_steps_if_suggested(steps, reflection)
    agent.save_summary(revised_steps, Path(output_file))

    print(f"✅ Final step list saved to {output_file}")
    print(f"📝 Reflection saved to {reflection_output}")
    return revised_steps


if __name__ == "__main__":
    run_planner_agent()
