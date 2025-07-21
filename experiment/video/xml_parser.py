import xml.etree.ElementTree as ET
import re
import json
import csv
from pathlib import Path
from typing import List

def extract_hercules_plan_steps(xml_file_path: str) -> List[str]:
    """
    Extracts the 'plan' property from the Hercules XML report
    and splits it into individual steps.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    plan_text = None

    for testcase in root.iter("testcase"):
        for prop in testcase.iter("property"):
            if prop.attrib.get("name") == "plan":
                plan_text = prop.attrib.get("value")
                break
        if plan_text:
            break

    if not plan_text:
        raise ValueError("No <property name='plan'> found in the XML.")

    plan_text = plan_text.replace("&#10;", "\n")
    raw_steps = re.split(r'(?:^|\n)\d+\.\s+', plan_text)
    steps = [step.strip() for step in raw_steps if step.strip()]
    return steps

def save_steps_to_json(steps: List[str], json_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(steps, f, indent=2)
    print(f"✅ Saved steps to JSON: {json_path}")

def save_steps_to_csv(steps: List[str], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step_id", "description"])
        for idx, step in enumerate(steps, 1):
            writer.writerow([idx, step])
    print(f"✅ Saved steps to CSV: {csv_path}")

def main():
    xml_input = Path("/data/shared/users/antara/rag/video/logs/test_result.xml")  # 🔁 Change path as needed
    json_output = Path("/data/shared/users/antara/rag/video/output/xml_parsed/hercules_plan_steps.json")
    csv_output = Path("/data/shared/users/antara/rag/video/output/xml_parsed/hercules_plan_steps.csv")

    steps = extract_hercules_plan_steps(str(xml_input))
    save_steps_to_json(steps, json_output)
    save_steps_to_csv(steps, csv_output)

    print(f"\n✅ Extracted {len(steps)} plan steps:")
    for i, step in enumerate(steps, 1):
        print(f"{i}. {step}")

if __name__ == "__main__":
    main()
