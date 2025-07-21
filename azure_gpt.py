import json
from bs4 import BeautifulSoup
from openai import AzureOpenAI
import os

# ---------- Config ---------- #
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = "gpt-4o"
subscription_key = os.getenv("AZURE_API_KEY")
api_version = "2024-12-01-preview"

# Initialize client
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# ---------- Load Inputs ---------- #

with open('output/comparison/llm_verification_report.json', 'r') as f:
    video_data = json.load(f)

with open('logs/test_result.html', 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f, 'html.parser')

# ---------- Parse Agent Report ---------- #

def extract_test_outcome(soup):
    outcome = {}
    test_case = soup.find('div', class_='test outcome outcome-failed')
    if test_case:
        rows = test_case.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            if len(cols) == 2:
                key = cols[0].text.strip().lower()
                val = cols[1].text.strip()
                outcome[key] = val
    return outcome

agent_report = extract_test_outcome(soup)

# ---------- Build Prompt for LLM ---------- #

def build_llm_prompt(video_data, agent_report):
    matched = video_data.get("matched", [])
    missing = video_data.get("missing", [])
    
    return f"""
You are an expert in test validation. Analyze if the agent’s behavior aligns with the video evidence and final test output.

Follow this exact decision tree:

Step in plan?
│
├──> Yes → Is there video evidence?
│     ├──> Yes → Does agent report match? → ✅ Aligned
│     └──> No  → ⚠️ Possibly incomplete
└──> No → Not applicable to current test

Inputs:
-----------------------------
1. Agent Final Test Outcome:
{json.dumps(agent_report, indent=2)}

2. Video Evidence:
Matched Steps:
{json.dumps(matched, indent=2)}

Missing Steps:
{json.dumps(missing, indent=2)}

Return ONLY in this JSON format:
{{
  "steps_aligned": [...matched step IDs...],
  "steps_with_deviation": [
    {{
      "step_id": <int>,
      "step_description": <string>,
      "video_evidence": <string>,
      "test_output_result": <string>,
      "reason": <string>
    }}
  ],
  "overall_alignment_status": <string>,
  "final_result": <string>
}}

Respond ONLY in valid JSON.
"""

# ---------- Call Azure OpenAI ---------- #

def call_llm(prompt: str) -> dict:
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in test validation. Analyze agent logs, plan expectations, video evidence, and test report to determine step alignment. "
                    "Use the provided decision tree and return only in valid JSON in the expected format only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=2048,
        temperature=0,
    )
    content = response.choices[0].message.content
    print("Raw LLM response content:")
    print(content)

    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        with open('output/deviation_report/llm_raw_response.txt', 'w') as f:
            f.write(content)
        raise

# ---------- Run Pipeline ---------- #

prompt = build_llm_prompt(video_data, agent_report)
alignment_report = call_llm(prompt)

# ---------- Save/Print Result ---------- #

print("\n✅ Final Alignment Report:")
print(json.dumps(alignment_report, indent=2))

with open('output/deviation_report/final_alignment_report.json', 'w') as f:
    json.dump(alignment_report, f, indent=2)
