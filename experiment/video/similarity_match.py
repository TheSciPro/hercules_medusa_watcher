import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict

HERCULES_JSON = Path("/data/shared/users/antara/rag/video/output/xml_parsed/hercules_plan_steps.json")
LLM_REPORT_JSON = Path("/data/shared/users/antara/rag/video/output/comparison/llm_verification_report.json")
OUTPUT_JSON = Path("/data/shared/users/antara/rag/video/output/hercules_similarity/step_alignment.json")

# Load a small and effective sentence transformer model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def load_hercules_steps() -> List[str]:
    with open(HERCULES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def load_llm_steps() -> List[Dict]:
    with open(LLM_REPORT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("matched", [])  # Safely extract matched steps

def match_steps(hercules_steps: List[str], llm_steps: List[Dict]) -> List[Dict]:
    if not llm_steps:
        raise ValueError("LLM steps are empty or missing from the report.")

    herc_embeddings = model.encode(hercules_steps, convert_to_tensor=True)
    llm_descriptions = [step["description"] for step in llm_steps]
    llm_embeddings = model.encode(llm_descriptions, convert_to_tensor=True)

    matches = []

    for idx, (herc_step, herc_emb) in enumerate(zip(hercules_steps, herc_embeddings), start=1):
        similarities = util.cos_sim(herc_emb, llm_embeddings)[0]
        best_match_idx = int(similarities.argmax())
        best_score = float(similarities[best_match_idx])
        best_llm_step = llm_steps[best_match_idx]

        match = {
            "hercules_step_id": idx,
            "hercules_step": herc_step,
            "matched_llm_step_id": best_llm_step.get("step_id"),
            "llm_description": best_llm_step.get("description"),
            "similarity_score": round(best_score, 4)
        }
        matches.append(match)

    return matches

def save_matches(matches: List[Dict]):
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2)
    print(f"✅ Matches saved to: {OUTPUT_JSON}")

def main():
    hercules_steps = load_hercules_steps()
    llm_steps = load_llm_steps()
    matches = match_steps(hercules_steps, llm_steps)
    save_matches(matches)

    print("\n📊 Top Matches:")
    for match in matches:
        print(f"[{match['similarity_score']}] Hercules #{match['hercules_step_id']} → LLM #{match['matched_llm_step_id']}")
        print(f"  - Hercules: {match['hercules_step']}")
        print(f"  - LLM:      {match['llm_description']}")
        print()

if __name__ == "__main__":
    main()
