import json
import pixeltable as pxt
import pandas as pd
from loguru import logger
from pathlib import Path

# Load Pixeltable view
view = pxt.get_table("zeus_cache.demo_video_6dd4_table_frames")

# Load test steps from summary.json
summary_path = Path("output/summary.json")
with summary_path.open("r", encoding="utf-8") as f:
    summary_data = json.load(f)

# Extract step_id and description
test_steps = [{"step_id": step["step_id"], "description": step["description"]} for step in summary_data]

aligned_steps = []

for step in test_steps:
    step_id = step["step_id"]
    description = step["description"]

    # Compute similarity expression (and give it a name)
    sim_expr = view.im_caption.similarity(description).with_name("score")

    # Run Pixeltable query
    result = (
        view
        .order_by(sim_expr, asc=False)
        .limit(1)
        .select(
            view.im_caption,
            view.pos_msec,
            view.frame_idx,
            sim_expr  # already labeled "score"
        )
        .collect()
    )

    if not result:
        logger.warning(f"No match found for Step {step_id}")
        continue

    matched_caption, timestamp, frame_id, score = result[0]

    # score is already a float
    status = "Observed" if score >= 0.75 else "Skipped"

    aligned_steps.append({
        "step_id": step_id,
        "step_description": description,
        "matched_frame_caption": matched_caption,
        "timestamp": str(timestamp),
        "frame_id": frame_id,
        "score": round(score, 4),
        "status": status,
    })

    logger.info(f"[{status}] Step {step_id} → Frame {frame_id} | Score: {score:.4f}")

# Convert to DataFrame
df = pd.DataFrame(aligned_steps)
print(df)

# Optional: Save to CSV
df.to_csv("alignment_output2.csv", index=False)
