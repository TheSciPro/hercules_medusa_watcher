import json
import pixeltable as pxt
import pandas as pd
from loguru import logger
from pathlib import Path

# Load Pixeltable view/table
view = pxt.get_table("zeus_cache.demo_video_6dd4_table_frames")

# Load test steps from summary.json
summary_path = Path("output/summary.json")
with summary_path.open("r", encoding="utf-8") as f:
    summary_data = json.load(f)

# Extract step_id and description from JSON
test_steps = [{"step_id": step["step_id"], "description": step["description"]} for step in summary_data]

aligned_steps = []

# Process each step
for step in test_steps:
    step_id = step["step_id"]
    description = step["description"]

    # Compute similarity expression (Pixeltable auto-handles embedding)
    sim_expr = view.im_caption.similarity(description)

    # Query top matching frame (no aliasing for score)
    result = (
        view
        .order_by(sim_expr, asc=False)
        .limit(1)
        .select(
            view.im_caption,
            view.pos_msec,
            view.frame_idx,  # Use frame_idx instead of rowid
            sim_expr
        )
        .collect()[0]
    )

    # Unpack result (no aliasing, so fourth value is unnamed similarity score)
    matched_caption, timestamp, frame_id, score = result

    try:
        score = float(score)
    except Exception as e:
        logger.error(f"Could not convert similarity score to float: {score} | Error: {e}")
        score = 0.0

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

# Save and print
df = pd.DataFrame(aligned_steps)
print(df)
df.to_csv("alignment_output.csv", index=False)
