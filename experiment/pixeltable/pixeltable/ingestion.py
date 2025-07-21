import uuid
from typing import Optional

import pixeltable as pxt
from loguru import logger
from pixeltable.functions.huggingface import clip
from pixeltable.iterators.video import FrameIterator

from utils import resize_image, get_settings, image_to_text

logger = logger.bind(name="VideoProcessor")
settings = get_settings()


class VideoProcessor:
    def __init__(self):
        self._video_table: Optional[pxt.Table] = None
        self._frames_view: Optional[pxt.View] = None
        self._video_mapping_idx: Optional[str] = None
        self._namespace: str = "zeus_cache"

        logger.info(
            f"VideoProcessor initialized:\n Split FPS: {settings.SPLIT_FRAMES_COUNT}"
        )

    def setup_table(self, video_name: str):
        """Set up Pixeltable tables and views using logical paths (no local dirs)."""
        self._video_mapping_idx = video_name

        suffix = uuid.uuid4().hex[-4:]
        table_base = f"{self._namespace}.{video_name}_{suffix}"

        self._video_table_name = f"{table_base}_table"
        self._frames_view_name = f"{self._video_table_name}_frames"

        self._setup_table()
        logger.info(f"✅ Created video index: '{self._video_table_name}'")

    def _setup_table(self):
        self._setup_namespace()
        self._create_video_table()
        self._setup_frame_processing()

    def _setup_namespace(self):
        """Create a Pixeltable logical namespace if needed."""
        logger.info(f"Creating Pixeltable namespace: {self._namespace}")
        pxt.create_dir(self._namespace, if_exists="replace_force")

    def _create_video_table(self):
        """Create the table that stores the input video."""
        self._video_table = pxt.create_table(
            self._video_table_name,
            schema={"video": pxt.Video},
            if_exists="replace_force",
        )

    def _setup_frame_processing(self):
        """Create a view that extracts, resizes, captions, and embeds frames."""
        self._frames_view = pxt.create_view(
            self._frames_view_name,
            self._video_table,
            iterator=FrameIterator.create(
                video=self._video_table.video,
                num_frames=settings.SPLIT_FRAMES_COUNT,
            ),
            if_exists="replace_force",
        )

        # Resize frames
        self._frames_view.add_computed_column(
            resized_frame=resize_image(
                self._frames_view.frame,
                width=settings.IMAGE_RESIZE_WIDTH,
                height=settings.IMAGE_RESIZE_HEIGHT,
            )
        )

        # Generate captions using BLIP (custom UDF)
        self._frames_view.add_computed_column(
            im_caption=image_to_text(self._frames_view.resized_frame)
        )

        # Embedding on image (CLIP)
        self._frames_view.add_embedding_index(
            column=self._frames_view.resized_frame,
            image_embed=clip.using(model_id=settings.IMAGE_SIMILARITY_EMBD_MODEL),
            if_exists="replace_force",
        )

        # Embedding on caption (CLIP)
        self._frames_view.add_embedding_index(
            column=self._frames_view.im_caption,
            string_embed=clip.using(model_id=settings.CAPTION_SIMILARITY_EMBD_MODEL),
            if_exists="replace_force",
        )

    def add_video(self, video_path: str) -> bool:
        """Insert a video file into the Pixeltable table."""
        if not self._video_table:
            raise ValueError("Video table is not initialized. Call setup_table() first.")
        logger.info(f"Inserting video {video_path} into {self._video_table_name}")
        self._video_table.insert([{"video": video_path}])
        return True

    def get_frame_view_name(self) -> str:
        return self._frames_view_name

    def get_video_mapping_index(self) -> str:
        return self._video_mapping_idx


# ✅ Run standalone
if __name__ == "__main__":
    video_path = r"C:\Users\antara1001\Downloads\my\hackathon\zeus\media\video.webm"
    video_name = "demo_video"

    processor = VideoProcessor()
    processor.setup_table(video_name)
    processor.add_video(video_path)

    print("\n✅ Ingestion completed.")
    print(f"📌 Table name: {processor._video_table_name}")
    print(f"📌 Frame view: {processor._frames_view_name}")
