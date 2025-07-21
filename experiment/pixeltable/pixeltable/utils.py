# utils.py
import pixeltable as pxt
from PIL import Image
from dataclasses import dataclass
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load BLIP model once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

@pxt.udf
def image_to_text(image: pxt.type_system.Image) -> str:
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image")
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

@pxt.udf
def resize_image(image: pxt.type_system.Image, width: int, height: int) -> pxt.type_system.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image")
    image.thumbnail((width, height))
    return image

@dataclass
class Settings:
    SPLIT_FRAMES_COUNT: int = 8
    IMAGE_RESIZE_WIDTH: int = 224
    IMAGE_RESIZE_HEIGHT: int = 224
    IMAGE_SIMILARITY_EMBD_MODEL: str = "openai/clip-vit-base-patch32"
    CAPTION_SIMILARITY_EMBD_MODEL: str = "openai/clip-vit-base-patch32"

def get_settings() -> Settings:
    return Settings()
