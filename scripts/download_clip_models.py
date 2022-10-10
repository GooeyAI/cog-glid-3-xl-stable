import os

os.environ["TRANSFORMERS_CACHE"] = "transformers_cache"

from transformers import CLIPTokenizer, CLIPTextModel

clip_version = "openai/clip-vit-large-patch14"
clip_tokenizer = CLIPTokenizer.from_pretrained(clip_version)
clip_transformer = CLIPTextModel.from_pretrained(clip_version)
