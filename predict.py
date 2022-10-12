# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import shutil
import subprocess
import sys
import typing

from cog import BasePredictor, Input, Path
import sample

os.environ["TRANSFORMERS_CACHE"] = "transformers_cache"


class Predictor(BasePredictor):
    script_models = None

    def setup(self):
        self.script_models = sample.do_load(
            sample.parser.parse_args(["--model_path", "inpaint.pt"])
        )

    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        num_inference_steps: int = Input(
            description="Number of denoising steps.",
            ge=1,
            le=500,
            default=50,
        ),
        edit_image: Path = Input(
            description="The Image you want to edit.",
        ),
        mask: Path = Input(
            description="Black and white image to use as mask for inpainting over init_image. "
            "White pixels = keep, Black pixels = discard",
        ),
        num_outputs: int = Input(
            description="Number of images to output",
            ge=1,
            le=4,
            default=1,
        ),
    ) -> typing.List[Path]:

        shutil.rmtree("output", ignore_errors=True)
        shutil.rmtree("output_npy", ignore_errors=True)

        sample.do_run(
            sample.parser.parse_args(
                [
                    "--model_path",
                    "inpaint.pt",
                    "--edit",
                    str(edit_image),
                    "--mask",
                    str(mask),
                    "--steps",
                    str(num_inference_steps),
                    "--text",
                    prompt,
                    "--num_batches",
                    str(num_outputs),
                ]
            ),
            *self.script_models,
        )

        return [Path(outfile) for outfile in Path("output").glob("*.png")]
