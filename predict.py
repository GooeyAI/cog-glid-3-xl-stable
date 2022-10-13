# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import gc
import os
import shutil
import typing

from cog import BasePredictor, Input, Path

import sample

os.environ["TRANSFORMERS_CACHE"] = "transformers_cache"


class Predictor(BasePredictor):
    inpaint_model_args = None
    diffusion_model_args = None

    def setup(self):
        self.inpaint_model_args = sample.do_load(
            sample.parser.parse_args(["--model_path", "inpaint.pt"])
        )
        self.diffusion_model_args = sample.do_load(
            sample.parser.parse_args(["--model_path", "diffusion.pt"])
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
        init_image: Path = Input(
            description="init image to use",
            default=None,
        ),
        edit_image: Path = Input(
            description="The Image you want to edit. If this is provided, then mask is required.",
            default=None,
        ),
        mask: Path = Input(
            description="Black and white image to use as mask for inpainting over init_image. "
            "White pixels = keep, Black pixels = discard",
            default=None,
        ),
        num_outputs: int = Input(
            description="Number of images to output",
            ge=1,
            le=4,
            default=1,
        ),
        negative_prompt: str = Input(
            description="Negative text prompt",
            default=None,
        ),
        outpaint: str = Input(
            choices=["expand", "wider", "taller", "left", "right", "top", "bottom"],
            default=None,
        ),
        skip_timesteps: int = Input(
            description="How many diffusion steps to skip",
            default=0,
        ),
    ) -> typing.List[Path]:

        if edit_image:
            assert mask, "mask must be provided if edit_image is being used"
            args = [
                "--model_path",
                "inpaint.pt",
                "--edit",
                str(edit_image),
                "--mask",
                str(mask),
            ]
            model_args = self.inpaint_model_args
        else:
            args = ["--model_path", "diffusion.pt"]
            model_args = self.diffusion_model_args

        if outpaint:
            assert (
                edit_image
            ), "edit_image and mask must be provided if outpaint is being used"
            args += ["--outpaint", outpaint]

        if init_image:
            args += ["--init_image", str(init_image)]

        if negative_prompt:
            args += ["--negative", negative_prompt]

        args += [
            "--steps",
            str(num_inference_steps),
            "--text",
            prompt,
            "--num_batches",
            str(num_outputs),
            "--skip_timesteps",
            str(skip_timesteps),
        ]

        shutil.rmtree("output", ignore_errors=True)
        shutil.rmtree("output_npy", ignore_errors=True)

        gc.collect()
        sample.do_run(sample.parser.parse_args(args), *model_args)

        return [Path(outfile) for outfile in Path("output").glob("*.png")]
