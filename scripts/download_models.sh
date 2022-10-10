#!/usr/bin/env bash

set -ex

# download stable diffusion model
wget -c -O diffusion.pt https://huggingface.co/Jack000/glid-3-xl-stable/resolve/main/default/diffusion-1.4.pt
wget -c -O kl.pt https://huggingface.co/Jack000/glid-3-xl-stable/resolve/main/default/kl-1.4.pt

# download inpaint model
wget -c -O inpaint.pt https://huggingface.co/Jack000/glid-3-xl-stable/resolve/main/inpaint/ema_0.9999_100000.pt

# download clip models
cog run scripts/download_clip_models.py
sudo chown -R $USER:$USER transformers_cache/*
