# Panorama-Generation

Text-to-panorama generation tool using SDXL (Stable Diffusion XL).

## Setup

```bash
# Create conda environment
conda create -n panorama python=3.10
conda activate panorama

# Install dependencies
pip install -r requirements.txt
git clone https://github.com/replicate/cog-sdxl cog_sdxl
```
`cog_sdxl` allows us to use Textual Inversion on Stable Diffusion-XL which can't be done natively in Diffusers. The trigger tokens are `<s0><s1>`. They are concatenated to your prompt during inference, so you only need to include the panorama subject in your prompt.

## Usage
Basic:
```bash
python main.py --prompt "your description"
```

Advanced:
```bash
python main.py \
    --prompt "sci-fi cryo pod" \
    --use-depth \
    --depth-map "pano_depth.png" \
    --seed 0 \
    --save-dir "output"\
    --upscale
```

## Parameters
- `--prompt:` Text prompt for panorama generation (required)
- `--use-depth:` Enable depth controlled panorama generation
- `--depth-map:` Path to the depth map
- `--seed:` Random seed
- `--save-dir:` Directory to save generation outputs
- `--upscale:` To allow upscaling the Panorama

## Project Structure
├── run.py                 # Core generation code
├── main.py                # CLI interface
├── requirements.txt       # Dependencies
├── pano_depth.png         # Given Depth Map
├── results/               # Output directory for Panorama Generation without Depth Control
└── results_depth_control/ # Output directory for Panorama Generation with Depth Control

## Results
Generated panoramas will be saved in the specified output directory.