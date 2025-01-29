import argparse
from run import Text2Panorama
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(description="Generate panoramic images from text descriptions")
    
    parser.add_argument("--prompt", 
                        type=str, 
                        required=True,
                        help="Text description of the desired panorama")
    
    parser.add_argument("--use-depth",
                        action="store_true",
                        help="Enable depth-controlled generation")
    
    parser.add_argument("--depth-map",
                        type=str,
                        default="pano_depth.png",
                        help="Path to depth map image file (default: pano_depth.png)")
    
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Random seed for reproducibility")
    
    parser.add_argument("--save-dir",
                        type=str,
                        default=None,
                        help="Directory to save generated images (if not provided,\
                              results will be saved in \'results\' or \'results_depth_control\'\
                                based on depth control)")
    
    parser.add_argument("--upscale",
                        action="store_true",
                        help="Enable 2x upscaling of the final result")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize the Text2Panorama model
    text2pano = Text2Panorama(use_depth=args.use_depth,
                              depth_map_dir=args.depth_map,
                              upscale=args.upscale)

    # Initialize Save Directory
    save_dir = args.save_dir
    if not save_dir:
        save_dir = "results" if not args.use_depth else "results_depth_control"
        save_dir = osp.join(save_dir, args.prompt)
    
    # Generate the panorama
    result = text2pano.generate(prompt=args.prompt,
                                seed=args.seed,
                                save_dir=save_dir)
    print(f"Panorama generated successfully at: {result}")

if __name__ == "__main__":
    main()