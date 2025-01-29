import torch
from diffusers import (
    StableDiffusionXLPipeline, 
    AutoPipelineForInpainting,
    StableDiffusionXLImg2ImgPipeline,
    ControlNetModel, 
    StableDiffusionXLControlNetPipeline
)
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
import os
import os.path as osp
from cog_sdxl.dataset_and_utils import TokenEmbeddingsHandler
from typing import Optional, Tuple

class Text2Panorama:
    """A class to generate panoramic images from text descriptions using SDXL."""
    
    def __init__(self, use_depth: bool = False, depth_map_dir: Optional[str] = "pano_depth.png", upscale: bool = False):
        """Initialize the Text2Panorama pipeline.
        
        Args:
            use_depth: Whether to use depth control in generation
            depth_map: Optional path to a depth map file
        """
        self.use_depth = use_depth
        self.depth_map = Image.open(depth_map_dir) if depth_map_dir and self.use_depth else None
        self.upscale = upscale

        # Base SDXL model (with optional depth control)
        self.pipe = None
        if use_depth:
            controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0",
                                                     torch_dtype=torch.float16).to("cuda")
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                variant="fp16",
                ).to("cuda")
        
        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                variant="fp16",
            ).to("cuda")
        
        # Load LoRA weights fine-tuned for panorama generation via SD-XL
        self.pipe.load_lora_weights(
            "jbilcke-hf/sdxl-panorama",
            weight_name="lora.safetensors",
        )

        # Loading LoRA Weights' specific token embeddings to enable Textual Inversion
        text_encoders = [self.pipe.text_encoder, self.pipe.text_encoder_2]
        tokenizers = [self.pipe.tokenizer, self.pipe.tokenizer_2]
        embedding_path = hf_hub_download(repo_id="jbilcke-hf/sdxl-panorama", filename="embeddings.pti", repo_type="model")
        embhandler = TokenEmbeddingsHandler(text_encoders, tokenizers)
        embhandler.load_embeddings(embedding_path)

        
        # Inpainting model to correct seams
        model_card = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        model_card = 'stabilityai/stable-diffusion-2-inpainting'
        self.pipe_inpaint = AutoPipelineForInpainting.from_pretrained(
            model_card,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
        
        # Upscaler model (Optional - only used if upscaling is enabled)
        self.pipe_upscaler = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")

    def _generate_initial_panorama(self, prompt: str, seed: int) -> Image.Image:
        """Generate initial Panorama by prompting SD-XL (it may have a seam in the middle).
        
        Args:
            prompt: Text prompt for generation
            seed: Random seed
            
        Returns:
            Generated image
        """
        prompt_sdxl = f"{prompt}, in the style of <s0><s1>"
        image = self.pipe(
            prompt_sdxl,
            image=self.depth_map,
            cross_attention_kwargs={"scale": 0.8},
            width=1024,
            height=512,
            generator=torch.manual_seed(seed),
            controlnet_conditioning_scale=0.5,  # Adjust strength of depth control
        ).images[0]
        return image

    def _upscale_image(self, image: Image.Image, prompt: str, seed: int) -> Image.Image:
        """Increase resoltution of the image by 2x using SD-XL.
        
        Args:
            image: Image to upscale
            prompt: Original prompt
            seed: Random seed
            upscale: Whether to perform upscaling
            
        Returns:
            Upscaled image
        """

        target_width, target_height = 2048, 1024
        image_resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        upscaled_img = self.pipe_upscaler(
            prompt=prompt,
            image=image_resized,
            guidance_scale=7.5,
            strength=0.3,
            width=target_width,
            height=target_height,
            generator=torch.manual_seed(seed)
        ).images[0]
        
        return upscaled_img

    def _swap_halves(self, image: Image.Image) -> Image.Image:
        """Swap the left and right halves of an image.
        
        Args:
            image: Input image
            
        Returns:
            Image with swapped halves
        """
        width, height = image.size
        midpoint = width // 2
        left_half = image.crop((0, 0, midpoint, height))
        right_half = image.crop((midpoint, 0, width, height))
        image.paste(right_half, (0, 0))
        image.paste(left_half, (midpoint, 0))
        return image
    
    def create_mask(self, width: int, height: int, divisions: int) -> Image.Image:
        """Create a mask image with a white rectangle in the middle.
        
        Args:
            width: Width of the mask
            height: Height of the mask
            divisions: Number of divisions for the mask width
            
        Returns:
            The created mask image
        """
        image = Image.new("RGB", (width, height), "black")
        left = width // divisions
        right = width // divisions * (divisions - 1)
        draw = ImageDraw.Draw(image)
        draw.rectangle([left, 0, right, height], fill="white")
        image.save('mask.png')
        return image

    def _inpaint_seam(self, image: Image.Image, prompt: str, seed: int) -> Image.Image:
        """Inpaint the seam in the middle of the image.
        
        Args:
            image: Input image
            prompt: Generation prompt
            seed: Random seed
            
        Returns:
            Inpainted image
        """
        divisions = 4 if self.use_depth else 6
        mask_image = self.create_mask(*image.size, divisions)
        return self.pipe_inpaint(
            prompt=f"{prompt}, high quality, photorealistic, seamless continuation, consistent lighting and color, sharp focus",
            negative_prompt="artifacts, blurry, distorted, seams, edges, inconsistent lighting, color mismatch, filtered look",
            image=image,
            mask_image=mask_image,
            guidance_scale=7.5,
            num_inference_steps=40,
            strength=0.85,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            width=image.size[0],
            height=image.size[1]
        ).images[0]

    def generate(self, prompt: str, seed: Optional[int] = None, save_dir: str = "") -> str:
        """Generate a panoramic image from a text prompt.
        
        Args:
            prompt: Text description of the desired image
            seed: Random seed for generation
            upscale: Whether to upscale the final image
            save_dir: Optional directory to save intermediate images
            
        Returns:
            Path to the generated image file
        """
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if save_dir and not osp.exists(save_dir):
            os.makedirs(save_dir)

        # Generate initial panorama
        image = self._generate_initial_panorama(prompt, seed)
        image.save(osp.join(save_dir, "1-base.png"))
        if self.upscale:
            image = self._upscale_image(image, prompt, seed)
            image.save(osp.join(save_dir, "3-upscaled.png"))

        # Swap Halves to prepare for inpainting
        image = self._swap_halves(image)
        image.save(osp.join(save_dir, "2-swap.png"))

        # Use Inpainting to correct seam
        width, height = image.size
        left = (width - height) // 2
        middle_square = image.crop((left, 0, left + height, height))
        middle_square.save(osp.join(save_dir, "4-square.png"))
        
        inpainted = self._inpaint_seam(middle_square, prompt, seed)
        inpainted.save(osp.join(save_dir, "5-inpainted.png"))

        # Compose final result
        image.paste(inpainted, (left, 0))
        image.save(osp.join(save_dir, "6-final.png"))
        
        final = self._swap_halves(image)
        final.save(osp.join(save_dir, "7-final-result.png"))
        
        return osp.join(save_dir, "7-final-result.png")
    

# Usage example
if __name__ == "__main__":
    generator = Text2Panorama(use_depth=True, depth_map_dir="pano_depth.png", upscale=True)
    prompt = "science lab"
    save_dir = "results_depth_control" if generator.use_depth else "results"
    save_dir = osp.join(save_dir, prompt)
    output_path = generator.generate(prompt, seed=42, save_dir=save_dir)
    print(f"Generated panorama saved to: {output_path}")