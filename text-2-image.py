import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

def setup_environment():
    print("Setting up environment...")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))

def initialize_pipeline(model_version="runwayml/stable-diffusion-v1-5"):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_version,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
    
    return pipe

def generate_images(
    prompt,
    pipe,
    num_images=1,
    output_dir="generated_images",
    guidance_scale=7.5,
    num_inference_steps=50
):
    os.makedirs(output_dir, exist_ok=True)
    
    generated_images = []
    for i in range(num_images):
        print(f"\nGenerating image {i+1}/{num_images}")
        image = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]
        
        filename = f"{output_dir}/{prompt[:50].replace(' ', '_')}_{i}.png"
        image.save(filename)
        generated_images.append(image)
        print(f"Saved: {filename}")
    
    return generated_images

def get_user_input():
    print("\n" + "="*40)
    prompt = input("Enter your text prompt: ").strip()
    while not prompt:
        print("Prompt cannot be empty!")
        prompt = input("Enter your text prompt: ").strip()
    
    num_images = input("Number of images to generate (default 1): ").strip()
    num_images = int(num_images) if num_images.isdigit() else 1
    
    return prompt, num_images

def main():
    setup_environment()
    pipe = initialize_pipeline()
    
    while True:
        try:
            prompt, num_images = get_user_input()
            print(f"\nGenerating {num_images} image(s) for prompt: '{prompt}'")
            
            images = generate_images(
                prompt=prompt,
                pipe=pipe,
                num_images=num_images,
                guidance_scale=8.0,
                num_inference_steps=75
            )
            
            # Optional: Display first image
            if input("\nShow first image? (y/n): ").lower() == 'y':
                images[0].show()
            
        except Exception as e:
            print(f"\nError: {str(e)}")
        
        if input("\nGenerate another image? (y/n): ").lower() != 'y':
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()