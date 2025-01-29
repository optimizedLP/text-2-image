# Text-to-Image Generation with Stable Diffusion

This project leverages the power of **Stable Diffusion** to generate images from text prompts. It uses the `diffusers` library by Hugging Face and PyTorch to create high-quality images based on user-provided text inputs.

## Features

- **Text-to-Image Generation**: Generate images from text prompts using the Stable Diffusion model.
- **Customizable Parameters**: Control the number of images, guidance scale, and inference steps.
- **Interactive CLI**: A simple command-line interface for easy interaction.
- **Image Saving**: Automatically saves generated images to a specified directory.

## Requirements

- Python 3.8 or higher
- PyTorch with CUDA support (recommended for GPU acceleration)
- Hugging Face `diffusers` library
- PIL (Pillow) for image handling

## Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/optimizedLP/text2image-stable-diffusion.git
   cd text2image-stable-diffusion

2. Install requirements:
    ```pip install -r requirements.txt

3. Run:
    ```python text-2-image.py

4. Enter the prompt!


## Acknowledgments
Hugging Face Diffusers for the Stable Diffusion implementation.
Stable Diffusion by Stability AI.