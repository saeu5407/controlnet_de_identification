import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import cv2
import gradio as gr

try:
    from src.utils import ImageProcessor
    from src.pipeline import controlnet_landmark_pipeline
    base_path = '...'
except:
    from ..utils import ImageProcessor
    from ..pipeline import controlnet_landmark_pipeline
    base_path = os.getcwd().split('src')[0]

import warnings
warnings.filterwarnings("ignore")

def setup_gradio_interface():

    # inputs
    prompt_input = gr.Textbox(label="Prompt")
    negative_prompt_input = gr.Textbox(label="Negative Prompt")
    image_input = gr.Image(label="Image", type="pil")
    num_steps_input = gr.Slider(minimum=1, maximum=100, step=5, label="Num Inference Steps")
    controlnet_scale_input = gr.Slider(minimum=0, maximum=1, step=0.1, label="ControlNet Conditioning Scale")
    guidance_scale_input = gr.Slider(minimum=0, maximum=10, step=0.1, label="Guidance Scale")

    # outputs
    image_landmark = gr.Image(label="Landmark Image", type="pil")
    image_output = gr.Image(label="Generated Image", type="pil")

    demo = gr.Interface(
        fn=controlnet_landmark_pipeline,
        inputs=[image_processor, pipe, prompt_input, negative_prompt_input, image_input, num_steps_input, controlnet_scale_input, guidance_scale_input],
        outputs=[image_output, image_landmark],
        title="Stable Diffusion Image Generator",
        description="Generate images using Stable Diffusion with ControlNet."
    )

    return demo

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # lora_path = "latent-consistency/lcm-lora-sdv1-5"
    model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
    controlnet_path = "saeu5407/controlnet-landmark"
    dtype = torch.float16

    image_processor = ImageProcessor(facial_landmarks=True, face_detection=True)
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                             controlnet=controlnet,
                                                             torch_dtype=dtype,
                                                             use_safetensors=False,
                                                             ).to(device)
    # variant="fp16"
    # pipe.load_lora_weights(lora_path, adapter_name="lora1")
    # pipe.set_adapters("lora1")
    # pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Run Gradio
    demo = setup_gradio_interface()
    demo.launch(share=True)

    # test not use gradio
    '''
    a, b = controlnet_landmark_pipeline(image_processor, pipe, 
                  "",
                  "",
                  "dataset/sample2.png",
                  50,
                  1,
                  1)
    '''