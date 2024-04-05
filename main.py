from typing import List
import torch
from diffusers import (StableDiffusionPipeline,
                       DDIMScheduler,
                       AutoencoderKL,
                       ControlNetModel,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler,
                       EulerDiscreteScheduler,
                       HeunDiscreteScheduler,
                       PNDMScheduler,
                       StableDiffusionXLControlNetPipeline,
                       StableDiffusionXLPipeline,
                       UniPCMultistepScheduler)
import numpy as np
import os
import sys
from diffusers.utils import load_image
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from utils import FaceidAcquirer, image_grid
from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID, IPAdapterFaceIDXL
# from transformers import pipeline
import uuid
import os


class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config,
                                                       use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

# base_model_path = "Norod78/sd-simpsons-model"  # "SG161222/Realistic_Vision_V4.0_noVAE"  # f"{source_dir}/SG161222--Realistic_Vision_V4.0_noVAE/"
# vae_model_path = "stabilityai/sd-vae-ft-mse"  # f"{source_dir}/stabilityai--sd-vae-ft-mse/"
# ip_ckpt = "models/ip-adapter-faceid-portrait-v11_sd15.bin"  # f"{source_dir}/h94--IP-Adapter/h94--IP-Adapter/models/ip-adapter-faceid-portrait_sd15.bin"
device = "cuda"
# stablediffusionapi/xxmix9realistic


def get_depth_map(image, depth_estimator):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    return depth_map


class Predictor:
    def setup(self):
        base_model_path = "SG161222/RealVisXL_V3.0"

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            # base_model_path,
            # "rubbrband/albedobaseXL_v21",
            # "krnl/realisticVisionV51_v51VAE",
            "stablediffusionapi/juggernaut-xl-v9",
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            add_watermarker=False,
            vae=vae
        )

        self.pipe.scheduler = SCHEDULERS["KarrasDPM"].from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights("models/Dune.safetensors", adapter_name="Dune")
        self.pipe.load_lora_weights("models/details_.safetensors", adapter_name="details")
        self.pipe.set_adapters(["Dune", "details"], adapter_weights=[0.8, 0.8])
        # self.pipe.load_lora_weights("models/Dune_Movie_Loha2.safetensors")

        # self.pipe.fuse_lora(lora_scale=0.8)
        self.app = FaceidAcquirer()  # Инициализируйте вашу собственную реализацию, если это необходимо

    def predict(
        self,
        input_path: str = None,
        pose_img_path: str = None,
        inference_steps: int = 50,
        scale: float = 0.8,
        guidance_scale: float = 8,
        save_path: str = "_output",
        prompt: str = "Your default prompt",
        negative_prompt: str = "Your default negative prompt",
        embeds_path: str = None,
        batch: int = 1,
        seed: int = None,
        width: int = 880,
        height: int = 1200
    ):
        # pose_image = load_image(input_path)
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        if embeds_path is None:
            image_paths = input_path
            faceid_embeds = self.app.get_multi_embeds(image_paths)
            n_cond = faceid_embeds.shape[1]
        else:
            print(f"loading embeds path ==>{embeds_path}")
            faceid_embeds = np.load(embeds_path)
            if len(faceid_embeds.shape) == 2 and len(faceid_embeds.shape) != 3:
                faceid_embeds = torch.from_numpy(faceid_embeds).unsqueeze(0)
            else:
                ValueError(f"faceid_embeds shape is error ==> {faceid_embeds.shape}")
            n_cond = faceid_embeds.shape[1]
        print(faceid_embeds.shape)

        # load ip-adapter
        # ip_model: IPAdapterFaceID = IPAdapterFaceID(
        #     self.pipe,
        #     "models/ip-adapter-faceid-portrait_sdxl.bin",
        #     device,
        #     num_tokens=16,
        #     n_cond=n_cond)
        ip_model: IPAdapterFaceIDXL = IPAdapterFaceIDXL(
            self.pipe,
            "models/ip-adapter-faceid-portrait_sdxl.bin",
            device,
            16,
            n_cond=n_cond)

        if embeds_path is None:
            suffix = os.path.basename(image_paths[0]).split('.')[1]
            txt_path = image_paths[0].replace(suffix, 'txt')
        else:
            txt_path = embeds_path.replace('npy', 'txt')
        print(f"txt_path:{txt_path}")

        if os.path.exists(txt_path):
            with open(txt_path, 'r')as f:
                prompt = f.readlines()[0]
        prompt = prompt if prompt is None else prompt

        print(f"prompt:{prompt}\n negative_prompt:{negative_prompt}")
        images = ip_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            faceid_embeds=faceid_embeds,
            num_samples=batch,
            width=width, height=height,
            num_inference_steps=inference_steps,
            seed=seed,
            scale=scale,
            guidance_scale=guidance_scale,
            s_scale=1,
            # image=pose_image,
        )
        grid = image_grid(images, int(batch**0.5), int(batch**0.5))
        output_img_path = f"./{uuid.uuid4()}.jpg"
        grid.save(output_img_path)
        print(f"result has saved in {save_path}")
        output_paths = []

        return output_paths


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    # Вызовите функцию predict с аргументами по умолчанию или задайте свои значения
    predictor.predict(
        input_path=["1.jpg", "4.jpg"],
        # input_path=["3.png"],
        pose_img_path="path_to_pose_image.jpg",
        inference_steps=30,
        scale=0.5,
        guidance_scale=8,
        save_path="_output",
        prompt="1man, 7-dune, aircraft, architecture, beard, blurry, blurry background, depth of field, dune environment 1girl, formal, letterboxed, monochrome, portrait, realistic, solo, street [reflections, realistic lighting, light rays, beams of light, realistic, high quality photo, 4k, 7-dune] <lora:Dune Style v1.0:0.8> <lora:add-detail-xl:0.6> <lora:xl_more_art-full_v1:0.4>",
        # prompt="breathtaking cinematic photo Dune, Movie, director Denis Villeneuve, cinematic, science fiction, futuristic, sci-fi epic, wearing fremen armor, (stillsuit:1.2), rugged dessert landscape, striking blue eyes, , In the vast expanse of the desert under a clear sky. He is clad in Fremen armor while standing on a rocky outcropping amidst the sandy dunes. making him stand out prominently in this cinematic science fiction epic.  . 35mm photograph, film, bokeh, professional, 4k, highly detailed . award-winning, professional, highly detailed",
        negative_prompt="angry, evil, wrinkles on the forehead, ugly, deformed, noisy, blurry, distorted, grainy, drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, bad quality, low quality, worst quality, (hands:1.3), cartoon, lowres, meme, blurry",
        embeds_path=None,
        batch=1,
        seed=None,
        width=1024,
        height=1024
    )

# docker build --tag sintecs/business_suit_faceid_portrait:latest --build-arg COG_REPO=mindfusion-dev --build-arg COG_MODEL=business_suit_faceid_portrait --build-arg COG_VERSION=85fdfc6ab898ea13c19b3c295d8797944a70bf0d9ee3c90f0757a9f114c1ab40 . 

# docker build --tag sintecs/business_girls_suite_faceid_portrait:latest --build-arg COG_REPO=mindfusion-dev --build-arg COG_MODEL=business_girls_suite_faceid_portrait --build-arg COG_VERSION=a23f91f1c395bb4dc2548fac0aa9d2abb1e412bcb5693ab6f5d012f1a040af05 . 