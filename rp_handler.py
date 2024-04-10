import gc
import os
import time
import torch
import cv2
import math
import random
import numpy as np
import requests
import base64
import traceback

from PIL import Image, ImageOps

import diffusers
from diffusers import (DDIMScheduler,
                       AutoencoderKL,
                       ControlNetModel,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler,
                       EulerDiscreteScheduler,
                       HeunDiscreteScheduler,
                       PNDMScheduler,
                       StableDiffusionXLControlNetPipeline)

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_download import file
from runpod.serverless.modules.rp_logger import RunPodLogger
from controlnet_aux import OpenposeDetector

from utils import FaceidAcquirer
from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceIDXL

from io import BytesIO
from schemas.input import INPUT_SCHEMA
from style_template import styles
from model_util import get_torch_device
import GPUtil


# Global variables
MAX_SEED = np.iinfo(np.int32).max
device = get_torch_device()
dtype = torch.float16 if str(device).__contains__('cuda') else torch.float32
STYLE_NAMES = list(styles.keys())
DEFAULT_MODEL = 'frankjoshua/albedobaseXL_v13'
DEFAULT_STYLE_NAME = 'Watercolor'


# Load face encoder
class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config,
                                                       use_karras_sigmas=True)


def print_gpu_info(step: str):
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"STEP: {step}")
        print(f"GPU: {gpu.id}, Name: {gpu.name}")
        print(f"\tTotal Memory: {gpu.memoryTotal}MB")
        print(f"\tUsed Memory: {gpu.memoryUsed}MB")
        print(f"\tFree Memory: {gpu.memoryFree}MB")
        print(f"\tGPU Load: {gpu.load * 100}%")


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

LORA_WEIGHTS_MAPPING = {
    "3D": "./loras/3DRedmond-3DRenderStyle-3DRenderAF.safetensors",
    "Emoji": "./loras/emoji.safetensors",
    "Pixels": "./loras/PixelArtRedmond-Lite64.safetensors",
    "Clay": "./loras/ClayAnimationRedm.safetensors",
    "Dune": "./loras/DuneStylev1.0.safetensors",
    "Neon": "./loras/PE_NeonSignStyle.safetensors",
    "dollx": "./loras/xdlx_style.safetensors",
    "PixelArt": "./loras/pixel-art-xl.safetensors",
    "Voxel": "./loras/VoxelXL_v1.safetensors",
    "Midieval": "./loras/vintage_illust.safetensors",
    "stop_motion": "./loras/Stop-Motion Animation.safetensors",
    "surreal": "./loras/Surreal Collage.safetensors",
    "stuffed_toy": "./loras/Ath_stuffed-toy_XL.safetensors",
    "cute_collect": "./loras/Cute_Collectible.safetensors",
    "comics": "./loras/EldritchComicsXL1.2.safetensors",
    "graphic_portrait": "./loras/Graphic_Portrait.safetensors",
    "cartoon": "./loras/J_cartoon.safetensors",
    "Lucasarts": "./loras/Lucasarts.safetensors",
    "mspaint": "./loras/SDXL_MSPaint_Portrait.safetensors",
    "southpark": "./loras/SouthParkRay.safetensors",
    "vintage": "./loras/Vintage_Street_Photo.safetensors",
    "poluzzle": "./loras/poluzzle.safetensors",
    "sketch": "./loras/sketch_it.safetensors",
    "vapor": "./loras/vapor_graphic_sdxl.safetensors",
    "oldgame": "./loras/y2k3dnerdessence_v0.0.1.safetensors",
}


def load_image(image_file: str):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content))
    else:
        image = load_image_from_base64(image_file)

    image = ImageOps.exif_transpose(image)
    image = image.convert('RGB')
    return image


def load_image_from_base64(base64_str: str):
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_bytes))
    return image


def determine_file_extension(image_data):
    image_extension = None

    try:
        if image_data.startswith('/9j/'):
            image_extension = '.jpg'
        elif image_data.startswith('iVBORw0Kg'):
            image_extension = '.png'
        else:
            # Default to png if we can't figure out the extension
            image_extension = '.png'
    except Exception:
        image_extension = '.png'

    return image_extension


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


def resize_img(input_image, max_side=1280, min_side=1024, size=None,
               pad_to_max_side=False, mode=Image.Resampling.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


def get_depth_map(image, depth_estimator):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    return depth_map


noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                    torch_dtype=torch.float16,
                                    use_safetensors=True)

openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

controlnet = ControlNetModel.from_pretrained(
    "thibaud/controlnet-openpose-sdxl-1.0",
    torch_dtype=torch.float16)

CURRENT_STYLE = "3D"
PIPELINE: StableDiffusionXLControlNetPipeline = \
    StableDiffusionXLControlNetPipeline.from_pretrained(
        DEFAULT_MODEL,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        add_watermarker=False,
        vae=vae,
        controlnet=controlnet
    ).to(device)

PIPELINE.scheduler = SCHEDULERS["KarrasDPM"].from_config(
        PIPELINE.scheduler.config)

PIPELINE.load_lora_weights(
    LORA_WEIGHTS_MAPPING.get(CURRENT_STYLE))

PIPELINE.fuse_lora()


app = FaceidAcquirer()

logger = RunPodLogger()

# ---------------------------------------------------------------------------- #
# Application Functions                                                        #
# ---------------------------------------------------------------------------- #


def apply_style(style_name: str,
                positive: str,
                negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace('{prompt}', positive), n + ' ' + negative


def predict(
        job_id: int,
        image_url: str = None,
        pose_img_path: str = None,
        inference_steps: int = 50,
        scale: float = 0.8,
        guidance_scale: float = 8,
        prompt: str = "Your default prompt",
        negative_prompt: str = "Your default negative prompt",
        embeds_path: str = None,
        batch: int = 1,
        seed: int = None,
        width: int = 880,
        height: int = 1200,
        style="3D",
        ):
    with torch.no_grad():
        print_gpu_info("START PREDICT")
        global CURRENT_STYLE, PIPELINE
        if style != CURRENT_STYLE:
            start_time = time.time()
            PIPELINE.unfuse_lora()
            PIPELINE.unload_lora_weights()
            PIPELINE.load_lora_weights(LORA_WEIGHTS_MAPPING.get(style))
            PIPELINE.fuse_lora(lora_scale=1)
            CURRENT_STYLE = style
            print(f"LORA change time: {time.time() - start_time}")
            print_gpu_info("UNFUSE AND LOAD NEW LORA")

        image = file(image_url)
        img_path = image["file_path"]
        faceid_embeds = app.get_multi_embeds(img_path)
        n_cond = faceid_embeds.shape[1]

        pose_image = diffusers.utils.load_image(image_url)
        openpose_image = openpose(pose_image)

        ip_model: IPAdapterFaceIDXL = IPAdapterFaceIDXL(
            PIPELINE,
            "ip-adapter-faceid-portrait_sdxl.bin",
            device,
            16,
            n_cond=n_cond)

        if embeds_path is None:
            suffix = os.path.basename(img_path).split('.')[1]
            txt_path = img_path.replace(suffix, 'txt')
        else:
            txt_path = embeds_path.replace('npy', 'txt')
        print(f"txt_path:{txt_path}")

        if os.path.exists(txt_path):
            with open(txt_path, 'r')as f:
                prompt = f.readlines()[0]
        prompt = prompt if prompt is None else prompt
        # prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

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
            image=openpose_image,
        )
    print_gpu_info("GENERATED")
    torch.cuda.empty_cache()
    gc.collect()

    return images


def handler(job):
    try:
        validated_input = validate(job['input'], INPUT_SCHEMA)

        if 'errors' in validated_input:
            return {
                'error': validated_input['errors']
            }

        payload = validated_input['validated_input']

        images = predict(
            job['id'],
            payload.get('image_url'),
            payload.get('pose_image'),
            payload.get('inference_steps'),
            payload.get('scale'),
            payload.get('guidance_scale'),
            payload.get('prompt'),
            payload.get('negative_prompt'),
            payload.get('embeds_path'),
            payload.get('batch'),
            payload.get('seed'),
            payload.get('width'),
            payload.get('height'),
            payload.get('style'),
        )

        result_image = images[0]
        output_buffer = BytesIO()
        result_image.save(output_buffer, format='JPEG')
        image_data = output_buffer.getvalue()

        return {
            'image': base64.b64encode(image_data).decode('utf-8')
        }
    except Exception as e:
        logger.error(f'An exception was raised: {e}')

        return {
            'error': str(e),
            'output': traceback.format_exc(),
            'refresh_worker': True
        }


# ---------------------------------------------------------------------------- #
# RunPod Handler                                                               #
# ---------------------------------------------------------------------------- #
if __name__ == '__main__':
    logger.info('Starting RunPod Serverless...')
    runpod.serverless.start(
        {
            'handler': handler
        }
    )
