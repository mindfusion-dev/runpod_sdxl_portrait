import torch
from diffusers import (DDIMScheduler,
                       AutoencoderKL,
                       StableDiffusionXLPipeline,
                       ControlNetModel,
                       StableDiffusionXLControlNetPipeline)
from controlnet_aux import OpenposeDetector

from huggingface_hub import hf_hub_download


def fetch_instantid_checkpoints():
    """
    Fetches InstantID checkpoints from the HuggingFace model hub.
    """

    hf_hub_download(
        repo_id='h94/IP-Adapter-FaceID',
        filename='ip-adapter-faceid-portrait_sdxl.bin',
        local_dir='./',
        local_dir_use_symlinks=False
    )

    # DL Loras
    hf_hub_download(
        repo_id='artificialguybr/3DRedmond-V1',
        filename='3DRedmond-3DRenderStyle-3DRenderAF.safetensors',
        local_dir='./loras',
        local_dir_use_symlinks=False
    )

    hf_hub_download(
        repo_id='artificialguybr/PixelArtRedmond',
        filename='PixelArtRedmond-Lite64.safetensors',
        local_dir='./loras',
        local_dir_use_symlinks=False
    )

    hf_hub_download(
        repo_id='artificialguybr/ClayAnimationRedmond',
        filename='ClayAnimationRedm.safetensors',
        local_dir='./loras',
        local_dir_use_symlinks=False
    )

    hf_hub_download(
        repo_id='sintecs/SDXL_Loras',
        filename='DuneStylev1.0.safetensors',
        local_dir='./loras',
        local_dir_use_symlinks=False
    )

    hf_hub_download(
        repo_id='ProomptEngineer/pe-neon-sign-style',
        filename='PE_NeonSignStyle.safetensors',
        local_dir='./loras',
        local_dir_use_symlinks=False
    )

    hf_hub_download(
        repo_id='BlaireSilver13/dollx_style',
        filename='xdlx_style.safetensors',
        local_dir='./loras',
        local_dir_use_symlinks=False
    )

    hf_hub_download(
        repo_id="nerijs/pixel-art-xl",
        filename="pixel-art-xl.safetensors",
        local_dir="./loras",
    )

    hf_hub_download(
        repo_id="Fictiverse/Voxel_XL_Lora",
        filename="VoxelXL_v1.safetensors",
        local_dir="./loras",
    )

    hf_hub_download(
        repo_id="Akbartus/Medieval-Illustration-Lora",
        filename="vintage_illust.safetensors",
        local_dir="./loras",
    )

    hf_hub_download(
        repo_id="KappaNeuro/stop-motion-animation",
        filename="Stop-Motion Animation.safetensors",
        local_dir="./loras",
    )

    hf_hub_download(
        repo_id="KappaNeuro/surreal-collage",
        filename="Surreal Collage.safetensors",
        local_dir="./loras",
    )

    hf_hub_download(
        repo_id='sintecs/SDXL_Loras',
        filename="Ath_stuffed-toy_XL.safetensors",
        local_dir="./loras",
    )

    hf_hub_download(
        repo_id='sintecs/SDXL_Loras',
        filename="Cute_Collectible.safetensors",
        local_dir="./loras",
    )

    hf_hub_download(
        repo_id='sintecs/SDXL_Loras',
        filename="EldritchComicsXL1.2.safetensors",
        local_dir="./loras",
    )

    hf_hub_download(
        repo_id='sintecs/SDXL_Loras',
        filename="Graphic_Portrait.safetensors",
        local_dir="./loras",
    )

    hf_hub_download(
        repo_id='sintecs/SDXL_Loras',
        filename="J_cartoon.safetensors",
        local_dir="./loras",
    )

    hf_hub_download(
        repo_id='sintecs/SDXL_Loras',
        filename="Lucasarts.safetensors",
        local_dir="./loras",
    )

    hf_hub_download(
        repo_id='sintecs/SDXL_Loras',
        filename="SDXL_MSPaint_Portrait.safetensors",
        local_dir="./loras",
    )

    hf_hub_download(
        repo_id='sintecs/SDXL_Loras',
        filename="SouthParkRay.safetensors",
        local_dir="./loras",
    )

    hf_hub_download(
        repo_id='sintecs/SDXL_Loras',
        filename="Vintage_Street_Photo.safetensors",
        local_dir="./loras",
    )

    hf_hub_download(
        repo_id='sintecs/SDXL_Loras',
        filename="poluzzle.safetensors",
        local_dir="./loras",
    )

    hf_hub_download(
        repo_id='sintecs/SDXL_Loras',
        filename="sketch_it.safetensors",
        local_dir="./loras",
    )

    hf_hub_download(
        repo_id='sintecs/SDXL_Loras',
        filename="vapor_graphic_sdxl.safetensors",
        local_dir="./loras",
    )

    hf_hub_download(
        repo_id='sintecs/SDXL_Loras',
        filename="y2k3dnerdessence_v0.0.1.safetensors",
        local_dir="./loras",
    )


def fetch_pretrained_model(model_name, **kwargs):
    """
    Fetches a pretrained model from the HuggingFace model hub.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return StableDiffusionXLControlNetPipeline.from_pretrained(
                model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f'Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...')
            else:
                raise


def get_instantid_pipeline():
    """
    Fetches the InstantID pipeline from the HuggingFace model hub.
    """
    torch_dtype = torch.float16
    OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    controlnet = ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0",
        torch_dtype=torch.float16)
    args = {
        'scheduler': DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        ),
        'vae': AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch_dtype),
        'torch_dtype': torch_dtype,
        'add_watermarker': False,
        'controlnet': controlnet,
    }

    pipeline = fetch_pretrained_model('frankjoshua/albedobaseXL_v13', **args)

    return pipeline


if __name__ == '__main__':
    fetch_instantid_checkpoints()
    get_instantid_pipeline()
