import torch
from diffusers import DDIMScheduler, AutoencoderKL, StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download


def fetch_instantid_checkpoints():
    """
    Fetches InstantID checkpoints from the HuggingFace model hub.
    """

    hf_hub_download(
        repo_id='h94/IP-Adapter-FaceID',
        filename='ip-adapter-faceid-portrait_sdxl.bin',
        local_dir='./models',
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
        repo_id='artificialguybr/ps1redmond-ps1-game-graphics-lora-for-sdxl',
        filename='PS1Redmond-PS1Game-Playstation1Graphics.safetensors',
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
        repo_id='artificialguybr/ToyRedmond-ToyLoraForSDXL10',
        filename='ToyRedmond-FnkRedmAF.safetensors',
        local_dir='./loras',
        local_dir_use_symlinks=False
    )

    hf_hub_download(
        repo_id='sintecs/SDXL_Loras',
        filename='DuneStylev1.0.safetensors',
        local_dir='./loras',
        local_dir_use_symlinks=False
    )


def fetch_pretrained_model(model_name, **kwargs):
    """
    Fetches a pretrained model from the HuggingFace model hub.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return StableDiffusionXLPipeline.from_pretrained(model_name, **kwargs)
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
    # depth-zoe-xl-v1.0-controlnet.safetensors
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
    }

    pipeline = fetch_pretrained_model('frankjoshua/albedobaseXL_v13', **args)

    return pipeline


if __name__ == '__main__':
    fetch_instantid_checkpoints()
    get_instantid_pipeline()
