INPUT_SCHEMA = {
    'image_url': {
        'type': str,
        'required': True,
    },
    'pose_image': {
        'type': str,
        'required': False,
        'default': None
    },
    "inference_steps": {
        'type': int,
        'required': False,
        'default': 50
    },
    "scale": {
        'type': float,
        'required': False,
        'default': 0.8
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 5
    },
    'prompt': {
        'type': str,
        'required': False,
        'default': 'a person'
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    "embeds_path": {
        'type': str,
        'required': False,
        'default': None
    },
    'batch': {
        'type': int,
        'required': False,
        'default': 1
    },
    "seed": {
        'type': int,
        'required': False,
        'default': None
    },
    'width': {
        'type': int,
        'required': False,
        'default': 880
    },
    'height': {
        'type': int,
        'required': False,
        'default': 1200
    },
    "style": {
        'type': str,
        'required': False,
        'default': '3D'
    },
    'style_name': {
        'type': str,
        'required': False,
        'default': 'Watercolor'
    }
}
