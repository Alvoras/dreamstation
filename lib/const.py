model_names = [
    "vqgan_imagenet_f16_16384",
    "vqgan_imagenet_f16_1024",
    "wikiart_1024",
    "wikiart_16384",
    "coco",
    "faceshq",
    "sflckr",
]

STEP_SIZE = 0.1
CUTN = 64
CUT_POW = 1.0
INIT_WEIGHT = 0.0
CLIP_MODEL = "ViT-B/32"

NOISE_PROMPT_SEED = []
NOISE_PROMPT_WEIGHTS = []

LOADING_TASK_STEPS = 3
