MODEL_NAMES = [
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

AVAILABLE_RATIOS = {
    "1:1": 1,
    "1/1": 1,
    "21/9": 21 / 9,
    "21:9": 21 / 9,
    "16/10": 16 / 10,
    "16:10": 16 / 10,
    "16/9": 16 / 9,
    "16:9": 16 / 9,
    "4/3": 4 / 3,
    "4:3": 4 / 3,
    "3/2": 3 / 2,
    "3:2": 3 / 2,
}

SIZE_PRESET_MAPPING = {
    "MMS": 96,
    "QQVGA": 120,
    "vlow": 144,
    "low": 240,
    "CD": 360,
    "PSP": 272,
    "DVD": 480,
    "SD": 576,
    "HD": 720,
    "FHD": 1080,
    "UHD": 2160,
}
