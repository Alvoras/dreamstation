import argparse

import torch

from lib import const
from lib.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prompts", required=True, help="Prompts")
parser.add_argument("-w", "--width", type=int, default=512, help="Width")
parser.add_argument("-H", "--height", type=int, default=512, help="Height")
parser.add_argument(
    "-m",
    "--model",
    default="sflckr",
    choices=const.model_names,
    help="Model",
)
parser.add_argument(
    "-f", "--display-freq", type=int, default=10, help="Display frequency"
)
parser.add_argument("-A", "--no-metadata", action="store_true", help="No metadata")
parser.add_argument("-a", "--author", default="VQGAN+CLIP", help="No metadata")
parser.add_argument("-i", "--initial-image", help="Initial image")
parser.add_argument("-T", "--target-images", help="Target images")
parser.add_argument("-t", "--target-image", help="Target image")
parser.add_argument("-s", "--seed", type=int, default=torch.seed(), help="Seed")
parser.add_argument(
    "-M", "--max-iterations", type=int, default=1000, help="Max iteration"
)
args = parser.parse_args()

if args.target_images:
    args.target_images = [line.strip() for line in args.target_images.split("|")]

args.prompts = [line.strip() for line in args.prompts.split("|")]
args.target_images = (
    [line.strip() for line in args.target_images.split("|")]
    if args.target_images
    else []
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if args.prompts:
    print("Using text prompt:", args.prompts)

if args.target_images:
    print("Using image prompts:", args.target_images)

torch.manual_seed(args.seed)
print("Using seed:", args.seed)

trainer = Trainer(args, device)
trainer.preflight()

try:
    for iteration in range(args.max_iterations):
        trainer.train(iteration)
except KeyboardInterrupt:
    pass
