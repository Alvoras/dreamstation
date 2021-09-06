import argparse
import shutil

import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from lib import const
from lib.const import LOADING_TASK_STEPS
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
    "-M", "--max-iterations", type=int, default=1000, help="Max iterations"
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

with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=Console()
) as progress:
    loading_task = progress.add_task(
        f"Loading...",
        total=LOADING_TASK_STEPS
    )

    iteration_task = progress.add_task(
        f"Running {str(args.max_iterations)} iteration{'s' if args.max_iterations > 1 else ''}",
        start=False,
        visible=False,
        total=args.max_iterations
    )

    progress.log(f"Iterations : {args.max_iterations}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    progress.log(f"Device : [bold]{device}[/bold]")
    progress.log("Text prompt:", args.prompts)
    if args.target_images:
        progress.log("Image prompts:", args.target_images)

    torch.manual_seed(args.seed)
    progress.log("Seed:", args.seed)
    progress.advance(loading_task)

    trainer = Trainer(args, progress, loading_task, device)
    trainer.preflight()
    progress.update(loading_task, completed=LOADING_TASK_STEPS)
    progress.update(loading_task, visible=False)

    progress.start_task(iteration_task)
    progress.update(iteration_task, visible=True)
    try:
        for iteration in range(args.max_iterations):
            trainer.train(iteration)
            progress.advance(iteration_task)
        progress.update(iteration_task, completed=args.max_iterations)

    except KeyboardInterrupt:
        pass
