import argparse
import os

import torch
from discord_webhook import DiscordWebhook, DiscordEmbed
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)
from rich.style import Style
from rich.table import Table

from lib import const
from lib.const import LOADING_TASK_STEPS
from lib.prompt_utils import make_table
from lib.trainer import Trainer

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prompts", required=True, help="Prompts")
parser.add_argument("-w", "--width", type=int, default=512, help="Width")
parser.add_argument("-H", "--height", type=int, default=512, help="Height")
parser.add_argument(
    "-m",
    "--model",
    default="sflckr",
    choices=const.MODEL_NAMES,
    help="Model",
)
parser.add_argument(
    "-f", "--display-freq", type=int, default=10, help="Display frequency"
)
parser.add_argument("-A", "--no-metadata", action="store_true", help="No metadata")
parser.add_argument("-a", "--author", default="VQGAN+CLIP", help="No metadata")
parser.add_argument("-i", "--initial-image", help="Initial image")
parser.add_argument("-t", "--target-images", help="Target images")
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
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console()
) as progress:
    loading_task = progress.add_task(f"Loading...", total=LOADING_TASK_STEPS)

    iteration_task = progress.add_task(
        f"Running {str(args.max_iterations)} iteration{'s' if args.max_iterations > 1 else ''}",
        start=False,
        visible=False,
        total=args.max_iterations,
    )

    row = []

    progress.console.rule("[[bold cyan] Parameters [/bold cyan]]")
    row.append(str(args.prompts))
    if args.initial_image:
        row.append(str(args.initial_image))
    else:
        row.append("-")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    row.append(str(device))
    row.append(str(args.max_iterations))

    row.append(str(args.width))
    row.append(str(args.height))
    row.append(str(args.display_freq))
    torch.manual_seed(args.seed)
    row.append(str(args.seed))
    progress.advance(loading_task)

    parameter_table = make_table(args.prompts, *row)
    progress.log(parameter_table, justify="center")

    progress.console.rule()
    trainer = Trainer(args, progress, loading_task, device)
    trainer.preflight()
    progress.update(loading_task, visible=False)

    progress.log("Loading [bold green]OK[/bold green]")

    progress.start_task(iteration_task)
    progress.update(iteration_task, visible=True)

    try:
        for iteration in range(args.max_iterations):
            trainer.train(iteration)
            progress.advance(iteration_task)
            progress.update(iteration_task, current_iteration=iteration)
        progress.update(iteration_task, completed=args.max_iterations)
        canceled = False
    except KeyboardInterrupt:
        canceled = True

with Console() as console:
    if canceled:
        console.out(
            "[!] ", end="", style=Style(color="yellow", bold=True), highlight=False
        )
        console.out("Canceled (っ ºДº)っ ︵ ⌨", highlight=False)
    else:
        console.out(
            "[OK] ", end="", style=Style(color="green", bold=True), highlight=False
        )
        console.out("Finished \( ﾟヮﾟ)/", highlight=False)

        DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK")
        if DISCORD_WEBHOOK:
            webhook = DiscordWebhook(
                url=DISCORD_WEBHOOK,
                username="Result")
            img_path = trainer.progress_img_path
            with open(img_path, "rb") as f:
                webhook.add_file(file=f.read(), filename="progress.jpg")

            embed = DiscordEmbed(title="Finished job", description=f"{str(args.prompts)} ({args.width}x{args.height}) - {args.max_iterations} iterations", color="42ba96")
            webhook.add_embed(embed)
            response = webhook.execute()
