import argparse
import os
from math import ceil

import torch
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.style import Style

from lib import const
from lib.config import AUTHOR, DISCORD_WEBHOOK
from lib.const import (
    LOADING_TASK_STEPS,
    AVAILABLE_RATIOS,
    SIZE_PRESET_MAPPING,
)
from lib.prompt_utils import make_table
from lib.trainer import Trainer

parser = argparse.ArgumentParser()
mutually_exclusive = parser.add_mutually_exclusive_group(required=True)
mutually_exclusive.add_argument("-p", "--prompts", action="append", help="Prompts")
mutually_exclusive.add_argument(
    "--test",
    action="store_true",
    help="Macro to test the program with default parameters",
)
parser.add_argument("-o", "--out", default="steps", help="Output directory")
parser.add_argument("-w", "--width", type=int, default=512, help="Image width")
parser.add_argument("-H", "--height", type=int, default=512, help="Image height")
parser.add_argument(
    "-r",
    "--ratio",
    choices=AVAILABLE_RATIOS.keys(),
    help="Aspect ratio to use when computing the smaller side",
)
parser.add_argument(
    "--preset",
    choices=SIZE_PRESET_MAPPING.keys(),
    help="Image size preset to use",
)
parser.add_argument(
    "--portrait",
    action="store_true",
    help="Use portrait orientation instead of landscape to get the smaller side",
)
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
parser.add_argument(
    "--discord-freq", type=int, default=50, help="Display frequency for Discord updates"
)
parser.add_argument(
    "-A",
    "--no-metadata",
    action="store_true",
    help="Don't save metadata",
)
parser.add_argument(
    "-S",
    "--no-stegano",
    action="store_true",
    help="Don't etch metadata with LSB in the generated images",
)
parser.add_argument(
    "-u",
    "--discord-update",
    action="store_true",
    help="Send a progress picture via discord every '--discord-freq' iterations. /!\ A DISCORD_WEBHOOK environment variable is needed /!\\",
)
parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
parser.add_argument(
    "--gpu",
    help="Force GPU usage. You must specify the device to use (ie. '--gpu cuda:0')",
)
parser.add_argument("-a", "--author", default="VQGAN+CLIP", help="No metadata")
parser.add_argument("-i", "--initial-image", help="Initial image")
parser.add_argument("-t", "--target-images", help="Target images")
parser.add_argument("-s", "--seed", type=int, default=torch.seed(), help="Seed")
parser.add_argument(
    "-M", "--max-iterations", type=int, default=300, help="Max iterations"
)
args = parser.parse_args()

size_override = False
has_preset = False

# Disable discord update if no webhook has been specified
if args.discord_update and not DISCORD_WEBHOOK:
    args.discord_update = False

if not args.preset:
    if args.ratio:
        size_override = True

        if args.portrait:
            args.width = ceil(args.height / AVAILABLE_RATIOS[args.ratio])
        else:
            args.height = ceil(args.width / AVAILABLE_RATIOS[args.ratio])

    if args.height > args.width:
        size_override = True
        args.height = ceil(args.width * AVAILABLE_RATIOS[args.ratio])
    elif args.width > args.height and args.portrait:
        size_override = True
        args.width = ceil(args.height * AVAILABLE_RATIOS[args.ratio])
else:
    has_preset = True
    ratio = args.ratio if args.ratio else "1:1"
    args.width = SIZE_PRESET_MAPPING[args.preset]
    args.height = SIZE_PRESET_MAPPING[args.preset]

    if args.portrait:
        args.height = ceil(args.width * AVAILABLE_RATIOS[ratio])
    else:
        args.width = ceil(args.height * AVAILABLE_RATIOS[ratio])

if args.test:
    args.width = 64
    args.height = 64
    args.max_iterations = 3
    args.display_freq = 1
    args.prompts = ["test"]

# Command line takes precedence
if (
    args.author == "VQGAN+CLIP" and AUTHOR
):  # Only if args.author is at its default value
    args.author = AUTHOR

if args.target_images:
    args.target_images = [line.strip() for line in args.target_images.split("|")]

for idx, prompt in enumerate(args.prompts):
    args.prompts[idx] = [line.strip() for line in prompt.split("|")]

args.target_images = (
    [line.strip() for line in args.target_images.split("|")]
    if args.target_images
    else []
)

if args.cpu:
    device = "cpu"
elif args.gpu:
    device = args.gpu
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)

for prompt in args.prompts:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(),
    ) as progress:
        loading_task = progress.add_task(f"Loading...", total=LOADING_TASK_STEPS)
        iteration_task = progress.add_task(
            f"Running {args.max_iterations} iteration{'s' if args.max_iterations > 1 else ''}",
            start=False,
            visible=False,
            total=args.max_iterations,
        )

        progress.console.rule("[[bold cyan] Parameters [/bold cyan]]")

        row = []
        row.append(str(prompt))
        if args.initial_image:
            row.append(str(args.initial_image))
        else:
            row.append("-")

        row.append(str(device))
        row.append(str(args.max_iterations))
        row.append(str(args.width))
        row.append(str(args.height))
        row.append(str(args.display_freq))
        row.append(str(args.seed))
        progress.advance(loading_task)

        parameter_table = make_table(prompt, *row)
        progress.log(parameter_table, justify="center")

        if size_override and args.portrait:
            progress.log(
                f"Width set to [bold]{args.width}px[/bold] (using ratio [bold cyan]{args.ratio}[/bold cyan])",
                highlight=False,
            )
        elif size_override:
            progress.log(
                f"Height set to [bold]{args.height}px[/bold] (using ratio [bold cyan]{args.ratio}[/bold cyan])",
                highlight=False,
            )

        if has_preset:
            progress.log(
                f"Preset override (using ratio [bold cyan]{args.ratio}[/bold cyan]) :\n\t Width = [bold]{args.width}px[/bold] \n\t Height = [bold]{args.height}px[/bold]",
                highlight=False,
            )

        progress.console.rule()

        trainer = Trainer(args, prompt, progress, loading_task, device)
        trainer.preflight()
        # progress.update(loading_task, visible=False)

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

    # Recreate a console to print below the progress bar
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

            trainer.push_finish()
