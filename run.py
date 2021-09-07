import argparse
import signal

import torch
from rich import box
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.style import Style
from rich.table import Table

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
    TimeElapsedColumn(),
    console=Console(),
) as progress:
    loading_task = progress.add_task(f"Loading...", total=LOADING_TASK_STEPS)

    iteration_task = progress.add_task(
        f"Running {str(args.max_iterations)} iteration{'s' if args.max_iterations > 1 else ''}",
        start=False,
        visible=False,
        total=args.max_iterations,
    )

    parameter_table = Table(title="", box=box.MINIMAL)
    parameter_table.add_column(
        f"Text prompt{'s' if len(args.prompts) > 1 else ''}", style="green"
    )
    parameter_table.add_column(
        f"Target image{'s' if len(args.target_images) > 1 else ''}", style="green"
    )
    parameter_table.add_column("Device", style="cyan")
    parameter_table.add_column("Iterations", style="cyan")
    parameter_table.add_column("Width", style="cyan")
    parameter_table.add_column("Height", style="cyan")
    parameter_table.add_column("Display frequency", style="cyan")
    parameter_table.add_column("Seed", style="cyan")

    row = []

    progress.console.rule("[[bold cyan] Parameters [/bold cyan]]")
    row.append(str(args.prompts))
    if args.target_images:
        row.append(str(args.target_images))
    else:
        row.append("-")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    row.append(str(device))
    row.append(str(args.max_iterations))

    row.append(str(args.width))
    row.append(str(args.height))
    row.append(str(args.display_freq))
    torch.manual_seed(args.seed)
    # progress.log("Seed:", args.seed)
    row.append(str(args.seed))
    progress.advance(loading_task)

    parameter_table.add_row(*row)
    progress.log(parameter_table, justify="center")

    progress.console.rule("")
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
