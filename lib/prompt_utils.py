from rich import box
from rich.table import Table


def parse_prompt(prompt):
    vals = prompt.rsplit(":", 2)
    vals = vals + ["", "1", "-inf"][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


def make_table(*rows):
    parameter_table = Table(title="", box=box.MINIMAL)
    parameter_table.add_column("Text prompt", style="green")
    parameter_table.add_column(f"Initial image", style="green")
    parameter_table.add_column("Device", style="cyan")
    parameter_table.add_column("Iterations", style="cyan")
    parameter_table.add_column("Repeats", style="cyan")
    parameter_table.add_column("Width", style="cyan")
    parameter_table.add_column("Height", style="cyan")
    parameter_table.add_column("Display freq.", style="cyan")
    parameter_table.add_column("Progress freq.", style="cyan")
    parameter_table.add_column("Seed", style="cyan")
    parameter_table.add_row(*rows)
    return parameter_table
