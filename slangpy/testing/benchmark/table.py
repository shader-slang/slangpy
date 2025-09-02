# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from _pytest._io import TerminalWriter

from .report import BenchmarkReport

from typing import Optional, Tuple

Part = Tuple[str, dict[str, bool]]
Cell = list[Part]


def display(
    writer: TerminalWriter,
    benchmarks: list[BenchmarkReport],
    baseline_benchmarks: Optional[list[BenchmarkReport]] = None,
):
    baseline_benchmarks_by_name = {
        benchmark["name"]: benchmark for benchmark in (baseline_benchmarks or [])
    }

    # For table printing we use the following conventions:
    # - a row is a list of cells
    # - a cell is a list of parts
    # - a part is a tuple of (text, markup)
    # - no spaces are used to separate parts
    # - 3 spaces are used to separate cells
    # When printing the table, we align the text within each cell.
    # Parts can have individual markup (color, bold etc.) passed to pytest's TerminalWriter.

    def make_part(text: str, **markup: bool) -> Part:
        return (text, markup)

    def make_cell(text: str) -> Cell:
        return [make_part(text)]

    def cell_length(cell: Cell) -> int:
        return sum(len(part) for part, _ in cell) + len(cell)

    column_titles = [
        make_cell("Name"),
        make_cell("Min (ms)"),
        make_cell("Max (ms)"),
        make_cell("Mean (ms)"),
        make_cell("Median (ms)"),
        make_cell("Stddev (ms)"),
    ]
    rows = []

    for benchmark in benchmarks:
        row = [
            make_cell(benchmark["name"]),
            make_cell(f"{benchmark['min']:.3f}"),
            make_cell(f"{benchmark['max']:.3f}"),
            make_cell(f"{benchmark['mean']:.3f}"),
            make_cell(f"{benchmark['median']:.3f}"),
            make_cell(f"{benchmark['stddev']:.3f}"),
        ]
        if benchmark["name"] in baseline_benchmarks_by_name:
            baseline_benchmark = baseline_benchmarks_by_name[benchmark["name"]]

            def delta_info(current: float, baseline: float) -> Part:
                if baseline == 0:
                    return make_part(" (inf%)")
                delta = ((current - baseline) / baseline) * 100
                markup = {}
                if delta < -5:
                    markup = {"green": True}
                elif delta > 5:
                    markup = {"red": True}
                else:
                    markup = {"light": True}
                return make_part(f" ({delta:+.1f}%)", **markup)

            row[1].append(delta_info(benchmark["min"], baseline_benchmark["min"]))
            row[2].append(delta_info(benchmark["max"], baseline_benchmark["max"]))
            row[3].append(delta_info(benchmark["mean"], baseline_benchmark["mean"]))
            row[4].append(delta_info(benchmark["median"], baseline_benchmark["median"]))
            row[5].append(delta_info(benchmark["stddev"], baseline_benchmark["stddev"]))

        rows.append(row)

    column_widths = [
        max(cell_length(item) for item in col) + 3 for col in zip(*([column_titles] + rows))
    ]

    def write_row(row: list[Cell]):
        for column_index, cell in enumerate(row):
            length = 0
            for part in cell:
                writer.write(f"{part[0]}", **part[1])
                length += len(part[0])
            writer.write(" " * (column_widths[column_index] - length))
        writer.line()

    writer.sep("-")
    write_row(column_titles)
    writer.sep("-")
    for row in rows:
        write_row(row)
    writer.sep("-")
