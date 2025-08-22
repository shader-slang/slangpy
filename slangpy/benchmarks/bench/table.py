# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from bench.report import BenchmarkReport


def display(benchmark_reports: list[BenchmarkReport]):

    column_titles = ["Name", "Min (ms)", "Max (ms)", "Mean (ms)", "Median (ms)", "Stddev (ms)"]
    rows = []

    for report in benchmark_reports:
        row = [
            report["name"],
            f"{report['min']:.3f}",
            f"{report['max']:.3f}",
            f"{report['mean']:.3f}",
            f"{report['median']:.3f}",
            f"{report['stddev']:.3f}",
        ]
        rows.append(row)

    column_widths = [
        max(len(str(item)) for item in col) + 3 for col in zip(*([column_titles] + rows))
    ]
    total_width = sum(column_widths)
    line = "-" * total_width

    print(line)
    print("".join(f"{title:{width}}" for title, width in zip(column_titles, column_widths)))
    print(line)
    for row in rows:
        print("".join(f"{value:{width}}" for value, width in zip(row, column_widths)))
    print(line)
