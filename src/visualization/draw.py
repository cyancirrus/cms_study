import matplotlib.pyplot as plt
from visualization.summary import (
    column_filter_summary,
    column_summary,
    zeroized_score_by_group,
    histogram_summary,
)


def draw_summary(
    database: str,
    year: int,
    output_file: str,
    table: str,
    columns: list[str],
    bins: int = 20,
):
    """Draw histograms for multiple columns as subplots."""
    cols = 2
    rows = (
        len(columns) + cols - 1
    ) // cols  # ceil division
    _, axes = plt.subplots(
        rows, cols, figsize=(10, 4 * rows)
    )
    axes = axes.flatten()  # make iteration easier

    for ax, column in zip(axes, columns):
        df = column_summary(database, year, table, column)
        hist_data = histogram_summary(df, column, bins)
        if hist_data is None:
            continue
        ax.hist(
            hist_data["data"],
            bins=hist_data["bins"],
            color="skyblue",
            edgecolor="black",
        )
        ax.set_title(hist_data["title"])
        ax.set_xlabel(column)
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"{output_file}_{year}", dpi=150)


def draw_filter_summary(
    database: str,
    year: int,
    output_file: str,
    table: str,
    column: str,
    measure: str,
    conditions: list[str],
    bins: int = 20,
):
    """Draw histograms for multiple columns as subplots."""
    cols = 2
    rows = (
        len(conditions) + cols - 1
    ) // cols  # ceil division
    fig, axes = plt.subplots(
        rows, cols, figsize=(10, 4 * rows)
    )
    axes = axes.flatten()  # make iteration easier

    for ax, c in zip(axes, conditions):
        df = column_filter_summary(
            database, year, table, column, measure, c
        )
        hist_data = histogram_summary(df, measure, bins)
        if hist_data is None:
            continue
        ax.hist(
            hist_data["data"],
            bins=hist_data["bins"],
            color="skyblue",
            edgecolor="black",
        )
        ax.set_title(hist_data["title"])
        ax.set_xlabel(c)
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"{output_file}_{year}", dpi=150)
    plt.close(fig)


def draw_zeroized_state_summary(
    database: str,
    year: int,
    output_file: str,
    table: str,
    labels: list[str],
    better: list[str],
    worse: list[str],
    count: list[str],
    bins: int = 10,
):
    """Draw histograms for multiple columns as subplots."""
    cols = 2
    rows = (len(labels) + cols - 1) // cols  # ceil division
    fig, axes = plt.subplots(
        rows, cols, figsize=(10, 4 * rows)
    )
    axes = axes.flatten()  # make iteration easier

    for ax, l, b, w, t in zip(
        axes, labels, better, worse, count
    ):
        df = zeroized_score_by_group(
            database, table, l, b, w, t, "state"
        )
        hist_data = histogram_summary(
            df, f"{l}_zeroized_score", bins
        )
        if hist_data is None:
            continue
        ax.hist(
            hist_data["data"],
            bins=hist_data["bins"],
            color="skyblue",
            edgecolor="black",
        )
        ax.set_title(hist_data["title"])
        ax.set_xlabel("Zeroized Score by State")
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"{output_file}_{year}", dpi=150)


def draw_zeroized_facility_summary(
    database: str,
    year: int,
    output_file: str,
    table: str,
    labels: list[str],
    better: list[str],
    worse: list[str],
    count: list[str],
    bins: int = 10,
):
    """Draw histograms for multiple columns as subplots."""
    cols = 2
    rows = (len(labels) + cols - 1) // cols  # ceil division
    fig, axes = plt.subplots(
        rows, cols, figsize=(10, 4 * rows)
    )
    axes = axes.flatten()  # make iteration easier

    for ax, l, b, w, t in zip(
        axes, labels, better, worse, count
    ):
        df = zeroized_score_by_group(
            database, table, l, b, w, t, "facility_id"
        )
        hist_data = histogram_summary(
            df, f"{l}_zeroized_score", bins
        )
        if hist_data is None:
            continue
        ax.hist(
            hist_data["data"],
            bins=hist_data["bins"],
            color="skyblue",
            edgecolor="black",
        )
        ax.set_title(hist_data["title"])
        ax.set_xlabel("Zeroized Score by Facility")
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"{output_file}_{year}", dpi=150)


def draw_zeroized_type_summary(
    database: str,
    year: int,
    output_file: str,
    table: str,
    labels: list[str],
    better: list[str],
    worse: list[str],
    count: list[str],
    bins: int = 10,
):
    """Draw histograms for multiple columns as subplots."""
    cols = 2
    rows = (len(labels) + cols - 1) // cols  # ceil division
    fig, axes = plt.subplots(
        rows, cols, figsize=(10, 4 * rows)
    )
    axes = axes.flatten()  # make iteration easier

    for ax, l, b, w, t in zip(
        axes, labels, better, worse, count
    ):
        df = zeroized_score_by_group(
            database, table, l, b, w, t, "hospital_type"
        )
        hist_data = histogram_summary(
            df, f"{l}_zeroized_score", bins
        )
        if hist_data is None:
            continue
        ax.hist(
            hist_data["data"],
            bins=hist_data["bins"],
            color="skyblue",
            edgecolor="black",
        )
        ax.set_title(hist_data["title"])
        ax.set_xlabel("Zeroized Score by hospital_type")
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"{output_file}_{year}", dpi=150)
