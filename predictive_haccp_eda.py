from pathlib import Path
from html import escape

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Dataset"
OUTPUT_DIR = PROJECT_DIR / "eda_outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

COLORS = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
CLASS_COLORS = {
    "Good": "#2e7d32",
    "Bad": "#c62828",
    "Unoccupied": "#4c78a8",
    "Occupied": "#f58518",
    0: "#4c78a8",
    1: "#f58518",
}


def pct(value):
    return f"{value:.1%}"


def color_for(label, fallback_index=0):
    return CLASS_COLORS.get(label, COLORS[fallback_index % len(COLORS)])


def df_to_markdown(df):
    cols = [""] + [str(c) for c in df.columns]
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("|" + "|".join(["---"] + ["---:" for _ in df.columns]) + "|")
    for idx, row in df.iterrows():
        values = [str(idx)]
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_svg(name, body, width, height):
    path = FIGURE_DIR / name
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f'<rect width="100%" height="100%" fill="white"/>\n{body}\n</svg>\n'
    )
    path.write_text(svg, encoding="utf-8")
    return path


def svg_text(x, y, value, size=12, anchor="start", weight="400", fill="#222"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{escape(str(value))}</text>'
    )


def bar_chart(name, title, values, width=820, height=420, semantic_colors=False, label_mode="count"):
    margin = {"top": 46, "right": 28, "bottom": 92, "left": 70}
    inner_w = width - margin["left"] - margin["right"]
    inner_h = height - margin["top"] - margin["bottom"]
    items = list(values.items())
    max_v = max(values.values()) if values else 1
    total = sum(values.values()) if values else 1
    gap = 18
    bar_w = max(18, (inner_w - gap * (len(items) - 1)) / max(1, len(items)))

    parts = [svg_text(width / 2, 28, title, 16, "middle", "500")]
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + inner_h}" x2="{width - margin["right"]}" y2="{margin["top"] + inner_h}" stroke="#999"/>')
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + inner_h}" stroke="#999"/>')

    for i, (label, value) in enumerate(items):
        x = margin["left"] + i * (bar_w + gap)
        h = (value / max_v) * inner_h
        y = margin["top"] + inner_h - h
        fill = color_for(label, i) if semantic_colors else COLORS[i % len(COLORS)]
        label_text = pct(value / total) if label_mode == "percent" else f"{int(value):,}"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{fill}"/>')
        parts.append(svg_text(x + bar_w / 2, y - 6, label_text, 11, "middle"))
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{margin["top"] + inner_h + 20}" '
            f'font-family="Arial, sans-serif" font-size="11" text-anchor="middle" fill="#222" '
            f'transform="rotate(28 {x + bar_w / 2:.1f} {margin["top"] + inner_h + 20})">{escape(str(label))}</text>'
        )

    return write_svg(name, "\n".join(parts), width, height)


def stacked_bar_chart(name, title, table, width=860, height=430, semantic_colors=False, label_mode="count"):
    margin = {"top": 46, "right": 120, "bottom": 96, "left": 70}
    inner_w = width - margin["left"] - margin["right"]
    inner_h = height - margin["top"] - margin["bottom"]
    totals = table.sum(axis=1)
    max_v = totals.max() if len(totals) else 1
    gap = 18
    bar_w = max(18, (inner_w - gap * (len(table) - 1)) / max(1, len(table)))

    parts = [svg_text(width / 2, 28, title, 16, "middle", "500")]
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + inner_h}" x2="{width - margin["right"]}" y2="{margin["top"] + inner_h}" stroke="#999"/>')

    for i, (idx, row) in enumerate(table.iterrows()):
        x = margin["left"] + i * (bar_w + gap)
        y_base = margin["top"] + inner_h
        cumulative = 0
        for j, col in enumerate(table.columns):
            value = row[col]
            h = (value / max_v) * inner_h
            y = y_base - cumulative - h
            fill = color_for(col, j) if semantic_colors else COLORS[j % len(COLORS)]
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{fill}"/>')
            if label_mode == "percent" and totals.loc[idx] > 0:
                segment_pct = value / totals.loc[idx]
                if h >= 18:
                    label_fill = "white" if segment_pct >= 0.18 else "#222"
                    parts.append(svg_text(x + bar_w / 2, y + h / 2 + 4, pct(segment_pct), 10, "middle", "500", label_fill))
            cumulative += h
        parts.append(svg_text(x + bar_w / 2, margin["top"] + inner_h + 18, idx, 10, "middle"))
        if label_mode == "count":
            parts.append(svg_text(x + bar_w / 2, margin["top"] + inner_h - cumulative - 6, f"{int(totals.loc[idx]):,}", 10, "middle"))

    legend_x = width - margin["right"] + 20
    for j, col in enumerate(table.columns):
        y = margin["top"] + j * 24
        fill = color_for(col, j) if semantic_colors else COLORS[j % len(COLORS)]
        parts.append(f'<rect x="{legend_x}" y="{y}" width="14" height="14" fill="{fill}"/>')
        parts.append(svg_text(legend_x + 22, y + 12, col, 11))

    return write_svg(name, "\n".join(parts), width, height)


def boxplot_chart(name, title, df, group_col, features, width=900, height=650, semantic_colors=False):
    parts = [svg_text(width / 2, 28, title, 16, "middle", "500")]
    groups = list(df[group_col].dropna().unique())
    panel_w = width / 2
    panel_h = (height - 50) / 2

    for k, feature in enumerate(features):
        px = (k % 2) * panel_w
        py = 46 + (k // 2) * panel_h
        vals = df[feature].dropna()
        ymin, ymax = float(vals.min()), float(vals.max())
        span = ymax - ymin or 1
        left = px + 65
        top = py + 30
        chart_w = panel_w - 95
        chart_h = panel_h - 80

        parts.append(svg_text(px + panel_w / 2, py + 16, feature, 13, "middle", "500"))
        parts.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#aaa"/>')
        parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#aaa"/>')
        parts.append(svg_text(left - 8, top + 4, f"{ymax:.1f}", 9, "end", fill="#555"))
        parts.append(svg_text(left - 8, top + chart_h, f"{ymin:.1f}", 9, "end", fill="#555"))

        for i, group in enumerate(groups):
            s = df.loc[df[group_col] == group, feature].dropna()
            q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
            low, high = s.min(), s.max()
            cx = left + (i + 0.5) * chart_w / len(groups)
            box_w = min(54, chart_w / len(groups) * 0.5)

            def sy(v):
                return top + chart_h - ((float(v) - ymin) / span) * chart_h

            yq1, ymed, yq3, ylow, yhigh = sy(q1), sy(med), sy(q3), sy(low), sy(high)
            parts.append(f'<line x1="{cx:.1f}" y1="{yhigh:.1f}" x2="{cx:.1f}" y2="{ylow:.1f}" stroke="#333"/>')
            parts.append(f'<line x1="{cx - box_w / 2:.1f}" y1="{yhigh:.1f}" x2="{cx + box_w / 2:.1f}" y2="{yhigh:.1f}" stroke="#333"/>')
            parts.append(f'<line x1="{cx - box_w / 2:.1f}" y1="{ylow:.1f}" x2="{cx + box_w / 2:.1f}" y2="{ylow:.1f}" stroke="#333"/>')
            fill = color_for(group, i) if semantic_colors else COLORS[i % len(COLORS)]
            parts.append(f'<rect x="{cx - box_w / 2:.1f}" y="{yq3:.1f}" width="{box_w:.1f}" height="{max(1, yq1 - yq3):.1f}" fill="{fill}" opacity="0.75" stroke="#333"/>')
            parts.append(f'<line x1="{cx - box_w / 2:.1f}" y1="{ymed:.1f}" x2="{cx + box_w / 2:.1f}" y2="{ymed:.1f}" stroke="#111" stroke-width="2"/>')
            parts.append(svg_text(cx, top + chart_h + 18, group, 10, "middle"))

    return write_svg(name, "\n".join(parts), width, height)


def heatmap_chart(name, title, corr, width=680, height=650):
    labels = list(corr.columns)
    n = len(labels)
    margin = {"top": 66, "right": 30, "bottom": 120, "left": 130}
    cell = min((width - margin["left"] - margin["right"]) / n, (height - margin["top"] - margin["bottom"]) / n)
    parts = [svg_text(width / 2, 28, title, 16, "middle", "500")]

    for i, row in enumerate(labels):
        parts.append(svg_text(margin["left"] - 8, margin["top"] + i * cell + cell * 0.62, row, 10, "end"))
        for j, col in enumerate(labels):
            v = float(corr.loc[row, col])
            if v >= 0:
                intensity = int(245 - abs(v) * 130)
                fill = f"rgb({intensity},{min(255, intensity + 20)},{245})"
            else:
                intensity = int(245 - abs(v) * 130)
                fill = f"rgb(245,{intensity},{intensity})"
            x = margin["left"] + j * cell
            y = margin["top"] + i * cell
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell:.1f}" height="{cell:.1f}" fill="{fill}" stroke="#ddd"/>')
            parts.append(svg_text(x + cell / 2, y + cell * 0.58, f"{v:.2f}", 9, "middle"))

    for j, col in enumerate(labels):
        x = margin["left"] + j * cell + cell / 2
        y = margin["top"] + n * cell + 12
        parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="10" '
            f'text-anchor="end" fill="#222" transform="rotate(-38 {x:.1f} {y:.1f})">{escape(col)}</text>'
        )

    return write_svg(name, "\n".join(parts), width, height)


def line_chart(name, title, df, features, width=980, height=760):
    parts = [svg_text(width / 2, 28, title, 16, "middle", "500")]
    margin = {"top": 55, "right": 30, "bottom": 30, "left": 70}
    panel_h = (height - margin["top"] - margin["bottom"]) / len(features)
    colors_by_file = {
        "datatraining.txt": COLORS[0],
        "datatest.txt": COLORS[1],
        "datatest2.txt": COLORS[2],
    }

    for k, feature in enumerate(features):
        py = margin["top"] + k * panel_h
        chart_h = panel_h - 34
        chart_w = width - margin["left"] - margin["right"]
        vals = df[feature]
        ymin, ymax = float(vals.min()), float(vals.max())
        span = ymax - ymin or 1

        parts.append(svg_text(width / 2, py + 12, feature, 12, "middle", "500"))
        parts.append(f'<line x1="{margin["left"]}" y1="{py + chart_h + 22}" x2="{margin["left"] + chart_w}" y2="{py + chart_h + 22}" stroke="#aaa"/>')
        parts.append(svg_text(margin["left"] - 8, py + 29, f"{ymax:.1f}", 9, "end", fill="#555"))
        parts.append(svg_text(margin["left"] - 8, py + chart_h + 22, f"{ymin:.1f}", 9, "end", fill="#555"))

        for fname, group in df.groupby("file"):
            g = group.sort_values("timestamp")
            sample_step = max(1, len(g) // 450)
            g = g.iloc[::sample_step]
            xmin = g["timestamp"].min().timestamp()
            xmax = g["timestamp"].max().timestamp()
            xspan = xmax - xmin or 1
            points = []
            for _, row in g.iterrows():
                x = margin["left"] + ((row["timestamp"].timestamp() - xmin) / xspan) * chart_w
                y = py + 22 + chart_h - ((float(row[feature]) - ymin) / span) * chart_h
                points.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors_by_file.get(fname, "#333")}" stroke-width="1.4" opacity="0.85"/>')

    legend_y = height - 8
    legend_x = margin["left"]
    for i, fname in enumerate(colors_by_file):
        x = legend_x + i * 190
        parts.append(f'<line x1="{x}" y1="{legend_y - 4}" x2="{x + 28}" y2="{legend_y - 4}" stroke="{colors_by_file[fname]}" stroke-width="3"/>')
        parts.append(svg_text(x + 36, legend_y, fname, 10))

    return write_svg(name, "\n".join(parts), width, height)


def fruit_eda():
    raw_path = DATA_DIR / "Dataset 1.csv"
    raw = pd.read_csv(raw_path)
    clean = raw.copy()
    clean["Class"] = clean["Class"].replace({"BAD": "Bad"})

    summary = {
        "shape": raw.shape,
        "columns": list(raw.columns),
        "missing": raw.isna().sum().to_dict(),
        "duplicates": int(raw.duplicated().sum()),
        "raw_class_counts": raw["Class"].value_counts().to_dict(),
        "clean_class_counts": clean["Class"].value_counts().to_dict(),
        "fruit_counts": raw["Fruit"].value_counts().to_dict(),
        "numeric_describe": raw.describe(include="number").round(3),
    }

    feature_cols = ["Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"]
    crosstab = pd.crosstab(clean["Fruit"], clean["Class"])
    plots = [
        bar_chart(
            "fruit_class_counts.svg",
            "Fruit Spoilage Class Balance",
            clean["Class"].value_counts().sort_index().to_dict(),
            semantic_colors=True,
            label_mode="percent",
        ),
        bar_chart("fruit_category_counts.svg", "Fruit Category Counts", raw["Fruit"].value_counts().to_dict()),
        stacked_bar_chart(
            "fruit_category_by_class.svg",
            "Fruit Category by Spoilage Class",
            crosstab,
            semantic_colors=True,
            label_mode="percent",
        ),
        boxplot_chart("fruit_sensor_boxplots_by_class.svg", "Fruit Sensor Ranges by Class", clean, "Class", feature_cols, semantic_colors=True),
        heatmap_chart("fruit_numeric_correlation.svg", "Fruit Dataset Numeric Correlation", clean[feature_cols].corr()),
    ]
    return summary, plots


def occupancy_eda():
    occ_dir = DATA_DIR / "occupancy+detection"
    frames = []
    per_file = {}

    for name in ["datatraining.txt", "datatest.txt", "datatest2.txt"]:
        df = pd.read_csv(occ_dir / name)
        df["timestamp"] = pd.to_datetime(df["date"])
        df["file"] = name
        frames.append(df)
        intervals = df["timestamp"].diff().dropna().dt.total_seconds()
        per_file[name] = {
            "shape": df.shape,
            "missing_total": int(df.isna().sum().sum()),
            "duplicates": int(df.drop(columns=["timestamp", "file"]).duplicated().sum()),
            "occupancy_counts": df["Occupancy"].value_counts().to_dict(),
            "start": df["timestamp"].min(),
            "end": df["timestamp"].max(),
            "median_interval_seconds": float(intervals.median()),
        }

    all_occ = pd.concat(frames, ignore_index=True)
    counts = pd.DataFrame({name: pd.Series(info["occupancy_counts"]) for name, info in per_file.items()}).fillna(0).T
    counts = counts.rename(columns={0: "Unoccupied", 1: "Occupied"})
    numeric = ["Temperature", "Humidity", "Light", "CO2"]
    corr_cols = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "Occupancy"]
    plots = [
        stacked_bar_chart(
            "uci_occupancy_class_balance.svg",
            "UCI Occupancy Class Balance by File",
            counts,
            semantic_colors=True,
            label_mode="percent",
        ),
        line_chart("uci_sensor_trends_by_file.svg", "UCI Sensor Trends by File", all_occ, numeric),
        boxplot_chart("uci_sensor_boxplots_by_occupancy.svg", "UCI Sensor Ranges by Occupancy", all_occ, "Occupancy", numeric, semantic_colors=True),
        heatmap_chart("uci_numeric_correlation.svg", "UCI Occupancy Numeric Correlation", all_occ[corr_cols].corr()),
    ]
    return per_file, all_occ, plots


def write_report(fruit_summary, fruit_plots, occ_summary, occ, occ_plots):
    report = OUTPUT_DIR / "predictive_haccp_eda_report.md"
    fruit_rows, fruit_cols = fruit_summary["shape"]
    clean_counts = fruit_summary["clean_class_counts"]
    bad_total = clean_counts.get("Bad", 0)
    good_total = clean_counts.get("Good", 0)
    total = bad_total + good_total

    occ_total = len(occ)
    occ_occupied = int((occ["Occupancy"] == 1).sum())
    occ_unoccupied = int((occ["Occupancy"] == 0).sum())

    occ_file_lines = []
    for name, info in occ_summary.items():
        counts = info["occupancy_counts"]
        occupied = counts.get(1, 0)
        unoccupied = counts.get(0, 0)
        occ_file_lines.append(
            f"| `{name}` | {info['shape'][0]:,} | {info['start']} | {info['end']} | "
            f"{info['median_interval_seconds']:.0f}s | {unoccupied:,} | {occupied:,} | "
            f"{info['missing_total']} | {info['duplicates']} |"
        )

    lines = [
        "# Predictive HACCP Dataset EDA Report",
        "",
        "## Data Sources Reviewed",
        "",
        f"- Fruit spoilage dataset: `{DATA_DIR / 'Dataset 1.csv'}`",
        f"- UCI occupancy dataset folder: `{DATA_DIR / 'occupancy+detection'}`",
        "",
        "## 1. Fruit Spoilage Dataset",
        "",
        f"- Shape: `{fruit_rows:,}` rows and `{fruit_cols}` columns.",
        f"- Columns: `{', '.join(fruit_summary['columns'])}`.",
        "- Timestamp column: not present.",
        f"- Missing values: `{sum(fruit_summary['missing'].values())}` total.",
        f"- Exact duplicate rows: `{fruit_summary['duplicates']:,}`.",
        f"- Raw class labels: `{fruit_summary['raw_class_counts']}`.",
        f"- Cleaned class labels after merging `BAD` into `Bad`: `{fruit_summary['clean_class_counts']}`.",
        f"- Cleaned class balance: `Good = {good_total:,}` ({pct(good_total / total)}), `Bad = {bad_total:,}` ({pct(bad_total / total)}).",
        f"- Fruit counts: `{fruit_summary['fruit_counts']}`.",
        "",
        "### Numeric Summary",
        "",
        df_to_markdown(fruit_summary["numeric_describe"]),
        "",
        "### Fruit Dataset Plots",
        "",
    ]

    for path in fruit_plots:
        rel = path.relative_to(OUTPUT_DIR)
        title = rel.stem.replace("_", " ").title()
        lines.append(f"![{title}]({rel})")
        lines.append("")

    lines.extend([
        "### Immediate Interpretation",
        "",
        "- This dataset is suitable for supervised spoilage classification, especially Random Forest.",
        "- It is also usable for anomaly detection experiments with Isolation Forest, provided the unsupervised nature is clearly explained.",
        "- It is not suitable for LSTM or detection-delay analysis because there is no timestamp.",
        "- The duplicate count and `BAD` label inconsistency must be handled before modelling.",
        "",
        "## 2. UCI Occupancy Dataset",
        "",
        f"- Combined rows across three files: `{occ_total:,}`.",
        f"- Combined occupancy balance: `Unoccupied = {occ_unoccupied:,}` ({pct(occ_unoccupied / occ_total)}), `Occupied = {occ_occupied:,}` ({pct(occ_occupied / occ_total)}).",
        "- This is a timestamped environmental sensor dataset with approximately one-minute sampling.",
        "- It is useful for temporal modelling, lag features, rolling windows, and LSTM.",
        "- It is not a food-safety dataset, so its results should be described as temporal IoT benchmark results only.",
        "",
        "### File-Level Summary",
        "",
        "| File | Rows | Start | End | Median interval | Unoccupied | Occupied | Missing | Duplicates |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
        *occ_file_lines,
        "",
        "### UCI Dataset Plots",
        "",
    ])

    for path in occ_plots:
        rel = path.relative_to(OUTPUT_DIR)
        title = rel.stem.replace("_", " ").title()
        lines.append(f"![{title}]({rel})")
        lines.append("")

    lines.extend([
        "## Recommended Next Coding Step",
        "",
        "Start with the fruit-spoilage cleaning pipeline:",
        "",
        "1. Standardise column names.",
        "2. Merge `BAD` into `Bad`.",
        "3. Decide how to handle duplicates.",
        "4. Create a stratified train/validation/test split.",
        "5. Train baseline and Random Forest models.",
        "",
        "Then handle UCI occupancy as a separate temporal benchmark for lag features and LSTM.",
        "",
    ])

    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    fruit_summary, fruit_plots = fruit_eda()
    occ_summary, occ, occ_plots = occupancy_eda()
    report = write_report(fruit_summary, fruit_plots, occ_summary, occ, occ_plots)

    print("EDA complete.")
    print(f"Report: {report}")
    print("Figures:")
    for path in fruit_plots + occ_plots:
        print(f"- {path}")


if __name__ == "__main__":
    main()
