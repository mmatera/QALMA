"""Figures for the chiral three-body benchmark paper.

Reads ``chiral_three_body_results.json`` (or the three JSONL files produced
by the parallel run) and writes one PDF per figure into ``figures/``.

Usage
-----
    python plot_chiral_benchmark.py [results.json]

If no path is given the script looks for
``benchmark_results/chiral_three_body_results.json`` relative to the current
working directory.  JSONL files (``field_sweep.jsonl``, etc.) in the same
directory are also accepted — the script auto-detects the format.

Figures produced
----------------
fig1_calibration.pdf
    Scatter: T-score vs relative free-energy gap (small systems, exact known).
    Bubble size encodes L.

fig2_tscore_by_model.pdf
    Bar chart: mean T-score by model and inverse temperature β.

fig3_varconv.pdf
    Two-panel line chart: variance ratio vs numfields.
    Left: L=12, β=2.  Right: L=8, β=5.

fig4_magnetization.pdf
    Four-panel line chart (one per model): ⟨Sz⟩/N vs B.
    Solid lines = MF, dashed = exact (where available, L=4 β=2).
    Multiple L shown; color encodes L.

fig5_gap_vs_B.pdf
    Relative free-energy gap (%) vs B for small-L systems.
    One line per model, L=4, β=2.

fig6_kappa_z.pdf
    Scalar chirality κ_z vs B for Heisenberg models.
    L=6 and L=8, β=2.  Highlights the frozen-minimum artifact.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from qalma.utils import convex_fit_derivative

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# Model palette — consistent across all figures
MODELS = [
    ("pure chiral", 0.0, 1.0, "#1D9E75", "o", "pure chiral ($J=0,\\chi=1$)"),
    ("Heis only", 1.0, 0.0, "#888780", "s", "Heisenberg only ($\\chi=0$)"),
    ("Heis+chi=0.5", 1.0, 0.5, "#7F77DD", "^", "Heis.+weak chiral ($\\chi=0.5J$)"),
    ("Heis+chi=1", 1.0, 1.0, "#D85A30", "D", "Heis.+strong chiral ($\\chi=J$)"),
]

L_COLORS = {2: "#B5D4F4", 3: "#378ADD", 4: "#185FA5", 6: "#0C447C", 8: "#042C53"}
BETAS = [1.0, 2.0, 5.0]
BETA_COLORS = {1.0: "#B5D4F4", 2.0: "#378ADD", 5.0: "#042C53"}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_jsonl(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_data(results_path: Path) -> dict:
    """Return dict with keys exact_validation / numfields_convergence / field_sweep."""
    if results_path.suffix == ".json":
        return load_json(results_path)
    # JSONL directory
    base = results_path.parent
    return {
        "exact_validation": load_jsonl(base / "exact_validation.jsonl"),
        "numfields_convergence": load_jsonl(base / "numfields_convergence.jsonl"),
        "field_sweep": load_jsonl(base / "field_sweep.jsonl"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def select(records, **kwargs):
    """Filter records matching all keyword conditions."""
    out = records
    for k, v in kwargs.items():
        out = [r for r in out if r[k] == v]
    return out


def sorted_by(records, key):
    return sorted(records, key=lambda r: r[key])


def exact_mag(rows):
    """Numerical derivative dF_exact/dB ≈ ⟨Sz_tot⟩_exact."""
    rows = sorted_by(rows, "B")
    data = [(row["B"], row["f_exact"]) for row in rows]
    mags = convex_fit_derivative(data, convexity=-1)
    return rows, mags


def var_f_from_nf(nf_records, J, chi, L, beta):
    rows = sorted_by(select(nf_records, J=J, chi=chi, L=L, beta=beta), "numfields")
    nfs = [r["numfields"] for r in rows]
    vrat = [r.get("var_f_ratio", float("nan")) for r in rows]
    return nfs, vrat


def calibration_points(ev, fs):
    """Build (gap_rel%, T-score, L, model_label) for all small systems."""
    points = []
    for r in ev:
        F_ex = r["F_exact"]
        F_mf = r["F_variational"]
        # var_f: from field_sweep B=0
        fs_row = next(
            (
                x
                for x in fs
                if x["J"] == r["J"]
                and x["chi"] == r["chi"]
                and x["L"] == r["L"]
                and x["beta"] == r["beta"]
                and x["B"] == 0.0
            ),
            None,
        )
        if fs_row is None or fs_row["var_f"] <= 0:
            continue
        gap_rel = (F_mf - F_ex) / abs(F_ex) * 100
        t_score = (F_mf - F_ex) / np.sqrt(fs_row["var_f"])
        label = next(m[0] for m in MODELS if m[1] == r["J"] and m[2] == r["chi"])
        points.append(
            dict(
                label=label, L=r["L"], beta=r["beta"], gap_rel=gap_rel, t_score=t_score
            )
        )
    return points


# ---------------------------------------------------------------------------
# Figure 1 — calibration scatter
# ---------------------------------------------------------------------------


def fig1_calibration(ev, fs, out_dir):
    points = calibration_points(ev, fs)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    for name, J, chi, color, marker, full_label in MODELS:
        pts = [p for p in points if p["label"] == name]
        if not pts:
            continue
        xs = [p["t_score"] for p in pts]
        ys = [p["gap_rel"] for p in pts]
        sizes = [20 + 12 * (p["L"] - 2) for p in pts]
        ax.scatter(
            xs,
            ys,
            s=sizes,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
            label=full_label,
        )

    # Reference line: perfect calibration y ∝ x (schematic)
    x_ref = np.linspace(0, 1.6, 100)
    ax.plot(
        x_ref,
        40 * x_ref,
        color="gray",
        lw=0.8,
        ls="--",
        alpha=0.5,
        zorder=1,
        label="linear guide",
    )

    ax.set_xlabel("T-score")
    ax.set_ylabel(
        "relative gap $(F_{\\mathrm{mf}}-F_{\\mathrm{ex}})/|F_{\\mathrm{ex}}|$ (%)"
    )
    ax.set_xlim(0, 1.65)
    ax.set_ylim(0, 72)
    ax.grid(True)

    # Bubble-size legend for L
    for L in [2, 3, 4]:
        ax.scatter(
            [],
            [],
            s=20 + 12 * (L - 2),
            color="gray",
            label=f"$L={L}$",
            edgecolors="white",
            linewidths=0.4,
        )

    ax.legend(
        fontsize=7, framealpha=0.9, loc="upper left", handletextpad=0.3, borderpad=0.4
    )
    ax.set_title("T-score vs relative free-energy gap\n(variational MF, small systems)")

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_calibration.pdf")
    plt.close(fig)
    print("  fig1_calibration.pdf")


# ---------------------------------------------------------------------------
# Figure 2 — T-score by model and beta
# ---------------------------------------------------------------------------


def fig2_tscore_by_model(ev, fs, out_dir):
    points = calibration_points(ev, fs)
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    x = np.arange(len(MODELS))
    width = 0.22
    for bi, beta in enumerate(BETAS):
        means = []
        for name, *_ in MODELS:
            pts = [p for p in points if p["label"] == name and p["beta"] == beta]
            means.append(np.mean([p["t_score"] for p in pts]) if pts else np.nan)
        # bars =
        ax.bar(
            x + (bi - 1) * width,
            means,
            width,
            color=BETA_COLORS[beta],
            label=f"$\\beta={beta}$",
            edgecolor="white",
            linewidth=0.4,
        )

    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.7, label="T-score = 1")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m[4].replace("Heis.", "Heis.") for m in MODELS],
        fontsize=7,
        rotation=12,
        ha="right",
    )
    ax.set_ylabel("mean T-score ($L=2,3,4$)")
    ax.set_ylim(0, 1.85)
    ax.grid(True, axis="y")
    ax.legend(fontsize=7, framealpha=0.9)
    ax.set_title("T-score by model and temperature")

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_tscore_by_model.pdf")
    plt.close(fig)
    print("  fig2_tscore_by_model.pdf")


# ---------------------------------------------------------------------------
# Figure 3 — variance ratio vs numfields
# ---------------------------------------------------------------------------


def fig3_varconv(nf, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=False)

    panels = [
        (axes[0], 12, 2.0, "$L=12$, $\\beta=2$"),
        (axes[1], 8, 5.0, "$L=8$,  $\\beta=5$"),
    ]

    for ax, L, beta, title in panels:
        for name, J, chi, color, marker, full_label in MODELS:
            nfs, vrat = var_f_from_nf(nf, J, chi, L, beta)
            if not nfs:
                continue
            ax.plot(
                nfs,
                vrat,
                color=color,
                marker=marker,
                markersize=4,
                label=full_label,
                linestyle="--" if name == "pure chiral" else "-",
            )
        ax.axhline(1.0, color="gray", lw=0.7, ls=":", alpha=0.6)
        ax.set_xlabel("numfields")
        ax.set_ylabel(
            "variance ratio $\\mathrm{Var}[\\hat{F}](n_f)/\\mathrm{Var}[\\hat{F}](0)$"
        )
        ax.set_title(title)
        ax.grid(True)
        ax.set_xticks([0, 1, 2, 3, 4, 6, 8])

    axes[0].legend(fontsize=7, framealpha=0.9, loc="lower left")
    fig.suptitle("Variance ratio vs number of variational fields", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_varconv.pdf")
    plt.close(fig)
    print("  fig3_varconv.pdf")


# ---------------------------------------------------------------------------
# Figure 4 — magnetization curves
# ---------------------------------------------------------------------------


def fig4_magnetization(fs, out_dir):
    beta = 2.0
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0), sharex=True)
    axes = axes.flatten()

    for ax, (name, J, chi, color, marker, full_label) in zip(axes, MODELS):
        # MF curves: multiple L
        for L in [2, 3, 4, 6, 8]:
            rows = sorted_by(select(fs, J=J, chi=chi, L=L, beta=beta), "B")
            if not rows:
                continue
            N = 2 * L
            Bs = [r["B"] for r in rows]
            mags = [r["total_mag"] / N for r in rows]
            lw = 1.4 if L in (4, 8) else 0.8
            ax.plot(Bs, mags, color=L_COLORS[L], lw=lw, label=f"MF $L={L}$")

        # Exact (numerical derivative dF/dB), L=4
        rows_ex = sorted_by(
            [
                r
                for r in select(fs, J=J, chi=chi, L=4, beta=beta)
                if r["f_exact"] is not None
            ],
            "B",
        )
        if rows_ex:
            rows_ex, mags_ex = exact_mag(rows_ex)
            Bs_ex = [rows_ex[i]["B"] for i in range(1, len(rows_ex) - 1)]
            ms_ex = [mags_ex[i] / (2 * 4) for i in range(1, len(rows_ex) - 1)]
            ax.plot(Bs_ex, ms_ex, color="black", lw=1.4, ls="--", label="exact $L=4$")

        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.axhline(-0.5, color="gray", lw=0.5, ls=":", alpha=0.5)
        ax.set_title(full_label, fontsize=8)
        ax.set_ylabel("$\\langle S^z \\rangle / N$")
        ax.set_xlim(0, 4)
        ax.set_ylim(-0.55, 0.1)
        ax.grid(True)

    for ax in axes[2:]:
        ax.set_xlabel("$B$")

    # Shared legend
    handles = [
        mpl.lines.Line2D([], [], color=L_COLORS[L], lw=1.2, label=f"MF $L={L}$")
        for L in [2, 3, 4, 6, 8]
    ]
    handles.append(
        mpl.lines.Line2D([], [], color="black", lw=1.2, ls="--", label="exact $L=4$")
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        fontsize=7,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.suptitle(
        "Magnetization $\\langle S^z\\rangle/N$ vs field $B$   ($\\beta=2$)",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_magnetization.pdf")
    plt.close(fig)
    print("  fig4_magnetization.pdf")


# ---------------------------------------------------------------------------
# Figure 5 — relative gap vs B
# ---------------------------------------------------------------------------


def fig5_gap_vs_B(fs, out_dir):
    beta = 2.0
    L = 4
    fig, ax = plt.subplots(figsize=(3.8, 3.0))

    for name, J, chi, color, marker, full_label in MODELS:
        rows = sorted_by(
            [
                r
                for r in select(fs, J=J, chi=chi, L=L, beta=beta)
                if r["f_exact"] is not None
            ],
            "B",
        )
        if not rows:
            continue
        Bs = [r["B"] for r in rows]
        gaps = [(r["f"] - r["f_exact"]) / abs(r["f_exact"]) * 100 for r in rows]
        ax.plot(Bs, gaps, color=color, marker=marker, markersize=3, label=full_label)

    ax.set_xlabel("$B$")
    ax.set_ylabel("$(F_{\\mathrm{mf}}-F_{\\mathrm{ex}})/|F_{\\mathrm{ex}}|$ (%)")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 75)
    ax.grid(True)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.set_title(f"Relative free-energy gap vs $B$\n($L={L}$, $\\beta={beta}$)")

    fig.tight_layout()
    fig.savefig(out_dir / "fig5_gap_vs_B.pdf")
    plt.close(fig)
    print("  fig5_gap_vs_B.pdf")


# ---------------------------------------------------------------------------
# Figure 6 — kappa_z vs B
# ---------------------------------------------------------------------------


def fig6_kappa_z(fs, out_dir):
    beta = 2.0
    chiral_models = [m for m in MODELS if m[2] > 0 and m[1] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=False)

    for ax, (name, J, chi, color, marker, full_label) in zip(axes, chiral_models):
        for L, ls in [(6, "-"), (8, "--")]:
            rows = sorted_by(select(fs, J=J, chi=chi, L=L, beta=beta), "B")
            if not rows:
                continue
            Bs = [r["B"] for r in rows]
            kzs = [r["kappa_z"] for r in rows]
            ax.plot(Bs, kzs, color=color, ls=ls, lw=1.2, label=f"$L={L}$")

        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.set_xlabel("$B$")
        ax.set_ylabel("$\\kappa_z = \\langle \\chi_{ijk} \\rangle$")
        ax.set_xlim(0, 3)
        ax.set_title(full_label, fontsize=8)
        ax.legend(fontsize=7, framealpha=0.9)
        ax.grid(True)

    # Annotate the frozen-minimum region on the Heis+chi=1 panel
    ax = axes[1]
    ax.axvspan(0, 0.4, alpha=0.08, color="red", label="frozen min. ($L=8$)")
    ax.legend(fontsize=7, framealpha=0.9)

    fig.suptitle(
        "Scalar spin chirality $\\kappa_z$ vs field $B$   ($\\beta=2$)",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_kappa_z.pdf")
    plt.close(fig)
    print("  fig6_kappa_z.pdf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "results",
        nargs="?",
        default="benchmark_results/chiral_three_body_results.json",
        help="Path to results JSON (or JSONL directory). Default: %(default)s",
    )
    parser.add_argument(
        "--out",
        default="figures",
        help="Output directory for PDF figures. Default: %(default)s",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        sys.exit(f"Results file not found: {results_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {results_path} ...")
    data = load_data(results_path)
    ev = data["exact_validation"]
    nf = data["numfields_convergence"]
    fs = data["field_sweep"]
    print(f"  exact_validation:      {len(ev)} records")
    print(f"  numfields_convergence: {len(nf)} records")
    print(f"  field_sweep:           {len(fs)} records")
    print(f"Writing figures to {out_dir}/")

    fig1_calibration(ev, fs, out_dir)
    fig2_tscore_by_model(ev, fs, out_dir)
    fig3_varconv(nf, out_dir)
    fig4_magnetization(fs, out_dir)
    fig5_gap_vs_B(fs, out_dir)
    fig6_kappa_z(fs, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
