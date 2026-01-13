import pandas as pd
import matplotlib as mpl


# --- IMPORTANT: TrueType (Type 42) fonts for ACM/IEEE PDF (avoid Type 3) ---
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42
mpl.rcParams["pdf.use14corefonts"] = False
mpl.rcParams["mathtext.fontset"] = "stix"

def plot_execution_times(df, out_pdf=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    # --------- IEEE-ish styling ----------
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",          # Windows
            "Times",                    # macOS
            "Nimbus Roman No9 L",       # common Linux
            "STIX Two Text",
            "STIXGeneral",
            "DejaVu Serif",
        ],
    })

    # --------- aggregate per dim and init_method ----------
    df_grouped = (
        df.groupby(["dim", "init_method"], as_index=False)[["init_time", "opt_time"]]
          .mean()
    )
    dims = sorted(df_grouped["dim"].unique())
    methods = sorted(df_grouped["init_method"].unique())

    x = np.arange(len(dims))
    n_methods = len(methods)
    total_width = 0.80
    width = total_width / max(1, n_methods)

    # --------- color-blind safe palette (Okabe–Ito) ----------
    okabe_ito = [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
        "#000000",  # black
    ]

    def lighten(color, amount=0.35):
        r, g, b = mcolors.to_rgb(color)
        return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)

    eps = 1e-9  # avoid zeros for log scale

    fig, ax = plt.subplots(figsize=(6.6, 3.6))  # ~single-column IEEE friendly

    for i, method in enumerate(methods):
        sub = (
            df_grouped[df_grouped["init_method"] == method]
            .set_index("dim")
            .reindex(dims, fill_value=0)
        )

        init = np.where(sub["init_time"].to_numpy() <= 0, eps, sub["init_time"].to_numpy())
        opt  = np.where(sub["opt_time"].to_numpy()  <= 0, eps, sub["opt_time"].to_numpy())

        offsets = x + (i - (n_methods - 1) / 2) * width

        base = okabe_ito[i % len(okabe_ito)]
        init_col = lighten(base, 0.45)  # lighter for init
        opt_col  = base                 # solid for opt

        ax.bar(offsets, init, width=width, color=init_col, edgecolor="black", linewidth=0.6)
        ax.bar(offsets, opt,  width=width, bottom=init, color=opt_col,
               edgecolor="black", linewidth=0.6, hatch="///")

        # Optional: label TOTAL time above each stacked bar (log-scale friendly)
        totals = init + opt
        for xo, t in zip(offsets, totals):
            if t > eps:
                ax.text(xo, t * 1.15, f"{t:.2g}", ha="center", va="bottom",
                        fontsize=7, rotation=90)

    ax.set_yscale("log")
    ax.set_xlabel("Dimensions")
    ax.set_ylabel("Time (s, log scale)")
    ax.set_title("Execution Time by Dimension and Initialization Method")

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims])

    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # --------- Legends (methods + phases) ----------
    method_handles = [
        Patch(facecolor=okabe_ito[i % len(okabe_ito)], edgecolor="black", label=str(m))
        for i, m in enumerate(methods)
    ]
    # "Missing labels": phase legend matches what is drawn (light + hatched)
    phase_handles = [
        Patch(facecolor=lighten("#0072B2", 0.45), edgecolor="black", label="Initialization"),
        Patch(facecolor="#0072B2", edgecolor="black", hatch="///", label="Optimization"),
    ]

    leg1 = ax.legend(handles=method_handles, title="init_method",
                     bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=phase_handles, title="Phase", loc="upper right", frameon=True)

    fig.tight_layout()

    if out_pdf:
        # Saving triggers embedded TrueType fonts (because pdf.fonttype=42)
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_success_rates(df, alpha=10e-8, out_pdf=None):
    """
    Success = best_delta <= alpha
    Plot success rate (%) grouped by (dim, init_method).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    # --- IEEE/ACM-friendly: TrueType fonts in PDF (avoid Type 3) ---
    import matplotlib as mpl
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"]  = 42
    mpl.rcParams["pdf.use14corefonts"] = False
    mpl.rcParams["mathtext.fontset"] = "stix"

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": [
            "Times New Roman", "Times", "Nimbus Roman No9 L",
            "STIX Two Text", "STIXGeneral", "DejaVu Serif",
        ],
    })

    # --------- compute success ----------
    d = df.copy()
    d["best_delta"] = pd.to_numeric(d["best_delta"], errors="coerce")
    d["success"] = d["best_delta"].le(alpha)

    # --------- aggregate success rate per dim and init_method ----------
    g = (d.groupby(["dim", "init_method"], as_index=False)["success"]
           .mean()
           .rename(columns={"success": "success_rate"}))
    g["success_rate"] *= 100.0

    dims = sorted(g["dim"].unique())
    methods = g["init_method"].unique()

    x = np.arange(len(dims))
    n_methods = len(methods)
    total_width = 0.80
    width = total_width / max(1, n_methods)

    # --------- Okabe–Ito palette ----------
    okabe_ito = [
        "#0072B2", "#D55E00", "#009E73", "#CC79A7",
        "#E69F00", "#56B4E9", "#F0E442", "#000000",
    ]

    def lighten(color, amount=0.35):
        r, g_, b = mcolors.to_rgb(color)
        return (r + (1 - r) * amount, g_ + (1 - g_) * amount, b + (1 - b) * amount)

    fig, ax = plt.subplots(figsize=(6.6, 3.2))

    for i, method in enumerate(methods):
        sub = g[g["init_method"] == method].set_index("dim").reindex(dims)
        rates = sub["success_rate"].to_numpy()
        rates = np.nan_to_num(rates, nan=0.0)

        offsets = x + (i - (n_methods - 1) / 2) * width
        base = okabe_ito[i % len(okabe_ito)]
        col = lighten(base, 0.20)

        ax.bar(offsets, rates, width=width, color=col, edgecolor="black", linewidth=0.6)

        # add labels on top (optional but usually useful)
        for xo, r in zip(offsets, rates):
            ax.text(xo, r + 1.0, f"{r:.0f}%", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xlabel("Dimensions")
    ax.set_ylabel("Success rate (%)")
    ax.set_title(f"Success Rate by Dimension and Initialization Method (α = {alpha:.1e})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d_) for d_ in dims])

    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    method_handles = [
        Patch(facecolor=lighten(okabe_ito[i % len(okabe_ito)], 0.20),
              edgecolor="black", label=str(m))
        for i, m in enumerate(methods)
    ]
    ax.legend(handles=method_handles, title="init_method",
              bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    fig.tight_layout()

    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_success_iterations(df, alpha=10e-8, out_pdf=None, agg="median"):
    """
    Among SUCCESS cases (best_delta <= alpha), compare number of performed iterations (best_n_iter)
    grouped by (dim, init_method).

    agg: "median" (recommended) or "mean"
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    import matplotlib as mpl

    # --- TrueType fonts in PDF (avoid Type 3) ---
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"]  = 42
    mpl.rcParams["pdf.use14corefonts"] = False
    mpl.rcParams["mathtext.fontset"] = "stix"

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": [
            "Times New Roman", "Times", "Nimbus Roman No9 L",
            "STIX Two Text", "STIXGeneral", "DejaVu Serif",
        ],
    })

    # ---------- filter to success cases ----------
    d = df.copy()
    d["best_delta"] = pd.to_numeric(d["best_delta"], errors="coerce")
    d["best_n_iter"] = pd.to_numeric(d["best_n_iter"], errors="coerce")

    succ = d[d["best_delta"].le(alpha)].copy()

    # If a (dim, init_method) has no successes, it won't appear; we'll fill with 0 for plotting.
    if agg not in ("median", "mean"):
        raise ValueError("agg must be 'median' or 'mean'")

    if agg == "median":
        g = (succ.groupby(["dim", "init_method"], as_index=False)["best_n_iter"]
                 .median()
                 .rename(columns={"best_n_iter": "n_iter"}))
        agg_label = "Median"
    else:
        g = (succ.groupby(["dim", "init_method"], as_index=False)["best_n_iter"]
                 .mean()
                 .rename(columns={"best_n_iter": "n_iter"}))
        agg_label = "Mean"

    # Ensure all dims/methods exist for grouped view (fill missing with 0)
    dims = sorted(d["dim"].unique())
    methods = sorted(d["init_method"].unique())

    # build full grid so missing success combos appear as 0
    grid = pd.MultiIndex.from_product([dims, methods], names=["dim", "init_method"]).to_frame(index=False)
    g = grid.merge(g, on=["dim", "init_method"], how="left")
    g["n_iter"] = g["n_iter"].fillna(0.0)

    x = np.arange(len(dims))
    n_methods = len(methods)
    total_width = 0.80
    width = total_width / max(1, n_methods)

    okabe_ito = [
        "#0072B2", "#D55E00", "#009E73", "#CC79A7",
        "#E69F00", "#56B4E9", "#F0E442", "#000000",
    ]

    def lighten(color, amount=0.20):
        r, g_, b = mcolors.to_rgb(color)
        return (r + (1 - r) * amount, g_ + (1 - g_) * amount, b + (1 - b) * amount)

    fig, ax = plt.subplots(figsize=(6.6, 3.2))

    for i, method in enumerate(methods):
        sub = g[g["init_method"] == method].set_index("dim").reindex(dims)
        iters = sub["n_iter"].to_numpy()
        iters = np.nan_to_num(iters, nan=0.0)

        offsets = x + (i - (n_methods - 1) / 2) * width
        base = okabe_ito[i % len(okabe_ito)]
        col = lighten(base, 0.20)

        ax.bar(offsets, iters, width=width, color=col, edgecolor="black", linewidth=0.6)

    ax.set_xlabel("Dimensions")
    ax.set_ylabel("Performed iterations (best_n_iter)")
    ax.set_title(f"{agg_label} iterations among SUCCESS runs (α = {alpha:.1e})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d_) for d_ in dims])

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    method_handles = [
        Patch(facecolor=lighten(okabe_ito[i % len(okabe_ito)], 0.20),
              edgecolor="black", label=str(m))
        for i, m in enumerate(methods)
    ]
    ax.legend(handles=method_handles, title="init_method",
              bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    fig.tight_layout()

    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

pop_size = 64
df = pd.read_csv(f'bbob_initpop_comparison_{pop_size}.csv')

# sorting the df by 'dim' and 'init_method' for better readability
# init method order: random, obl, lhs, sobol, oblesa_lhs, oblesa_sobol, oblesa_ess
init_method_order = [
    'random', 'obl', 'lhs', 'sobol', 'oblesa_lhs', 'oblesa_sobol', 'oblesa_ess'
]
df['init_method'] = pd.Categorical(df['init_method'], categories=init_method_order, ordered=True)
df = df.sort_values(by=['dim', 'init_method']).reset_index(drop=True)

# And save an ACM/IEEE-compliant PDF:
# plot_execution_times(df, out_pdf=f"execution_times_pop{pop_size}.pdf")
plot_execution_times(df)

plot_success_rates(df, alpha=10e-8)

plot_success_iterations(df, alpha=10e-8)