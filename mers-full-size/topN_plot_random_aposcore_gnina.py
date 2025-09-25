import csv
import matplotlib.pyplot as plt

""" use csv below to generate a single plot with top N plots from random_pick.py and topN_aposcore_analysis.py and topN_gnina_analysis"
"""

def load_summary_csv(path):
    Ns, means, errors = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            Ns.append(int(row["N"]))
            mean = row.get("percent_below_2A") or row.get("avg_percent_below_2A") or row.get("percent_below_2.0A")
            error = row.get("ci_95") or row.get("stddev") or "0"
            means.append(float(mean))
            errors.append(float(error))
    return Ns, means, errors

def plot_comparison(
    scored_csv="topN_rmsd_summary.csv",
    random_csv="random_pick_rmsd_summary.csv",
    gnina_csv="gnina_topN_summary.csv",
    out_png="compare_rmsd_plot.png"
):
    # Load ApoScore results
    N_scored, pct_scored, _ = load_summary_csv(scored_csv)

    # Load Random results
    N_random, pct_random, ci_random = load_summary_csv(random_csv)

    # Load GNINA results
    N_gnina, pct_gnina, _ = load_summary_csv(gnina_csv)

    # Plot all three
    plt.figure(figsize=(6, 4))
    plt.plot(N_scored, pct_scored, marker='o', label="ApoScore top-N")
    plt.errorbar(N_random, pct_random, yerr=ci_random, fmt='o--', capsize=4, label="Random top-N ± 95% CI")
    plt.plot(N_gnina, pct_gnina, marker='s', color='blue', label="GNINA top-N")

    # Formatting
    plt.xlabel("Top N Ligands per molX")
    plt.ylabel("% RMSD < 2 Å")
    plt.title("MERS-CoV (full size) Top-N Ligand RMSD Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"✅ Comparison plot saved to: {out_png}")
    plt.show()

if __name__ == "__main__":
    plot_comparison()
