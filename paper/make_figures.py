"""Figure for the paper. Numbers are copied from README.md (closed-loop tables, 'Against reviewing every turn')."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

arms = ["None", "Every\nturn", "Gated", "Gated,\n1 call"]
files_left = [2.48, 0.00, 0.00, 0.15]       # induced drift, unrequested files left per session
checks = [84, 100, 99, 100]                 # induced drift, task check pass %
calls_drift = [0, 4.81, 1.39, 0.77]         # advisor calls per induced-drift session
calls_clean = [0, 2.81, 0.06, 0.06]         # advisor calls per clean session

plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, (a, b) = plt.subplots(1, 2, figsize=(3.15, 1.75))
x = range(len(arms))
colors = ["#9e9e9e", "#4a4a4a", "#FF4B0A", "#FF9E7A"]

a.bar(x, files_left, color=colors)
for i, (v, c) in enumerate(zip(files_left, checks)):
    a.text(i, v + 0.05, f"{v:.2f}\n({c}%)", ha="center", va="bottom", fontsize=6)
a.set_ylim(0, 3.3)
a.set_title("Unrequested files left\n(task check pass rate)", fontsize=7)

w = 0.38
b.bar([i - w / 2 for i in x], calls_drift, w, color=colors, label="induced drift")
b.bar([i + w / 2 for i in x], calls_clean, w, color=colors, alpha=0.45, hatch="///", label="clean")
for i, (d, c) in enumerate(zip(calls_drift, calls_clean)):
    # drift labels end at the bar's right edge and clean labels start at the bar's left edge, so the pair never collides
    if d: b.text(i, d + 0.08, f"{d:.2f}", ha="right", fontsize=5.5)
    if c: b.text(i + 0.06, c + 0.08, f"{c:.2f}", ha="left", fontsize=5.5)
b.set_ylim(0, 5.6)
b.set_xlim(-0.6, 3.75)
b.set_title("Advisor calls per session\n(solid: drift, hatched: clean)", fontsize=7)

for ax in (a, b):
    ax.set_xticks(list(x))
    ax.set_xticklabels(arms, fontsize=5.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=6)
fig.tight_layout(pad=0.3)
fig.savefig("fig_loop.pdf")
print("wrote fig_loop.pdf")
