"""Generate composite dashboard image for README."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

fig = plt.figure(figsize=(20, 12), dpi=200, facecolor="#1a1a2e")
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

colors = {
    "bg": "#1a1a2e",
    "card": "#16213e",
    "text": "#e0e0e0",
    "accent1": "#0f3460",
    "accent2": "#e94560",
    "green": "#2ecc71",
    "blue": "#3498db",
    "orange": "#e67e22",
    "purple": "#9b59b6",
}

fig.suptitle("TrainLens: AI Training Data Quality Platform", fontsize=22, color="white", fontweight="bold", y=0.98)

ax1 = fig.add_subplot(gs[0, 0])
categories = ["Billing", "Tech Sup", "Shipping", "Product", "Cancel", "Refund"]
counts = [148, 152, 138, 130, 118, 114]
bars = ax1.barh(categories, counts, color=[colors["blue"], colors["green"], colors["orange"],
                                            colors["purple"], colors["accent2"], colors["accent1"]])
ax1.set_facecolor(colors["card"])
ax1.set_title("Category Distribution", color=colors["text"], fontsize=12, fontweight="bold")
ax1.tick_params(colors=colors["text"], labelsize=9)
ax1.set_xlim(0, 170)

ax2 = fig.add_subplot(gs[0, 1])
labels = ["Positive", "Neutral", "Negative"]
sizes = [35, 40, 25]
explode = (0.05, 0.05, 0.05)
wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels,
                                     colors=[colors["green"], colors["blue"], colors["accent2"]],
                                     autopct="%1.0f%%", startangle=90, textprops={"color": colors["text"], "fontsize": 9})
ax2.set_title("Sentiment Split", color=colors["text"], fontsize=12, fontweight="bold")

ax3 = fig.add_subplot(gs[0, 2])
metrics = ["Accuracy", "Precision", "Recall", "F1"]
values = [80.8, 79.5, 80.8, 80.8]
bars = ax3.bar(metrics, values, color=[colors["green"], colors["blue"], colors["orange"], colors["purple"]], width=0.6)
ax3.set_facecolor(colors["card"])
ax3.set_title("Model Performance", color=colors["text"], fontsize=12, fontweight="bold")
ax3.tick_params(colors=colors["text"], labelsize=9)
ax3.set_ylim(0, 100)
for bar, val in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val}%",
             ha="center", va="bottom", color=colors["text"], fontsize=9)

ax4 = fig.add_subplot(gs[0, 3])
ax4.text(0.5, 0.7, "98.8%", fontsize=36, ha="center", va="center", color=colors["green"], fontweight="bold")
ax4.text(0.5, 0.4, "Data Quality", fontsize=14, ha="center", va="center", color=colors["text"])
ax4.text(0.5, 0.2, "Score", fontsize=12, ha="center", va="center", color=colors["text"])
ax4.set_facecolor(colors["card"])
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis("off")
ax4.set_title("Overall Score", color=colors["text"], fontsize=12, fontweight="bold")

ax5 = fig.add_subplot(gs[1, 0:2])
weeks = np.arange(1, 27)
tickets = np.random.poisson(30, 26) + np.random.randint(20, 40, 26)
ax5.fill_between(weeks, tickets, alpha=0.3, color=colors["blue"])
ax5.plot(weeks, tickets, color=colors["blue"], linewidth=2)
ax5.set_facecolor(colors["card"])
ax5.set_title("Weekly Ticket Volume", color=colors["text"], fontsize=12, fontweight="bold")
ax5.tick_params(colors=colors["text"], labelsize=9)
ax5.set_xlabel("Week", color=colors["text"], fontsize=9)
ax5.set_ylabel("Tickets", color=colors["text"], fontsize=9)

ax6 = fig.add_subplot(gs[1, 2:4])
channels = ["Email", "Chat", "Phone", "Social"]
resolution = [87.5, 82.3, 91.2, 78.6]
csat = [3.8, 3.5, 4.1, 3.2]
x = np.arange(len(channels))
width = 0.35
bars1 = ax6.bar(x - width/2, resolution, width, label="Resolution %", color=colors["green"], alpha=0.8)
ax6_twin = ax6.twinx()
bars2 = ax6_twin.bar(x + width/2, csat, width, label="Avg CSAT", color=colors["orange"], alpha=0.8)
ax6.set_facecolor(colors["card"])
ax6.set_title("Channel Performance", color=colors["text"], fontsize=12, fontweight="bold")
ax6.set_xticks(x)
ax6.set_xticklabels(channels)
ax6.tick_params(colors=colors["text"], labelsize=9)
ax6_twin.tick_params(colors=colors["text"], labelsize=9)
ax6.set_ylabel("Resolution Rate (%)", color=colors["text"], fontsize=9)
ax6_twin.set_ylabel("Avg CSAT", color=colors["text"], fontsize=9)

ax7 = fig.add_subplot(gs[2, 0:2])
conf_bins = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
cat_acc = [45, 62, 78, 89, 95]
sent_acc = [30, 42, 55, 68, 75]
ax7.plot(conf_bins, cat_acc, "o-", color=colors["blue"], linewidth=2, markersize=8, label="Category")
ax7.plot(conf_bins, sent_acc, "s-", color=colors["accent2"], linewidth=2, markersize=8, label="Sentiment")
ax7.set_facecolor(colors["card"])
ax7.set_title("Confidence Calibration", color=colors["text"], fontsize=12, fontweight="bold")
ax7.tick_params(colors=colors["text"], labelsize=9)
ax7.set_ylabel("Accuracy (%)", color=colors["text"], fontsize=9)
ax7.legend(facecolor=colors["card"], edgecolor=colors["text"], labelcolor=colors["text"], fontsize=9)

ax8 = fig.add_subplot(gs[2, 2:4])
labels_cm = ["Billing", "Tech", "Ship", "Prod", "Cancel", "Refund"]
cm = np.array([[120, 8, 5, 3, 7, 5],
               [10, 125, 4, 6, 2, 5],
               [6, 3, 115, 8, 3, 3],
               [5, 7, 6, 100, 4, 8],
               [8, 2, 2, 3, 95, 8],
               [4, 4, 3, 5, 6, 92]])
im = ax8.imshow(cm, cmap="Blues", aspect="auto")
ax8.set_xticks(np.arange(len(labels_cm)))
ax8.set_yticks(np.arange(len(labels_cm)))
ax8.set_xticklabels(labels_cm, fontsize=8, color=colors["text"])
ax8.set_yticklabels(labels_cm, fontsize=8, color=colors["text"])
ax8.set_title("Confusion Matrix", color=colors["text"], fontsize=12, fontweight="bold")
ax8.set_xlabel("Predicted", color=colors["text"], fontsize=9)
ax8.set_ylabel("Actual", color=colors["text"], fontsize=9)
for i in range(len(labels_cm)):
    for j in range(len(labels_cm)):
        text = ax8.text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > 60 else "black", fontsize=8)

output_path = ROOT / "assets" / "dashboard_composite.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=colors["bg"])
plt.close()
print(f"Saved to {output_path}")
