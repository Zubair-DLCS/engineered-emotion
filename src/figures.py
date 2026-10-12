"""Draws every figure from saved results."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RES, FIG, EMOTIONS6, EKMAN

S1, S2 = "#2a78d6", "#eb6834"
SURF, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2, "font.size": 9.5, "legend.frameon": False, "figure.dpi": 150,
})
J = lambda f: json.load(open(os.path.join(RES, f)))
AUD = ["Black American", "Conservative / patriot", "Latino", "LGBT", "Muslim American", "Other / general"]


def title(fig, t, sub=None):
    h = fig.get_size_inches()[1]
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - (0.62 if sub else 0.4) / h))
    fig.text(0.008, 1 - 0.08 / h, t, ha="left", va="top", fontweight="bold", fontsize=13)
    if sub:
        fig.text(0.008, 1 - 0.36 / h, sub, ha="left", va="top", color=INK2, fontsize=9.5)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight"); plt.close(fig)


def heat(ax, M, rows, cols, vmax, fmt="{:.0f}"):
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.tick_params(length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=9,
                    color=SURF if M[i, j] > 0.55 * vmax else INK)
    return im


def fig1_targeting():
    t = J("analysis.json")["emotional_targeting"]
    rows = AUD + ["ALL"]
    cols = EMOTIONS6 + ["neutral_only"]
    M = np.array([[100 * (t[a][c]["share"] if c != "neutral_only" else t[a][c]) for c in cols] for a in rows])
    fig, ax = plt.subplots(figsize=(9, 4.1), constrained_layout=True)
    im = heat(ax, M, [f"{a}  ({t[a]['ads']})" for a in rows], EMOTIONS6 + ["no emotion"], vmax=65)
    ax.axhline(len(rows) - 1.5, color=SURF, lw=3); ax.axvline(5.5, color=SURF, lw=3)
    cb = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.9); cb.set_label("% of ads"); cb.outline.set_visible(False)
    title(fig, "Emotional targeting: what each audience was shown",
          "% of IRA Facebook ads expressing each emotion (multi-label), by the audience the ad targeted")
    save(fig, "fig1_emotional_targeting.png")


def fig2_engagement():
    e = J("analysis.json")["engagement"]
    pn = e["pooled_negative"]
    rows = [("All audiences", pn["flags"]["negative_odds_change_pct"], pn["flags"]["negative_ci95_pct"],
             pn["flags"]["negative_p"], pn["n_negative_ads"], e["n_ads"])]
    for aud, v in pn["within_audience"].items():
        rows.append((aud, v["odds_change_pct"], v["ci95_pct"], v["p"], v["n_negative"], v["n"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True,
                                 gridspec_kw=dict(width_ratios=[1.2, 1]))
    y = np.arange(len(rows))[::-1]
    for yi, (name, m, ci, p, nn, n) in zip(y, rows):
        a1.plot(ci, [yi, yi], color=S1, lw=2.5); a1.plot(m, yi, "o", color=S1, ms=7)
        a1.text(ci[1] + 3, yi, f"{m:+.0f}%  (p={p:.3f})", va="center", fontsize=8.5)
    a1.axvline(0, color=INK2, lw=1)
    a1.set_yticks(y); a1.set_yticklabels([f"{r[0]}\n{r[4]} negative / {r[5]} ads" for r in rows], fontsize=8.5)
    a1.grid(axis="y", visible=False); a1.set_xlim(-30, 260)
    a1.set_xlabel("change in odds of a click when the ad expresses a negative emotion (%)")
    a1.set_title("Negative emotion earns clicks")
    d = e["detected_flags"]
    ys = np.arange(len(EMOTIONS6))[::-1]
    for yi, em in zip(ys, EMOTIONS6):
        v = d[em]; small = v["n_with"] < 30
        c = NEUTRAL if small else S1
        a2.plot(v["ci95_pct"], [yi, yi], color=c, lw=2.5); a2.plot(v["odds_change_pct"], yi, "o", color=c, ms=7)
    a2.axvline(0, color=INK2, lw=1); a2.grid(axis="y", visible=False)
    a2.set_yticks(ys); a2.set_yticklabels([f"{em}  (n={d[em]['n_with']})" for em in EMOTIONS6])
    a2.set_xlabel("change in odds of a click (%), each emotion vs absent")
    a2.set_title("By emotion (grey: fewer than 30 ads)")
    title(fig, "Natural experiment in the ad market: which emotions earned clicks?",
          f"{e['n_ads']} ads with ≥100 impressions; controls: audience, quarter, log impressions; "
          f"95% CIs clustered by Facebook page ({e['n_page_clusters']} pages)")
    save(fig, "fig2_engagement.png")


def fig3_paid_vs_earned():
    p = J("analysis.json")["paid_vs_earned"]["by_emotion"]
    order = [k for k in ["neutral", "joy", "surprise", "anger", "sadness", "fear"] if k in p]
    fig, ax = plt.subplots(figsize=(8.5, 3.6), constrained_layout=True)
    y = np.arange(len(order))[::-1]
    for yi, k in zip(y, order):
        s, c = 100 * p[k]["spend_share"], 100 * p[k]["click_share"]
        ax.plot([s, c], [yi, yi], color=NEUTRAL, lw=3, zorder=1)
        ax.scatter(s, yi, color=S2, s=50, zorder=2, label="share of roubles spent" if k == order[0] else None)
        ax.scatter(c, yi, color=S1, s=50, zorder=2, label="share of clicks earned" if k == order[0] else None)
        ax.text(max(s, c) + 1.5, yi, f"{p[k]['rub_per_click']:.2f} ₽ per click", va="center", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels([f"{k}  ({int(p[k]['ads'])} ads)" for k in order]); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 75); ax.set_xlabel("% of total (ads grouped by their dominant emotion)")
    ax.legend(loc="lower right", fontsize=8.5)
    title(fig, "Paid vs earned: neutral ads took 59% of the budget but 51% of the clicks")
    save(fig, "fig3_paid_vs_earned.png")


def fig4_timeline():
    m = pd.read_csv(os.path.join(RES, "monthly_emotion_mix.csv"), parse_dates=["month"]).set_index("month")
    sm = m[["joy", "anger"]].rolling(3, center=True, min_periods=2).mean() * 100
    fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)
    for col, c in (("joy", S1), ("anger", S2)):
        ax.plot(sm.index, sm[col], color=c, label=col)
        ax.text(sm.index[-1] + pd.Timedelta(days=12), sm[col].dropna().iloc[-1], col, color=INK, va="center", fontsize=9)
    ax.axvline(pd.Timestamp("2016-11-08"), color=INK2, lw=1, ls="--")
    ax.text(pd.Timestamp("2016-11-15"), ax.get_ylim()[1] * 0.95, "US election", fontsize=8.5, va="top")
    ax.set_ylabel("% of that month's ads (3-month mean)"); ax.set_ylim(0, None)
    ax.legend(loc="upper left", fontsize=8.5)
    title(fig, "After the election, the ads turned warm", "Monthly share of new ads expressing joy vs anger")
    save(fig, "fig4_timeline.png")


def fig5_model():
    r = J("emotion_model.json")
    bert = r["bert_reference"]
    labs = EKMAN
    ours = [r["test"]["per_label_f1"][l] for l in labs] + [r["test"]["macro_f1"]]
    ref = [bert[l] for l in labs] + [bert["macro"]]
    names = labs + ["macro avg"]
    fig, ax = plt.subplots(figsize=(8.5, 3.8), constrained_layout=True)
    y = np.arange(len(names))[::-1]
    for yi, o, b in zip(y, ours, ref):
        ax.plot([o, b], [yi, yi], color=NEUTRAL, lw=3, zorder=1)
    ax.scatter(ref, y, color=S2, s=50, zorder=2, label="fine-tuned BERT (Demszky et al., 2020)")
    ax.scatter(ours, y, color=S1, s=50, zorder=2, label="this project: TF-IDF + GloVe ensemble (CPU)")
    ax.set_yticks(y); ax.set_yticklabels(names); ax.grid(axis="y", visible=False); ax.set_xlim(0.3, 0.9)
    ax.set_xlabel("F1 on the GoEmotions test set (Ekman grouping)"); ax.legend(loc="lower right", fontsize=8.5)
    title(fig, f"The detector: {r['test']['macro_f1']:.2f} macro-F1 vs BERT's {bert['macro']:.2f}",
          f"Trained on {r['n_train']:,} human-labelled Reddit comments; thresholds tuned on dev, test touched once")
    save(fig, "fig5_detector_benchmark.png")


def fig6_twitter():
    cp = J("analysis.json")["cross_platform_tweets"]
    if not cp:
        return
    rows = ["RightTroll", "LeftTroll", "HashtagGamer", "Fearmonger", "NewsFeed"]
    M = np.array([[100 * cp[r][e] for e in EMOTIONS6] for r in rows])
    fig, ax = plt.subplots(figsize=(8.5, 3.4), constrained_layout=True)
    im = heat(ax, M, [f"{r}  ({cp[r]['tweets']:,})" for r in rows], EMOTIONS6, vmax=30, fmt="{:.1f}")
    cb = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.9); cb.set_label("% of tweets"); cb.outline.set_visible(False)
    title(fig, "Expressed is not induced: the Fearmonger unit barely expresses fear",
          "Same detector on the operation's original English tweets (sample per persona)")
    save(fig, "fig6_twitter_personas.png")


if __name__ == "__main__":
    for f in (fig1_targeting, fig2_engagement, fig3_paid_vs_earned, fig4_timeline, fig5_model, fig6_twitter):
        f(); print("drew", f.__name__)
