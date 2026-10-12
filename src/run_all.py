"""Reproduce everything:  python src/run_all.py   (~8 min on a 2-core laptop)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import emotion_model, analysis, figures
from ads import load_ads

t0 = time.time()
print("[1/3] training the emotion detector on GoEmotions")
emotion_model.train(load_ads().text.tolist())
print("[2/3] scoring the IRA Facebook ads and running the analyses")
analysis.run()
print("[3/3] figures")
for f in (figures.fig1_targeting, figures.fig2_engagement, figures.fig3_paid_vs_earned,
          figures.fig4_timeline, figures.fig5_model, figures.fig6_twitter):
    f()
print(f"done in {(time.time() - t0) / 60:.1f} min")
