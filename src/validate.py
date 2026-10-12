"""
Human validation of the detector on the target domain (Facebook ads).

The detector learned from Reddit comments. To measure how well it transfers,
label `results/human_validation_sample.csv` yourself:

  1. Open it in Excel / Google Sheets / LibreOffice.
  2. For each ad, put 1 under every emotion the TEXT EXPRESSES (several are fine),
     or 1 under `neutral` if it expresses none. Leave other cells empty.
     Label expressed emotion, not the emotion you think the ad tries to provoke.
  3. Save as CSV (same name) and run:   python src/validate.py

Optionally add a second annotator's file as results/human_validation_sample_2.csv
to also get inter-annotator agreement (Cohen's kappa per label).
"""
import json, os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
from common import EKMAN, MODELS, RES


def load(path):
    d = pd.read_csv(path)
    labelled = d[EKMAN].notna().any(axis=1)
    d = d[labelled].copy()
    d[EKMAN] = d[EKMAN].fillna(0).astype(int)
    return d


def main():
    path = os.path.join(RES, "human_validation_sample.csv")
    h = load(path)
    if len(h) == 0:
        print(__doc__); return
    model = joblib.load(os.path.join(MODELS, "emotion_detector.joblib"))
    P = model.predict(h.text.tolist())
    Y = h[EKMAN].to_numpy()
    res = dict(n_labelled=int(len(h)),
               macro_f1=float(f1_score(Y, P, average="macro", zero_division=0)),
               per_label_f1={e: float(f1_score(Y[:, j], P[:, j], zero_division=0)) for j, e in enumerate(EKMAN)},
               human_label_rate={e: float(Y[:, j].mean()) for j, e in enumerate(EKMAN)},
               model_label_rate={e: float(P[:, j].mean()) for j, e in enumerate(EKMAN)})
    second = os.path.join(RES, "human_validation_sample_2.csv")
    if os.path.exists(second):
        h2 = load(second).set_index("id").loc[h.id]
        res["cohen_kappa"] = {e: float(cohen_kappa_score(h[e], h2[e])) for e in EKMAN}
    json.dump(res, open(os.path.join(RES, "human_validation.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
