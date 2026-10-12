"""
Emotion detector trained on GoEmotions (58k Reddit comments, human-labelled),
grouped into Ekman's six basic emotions plus neutral (multi-label).

Two models, ensembled:
  A  TF-IDF (word 1-2 grams + character 2-5 grams) -> one-vs-rest logistic regression
  B  SIF-weighted GloVe-Twitter vectors -> multi-label MLP
Per-label decision thresholds and the ensemble weight are tuned on the dev split;
the test split is touched once. The reference is fine-tuned BERT from the GoEmotions
paper (Demszky et al., 2020, Table 6): Ekman macro-F1 0.64.
"""
import json, os
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from common import EKMAN, MODELS, RES, SIF, clean, fetch

BERT_EKMAN = {"anger": 0.57, "disgust": 0.53, "fear": 0.68, "joy": 0.82, "neutral": 0.66,
              "sadness": 0.59, "surprise": 0.61, "macro": 0.64}


def load_split(name):
    names = open(fetch("emotions.txt")).read().split()
    ekman = json.load(open(fetch("ekman_mapping.json")))
    to_ek = {fine: ek for ek, fines in ekman.items() for fine in fines}
    to_ek["neutral"] = "neutral"
    df = pd.read_csv(fetch(f"goemotions_{name}.tsv"), sep="\t", header=None,
                     names=["text", "labels", "id"], quoting=3)
    Y = np.zeros((len(df), len(EKMAN)), int)
    for i, lab in enumerate(df.labels.astype(str)):
        for k in lab.split(","):
            Y[i, EKMAN.index(to_ek[names[int(k)]])] = 1
    return df.text.tolist(), Y


class EmotionDetector:
    def __init__(self, extra_vocab_texts=()):
        self.extra = list(extra_vocab_texts)

    # --- model pieces
    def _tfidf_fit(self, X):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=150_000,
                                    preprocessor=clean, token_pattern=r"<\w+>|#?\w+|[!?]+")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                                    max_features=200_000, preprocessor=clean)
        M = hstack([self.word.fit_transform(X), self.char.fit_transform(X)]).tocsr(); M.sort_indices(); return M

    def _tfidf(self, X):
        return hstack([self.word.transform(X), self.char.transform(X)]).tocsr()

    def fit(self, Xtr, Ytr, Xdev, Ydev):
        self.lr = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=3000, solver="liblinear"))
        self.lr.fit(self._tfidf_fit(Xtr), Ytr)
        self.sif = SIF(list(Xtr) + self.extra)
        self.mlp = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-4, early_stopping=True,
                                 max_iter=200, random_state=0)
        self.mlp.fit(self.sif.transform(Xtr), Ytr)
        # tune ensemble weight + per-label thresholds on dev
        pa, pb = self._parts(Xdev)
        best = (-1, None, None)
        for w in np.linspace(0, 1, 11):
            p = w * pa + (1 - w) * pb
            th = np.array([self._best_th(Ydev[:, j], p[:, j]) for j in range(len(EKMAN))])
            f = f1_score(Ydev, (p >= th).astype(int), average="macro", zero_division=0)
            if f > best[0]:
                best = (f, w, th)
        self.dev_macro_f1, self.w, self.th = best
        return self

    @staticmethod
    def _best_th(y, p):
        grid = np.linspace(0.05, 0.9, 35)
        return grid[np.argmax([f1_score(y, p >= t, zero_division=0) for t in grid])]

    def _parts(self, X):
        return self.lr.predict_proba(self._tfidf(X)), self.mlp.predict_proba(self.sif.transform(X))

    def predict_proba(self, X):
        pa, pb = self._parts(list(X))
        return self.w * pa + (1 - self.w) * pb

    def predict(self, X):
        return (self.predict_proba(X) >= self.th).astype(int)

    def intensity(self, X):
        """Probability rescaled by each label's threshold: >= 1 means 'detected'."""
        return self.predict_proba(X) / self.th


def evaluate(model, X, Y):
    P = model.predict(X)
    per = {e: float(f1_score(Y[:, j], P[:, j], zero_division=0)) for j, e in enumerate(EKMAN)}
    return dict(macro_f1=float(f1_score(Y, P, average="macro", zero_division=0)),
                macro_precision=float(precision_score(Y, P, average="macro", zero_division=0)),
                macro_recall=float(recall_score(Y, P, average="macro", zero_division=0)),
                per_label_f1=per)


def train(extra_vocab_texts=()):
    Xtr, Ytr = load_split("train"); Xdv, Ydv = load_split("dev"); Xte, Yte = load_split("test")
    m = EmotionDetector(extra_vocab_texts).fit(Xtr, Ytr, Xdv, Ydv)
    res = dict(n_train=len(Xtr), n_dev=len(Xdv), n_test=len(Xte), ensemble_weight_tfidf=float(m.w),
               thresholds=dict(zip(EKMAN, m.th.round(3).tolist())), dev_macro_f1=float(m.dev_macro_f1),
               glove_coverage=float(m.sif.coverage), test=evaluate(m, Xte, Yte), bert_reference=BERT_EKMAN)
    # ablations: each component alone (thresholds re-tuned on dev, same protocol)
    for name, w in (("tfidf_only", 1.0), ("glove_mlp_only", 0.0)):
        w0, th0 = m.w, m.th
        m.w = w
        p = m.predict_proba(Xdv); m.th = np.array([m._best_th(Ydv[:, j], p[:, j]) for j in range(len(EKMAN))])
        res[name] = evaluate(m, Xte, Yte)["macro_f1"]
        m.w, m.th = w0, th0
    joblib.dump(m, os.path.join(MODELS, "emotion_detector.joblib"), compress=3)
    json.dump(res, open(os.path.join(RES, "emotion_model.json"), "w"), indent=2)
    return m, res


if __name__ == "__main__":
    # import by module name so the saved model unpickles from any entry point
    import emotion_model
    from ads import load_ads
    m, r = emotion_model.train(load_ads().text.tolist())
    print(json.dumps({k: v for k, v in r.items() if k != "bert_reference"}, indent=1))
