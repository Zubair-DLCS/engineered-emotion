"""Paths, label sets and shared text utilities."""
import gzip, json, os, re, subprocess
from collections import Counter
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RAW = os.path.join(ROOT, "data"), os.path.join(ROOT, "data", "raw")
RES, FIG, MODELS = (os.path.join(ROOT, d) for d in ("results", "figures", "models"))
for d in (RAW, RES, FIG, MODELS):
    os.makedirs(d, exist_ok=True)

GH = "https://raw.githubusercontent.com"
SOURCES = {
    "irads.json": f"{GH}/umd-mith/irads/master/site/index.json",
    "goemotions_train.tsv": f"{GH}/google-research/google-research/master/goemotions/data/train.tsv",
    "goemotions_dev.tsv": f"{GH}/google-research/google-research/master/goemotions/data/dev.tsv",
    "goemotions_test.tsv": f"{GH}/google-research/google-research/master/goemotions/data/test.tsv",
    "emotions.txt": f"{GH}/google-research/google-research/master/goemotions/data/emotions.txt",
    "ekman_mapping.json": f"{GH}/google-research/google-research/master/goemotions/data/ekman_mapping.json",
}
GLOVE_URL = ("https://github.com/RaRe-Technologies/gensim-data/releases/download/"
             "glove-twitter-50/glove-twitter-50.gz")
GLOVE = os.path.join(DATA, "glove", "glove-twitter-50.gz")

EKMAN = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
EMOTIONS6 = EKMAN[:6]


def fetch(name):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        subprocess.run(["curl", "-sL", "-o", path, SOURCES[name]], check=True)
    return path


# ---------------------------------------------------------------- text

URL = re.compile(r"https?://\S+|www\.\S+")
USER = re.compile(r"@\w+")
NUM = re.compile(r"(?<![\w#])[-+]?\d[\d,.:]*")
HASH = re.compile(r"#(\w+)")
TOK = re.compile(r"<\w+>|#?\w+(?:'\w+)?|[!?]+|[^\w\s]")


def clean(text):
    t = URL.sub(" <url> ", str(text).replace("[NAME]", "<user>").lower())
    t = USER.sub(" <user> ", t)
    return NUM.sub(" <number> ", t)


def tokens(text):
    return TOK.findall(HASH.sub(r" <hashtag> \1 ", clean(text)))


class SIF:
    """Smooth-inverse-frequency GloVe-Twitter sentence vectors (Arora et al., 2017)."""

    def __init__(self, texts, a=1e-3):
        counts = Counter(t for x in texts for t in tokens(x))
        total = sum(counts.values())
        if not os.path.exists(GLOVE):
            os.makedirs(os.path.dirname(GLOVE), exist_ok=True)
            subprocess.run(["curl", "-sL", "-o", GLOVE, GLOVE_URL], check=True)
        self.vec = {}
        with gzip.open(GLOVE, "rt", encoding="utf8", errors="ignore") as f:
            next(f)
            for line in f:
                w, rest = line.split(" ", 1)
                if w in counts:
                    self.vec[w] = np.array(rest.split(), dtype=np.float32)
        self.w = {t: a / (a + c / total) for t, c in counts.items() if t in self.vec}
        self.coverage = sum(c for t, c in counts.items() if t in self.vec) / total

    def transform(self, texts):
        out = np.zeros((len(texts), 50), np.float32)
        for i, x in enumerate(texts):
            ws = [(self.w[t], self.vec[t]) for t in tokens(x) if t in self.w]
            if ws:
                wt = np.array([w for w, _ in ws])[:, None]
                out[i] = (wt * np.stack([v for _, v in ws])).sum(0) / wt.sum()
        return out
