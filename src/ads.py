"""
IRA Facebook ads released by the US House Permanent Select Committee on
Intelligence (HPSCI), parsed from PDFs by Ed Summers (umd-mith/irads).

Each ad is assigned the audience it *targeted*. The assignment uses only the
ad's targeting fields and its Facebook page name, never its text, so the
emotion analysis can't leak into the audience labels. Rules apply in priority
order and are transparent keyword patterns, tolerant of PDF-parsing noise.
"""
import json, re
import numpy as np
import pandas as pd
from common import fetch

AUDIENCES = [
    ("Black American", r"africa|afro|black|malcolm|martin luther|blacktivist|woke|cop ?block|police (brutality|misconduct)"
                       r"|panther|angela davis|kalvin|halvin|ka\|vin|dont-shoot|don't shoot"),
    ("Latino", r"chicano|la raza|hispan|latin|mexic|lowrider|brown ?united|brownunitedfront"),
    ("LGBT", r"lgbt|\bgay\b|lesbian|homosex|transgender|queer"),
    ("Muslim American", r"muslim|islam|quran|mosque|ramadan"),
    ("Conservative / patriot", r"patriot|conservat|republican|tea party|trump|2nd|second amendment|gun owner|\bnra\b"
                               r"|national rifle|confedera|texas|secede|southern|veteran|military|army|border|refugee"
                               r"|immigra|invaders|stop-ai|stop-al|stop a\.|christian|jesus|bible|god\b|police officer|blue lives"),
]


def _flat(x):
    if isinstance(x, dict):
        return [v for k in x for v in _flat(x[k])]
    if isinstance(x, list):
        return [v for i in x for v in _flat(i)]
    return [str(x)]


def _targeting_text(t):
    if not t:
        return ""
    vals = []
    for k, v in t.items():
        if k in ("age", "language", "placements") or k.startswith("location") or "exclude" in k:
            continue
        vals += [s for s in _flat(v) if not s.lower().startswith("exclude")]
    return " | ".join(vals).lower()


def audience_of(page, targeting):
    s = f"{page} | {targeting}"
    for name, pat in AUDIENCES:
        if re.search(pat, s):
            return name
    return "Other / general"


def load_ads():
    raw = json.load(open(fetch("irads.json")))
    rows = []
    for a in raw:
        page = re.search(r"facebook\.com/([^/?]+)", a.get("url") or "")
        page = page.group(1).lower() if page else ""
        tgt = _targeting_text(a.get("targeting"))
        spend = a.get("spend") or {}
        rows.append(dict(id=a["id"], text=(a.get("text") or "").strip(), page=page, targeting=tgt,
                         audience=audience_of(page, tgt), impressions=a.get("impressions") or 0,
                         clicks=a.get("clicks") or 0, created=a.get("created"),
                         spend_rub=float(spend["amount"]) if spend.get("currency") == "RUB" and spend.get("amount") else np.nan))
    df = pd.DataFrame(rows)
    df["created"] = pd.to_datetime(df.created, utc=True, errors="coerce")
    df["has_text"] = df.text.str.len() >= 10
    df["ran"] = df.impressions >= 100                  # ads that actually reached people
    df["ctr"] = np.where(df.impressions > 0, df.clicks / df.impressions.where(df.impressions > 0), np.nan)
    df["valid_ctr"] = df.ran & (df.clicks <= df.impressions) & df.has_text
    return df


if __name__ == "__main__":
    df = load_ads()
    print(len(df), "ads;", int(df.valid_ctr.sum()), "with text and >=100 impressions")
    print(df.audience.value_counts())
    print(df[df.valid_ctr].groupby("audience").agg(ads=("id", "size"), impressions=("impressions", "sum"),
                                                     clicks=("clicks", "sum"), rub=("spend_rub", "sum"),
                                                     median_ctr=("ctr", "median")))
    print(df[df.audience == "Other / general"].page.value_counts().head(15))
