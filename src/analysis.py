"""
What emotions did the IRA buy on Facebook, and which ones earned clicks?

1. Emotional targeting  share of ads expressing each emotion, by targeted audience
2. Engagement           click-through rate vs emotion: logistic-scale regression with
                        audience and quarter fixed effects, log-impressions control,
                        standard errors clustered by Facebook page
3. Paid vs earned       share of roubles spent vs share of clicks earned, by emotion
4. Timeline             monthly emotional mix around the 2016 election
5. Cross-platform       the same operation's tweets (Project "Two Clocks"), by persona
"""
import json, os, re
import numpy as np
import pandas as pd
import joblib
import statsmodels.formula.api as smf
from common import EKMAN, EMOTIONS6, MODELS, RES, ROOT
from ads import load_ads

TWEETS = os.path.join(os.path.dirname(ROOT), "two-clocks", "data", "tweets.parquet")


def page_family(p):
    p = re.sub(r"-?\d{6,}$", "", p.lower())
    return re.sub(r"[^a-z]", "", p.replace("|", "l"))


def score_ads(model):
    df = load_ads()
    df = df[df.has_text].copy()
    inten = model.intensity(df.text.tolist())
    for j, e in enumerate(EKMAN):
        df[f"i_{e}"] = inten[:, j]
        df[f"d_{e}"] = (inten[:, j] >= 1).astype(int)
    six = inten[:, :6]
    df["dominant"] = np.where(six.max(1) >= 1, np.array(EMOTIONS6)[six.argmax(1)], "neutral")
    df["page_family"] = df.page.map(page_family)
    df["quarter"] = df.created.dt.tz_convert(None).dt.to_period("Q").astype(str)
    df.drop(columns=["targeting"]).to_csv(os.path.join(RES, "ads_scored.csv"), index=False)
    return df


def bootstrap_share(x, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def emotional_targeting(df):
    out = {}
    for aud, g in df.groupby("audience"):
        out[aud] = dict(ads=int(len(g)), **{e: dict(share=float(g[f"d_{e}"].mean()), ci95=bootstrap_share(g[f"d_{e}"]))
                                            for e in EMOTIONS6},
                        neutral_only=float((g.dominant == "neutral").mean()))
    out["ALL"] = dict(ads=int(len(df)), **{e: dict(share=float(df[f"d_{e}"].mean()), ci95=bootstrap_share(df[f"d_{e}"]))
                                           for e in EMOTIONS6}, neutral_only=float((df.dominant == "neutral").mean()))
    return out


def engagement(df):
    d = df[df.valid_ctr & df.created.notna()].copy()
    d["logit_ctr"] = np.log((d.clicks + 0.5) / (d.impressions - d.clicks + 0.5))
    d["log_impr"] = np.log(d.impressions)
    rhs = " + ".join(f"d_{e}" for e in EMOTIONS6)
    f = f"logit_ctr ~ {rhs} + log_impr + C(audience) + C(quarter)"
    fit = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.page_family)[0]})
    res = {}
    for e in EMOTIONS6:
        b, (lo, hi) = fit.params[f"d_{e}"], fit.conf_int().loc[f"d_{e}"]
        res[e] = dict(odds_change_pct=float(100 * (np.exp(b) - 1)), ci95_pct=[float(100 * (np.exp(lo) - 1)),
                      float(100 * (np.exp(hi) - 1))], p=float(fit.pvalues[f"d_{e}"]), n_with=int(d[f"d_{e}"].sum()))
    # robustness: continuous intensities instead of detected flags
    rhs2 = " + ".join(f"np.log(i_{e})" for e in EMOTIONS6)
    fit2 = smf.ols(f"logit_ctr ~ {rhs2} + log_impr + C(audience) + C(quarter)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.page_family)[0]})
    robust = {e: dict(coef=float(fit2.params[f"np.log(i_{e})"]), p=float(fit2.pvalues[f"np.log(i_{e})"]))
              for e in EMOTIONS6}
    # pooled test (more power): any negative emotion vs joy vs neither, same controls
    neg = ["anger", "disgust", "fear", "sadness"]
    d["negative"] = d[[f"d_{e}" for e in neg]].max(axis=1)
    d["neg_intensity"] = np.log(d[[f"i_{e}" for e in neg]].max(axis=1))
    pooled = {}
    for name, f3 in (("flags", "logit_ctr ~ negative + d_joy + log_impr + C(audience) + C(quarter)"),
                     ("intensity", "logit_ctr ~ neg_intensity + np.log(i_joy) + log_impr + C(audience) + C(quarter)")):
        ft = smf.ols(f3, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.page_family)[0]})
        key = "negative" if name == "flags" else "neg_intensity"
        jkey = "d_joy" if name == "flags" else "np.log(i_joy)"
        lo, hi = ft.conf_int().loc[key]
        pooled[name] = dict(negative_coef=float(ft.params[key]), negative_p=float(ft.pvalues[key]),
                            negative_odds_change_pct=float(100 * (np.exp(ft.params[key]) - 1)),
                            negative_ci95_pct=[float(100 * (np.exp(lo) - 1)), float(100 * (np.exp(hi) - 1))],
                            joy_coef=float(ft.params[jkey]), joy_p=float(ft.pvalues[jkey]))
    pooled["n_negative_ads"] = int(d.negative.sum())
    # within-audience check: does the sign hold inside each large audience?
    within = {}
    for aud, g in d.groupby("audience"):
        if len(g) >= 80 and g.negative.sum() >= 15:
            fw = smf.ols("logit_ctr ~ negative + d_joy + log_impr + C(quarter)", data=g).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(g.page_family)[0]})
            lo, hi = fw.conf_int().loc["negative"]
            within[aud] = dict(n=int(len(g)), n_negative=int(g.negative.sum()),
                               odds_change_pct=float(100 * (np.exp(fw.params["negative"]) - 1)),
                               ci95_pct=[float(100 * (np.exp(lo) - 1)), float(100 * (np.exp(hi) - 1))],
                               p=float(fw.pvalues["negative"]))
    pooled["within_audience"] = within
    raw = d.groupby("dominant").apply(lambda g: pd.Series(dict(ads=len(g), clicks=g.clicks.sum(),
                                      impressions=g.impressions.sum(), pooled_ctr=g.clicks.sum() / g.impressions.sum(),
                                      median_ctr=g.ctr.median())), include_groups=False)
    return dict(n_ads=int(len(d)), n_page_clusters=int(d.page_family.nunique()), r2=float(fit.rsquared),
                pooled_negative=pooled, detected_flags=res, intensity_model=robust, raw_by_dominant=raw.round(4).to_dict(orient="index"))


def paid_vs_earned(df):
    d = df[df.valid_ctr & df.spend_rub.notna()]
    g = d.groupby("dominant").agg(rub=("spend_rub", "sum"), clicks=("clicks", "sum"), impressions=("impressions", "sum"),
                                  ads=("id", "size"))
    g["spend_share"], g["click_share"] = g.rub / g.rub.sum(), g.clicks / g.clicks.sum()
    g["rub_per_click"] = g.rub / g.clicks
    g["rub_per_1000_impressions"] = 1000 * g.rub / g.impressions
    return dict(n_ads=int(len(d)), total_rub=float(d.spend_rub.sum()), by_emotion=g.round(4).to_dict(orient="index"))


def timeline(df):
    d = df.dropna(subset=["created"]).copy()
    d = d[(d.created >= "2015-06-01") & (d.created < "2017-09-01")]
    d["month"] = d.created.dt.tz_convert(None).dt.to_period("M").dt.start_time
    m = d.groupby("month").agg(ads=("id", "size"), **{e: (f"d_{e}", "mean") for e in EMOTIONS6})
    m.to_csv(os.path.join(RES, "monthly_emotion_mix.csv"))
    pre = d[(d.created >= "2016-08-01") & (d.created < "2016-11-09")]
    base = d[(d.created >= "2016-01-01") & (d.created < "2016-08-01")]
    post = d[(d.created >= "2016-11-09") & (d.created < "2017-02-01")]
    win = lambda x: {e: float(x[f"d_{e}"].mean()) for e in EMOTIONS6} | {"ads": int(len(x))}
    return dict(jan_jul_2016=win(base), aug_to_election=win(pre), election_to_jan_2017=win(post))


def cross_platform(model, per_role=15000, seed=0):
    if not os.path.exists(TWEETS):
        return None
    t = pd.read_parquet(TWEETS, columns=["content", "language", "account_category", "retweet", "utc"])
    t = t[(t.language == "English") & (t.retweet == 0) & (t.utc >= "2015-06-19") & (t.utc < "2018-01-01")]
    t = t[t.account_category.isin(["RightTroll", "LeftTroll", "HashtagGamer", "Fearmonger", "NewsFeed"])]
    t = t[t.content.str.len() >= 20]
    s = pd.concat([g.sample(min(per_role, len(g)), random_state=seed) for _, g in t.groupby("account_category")])
    det = (model.intensity(s.content.tolist())[:, :6] >= 1)
    out = {}
    for role in s.account_category.unique():
        m = (s.account_category == role).to_numpy()
        out[role] = dict(tweets=int(m.sum()), **{e: float(det[m, j].mean()) for j, e in enumerate(EMOTIONS6)})
    return out


def validation_sample(df, n_per=25, seed=0):
    """Stratified sample for human annotation (the Reddit -> Facebook domain check)."""
    s = pd.concat([g.sample(min(n_per, len(g)), random_state=seed) for _, g in df.groupby("dominant")])
    s = s.sample(frac=1, random_state=seed)[["id", "text"]]
    for e in EKMAN:
        s[e] = ""
    s.to_csv(os.path.join(RES, "human_validation_sample.csv"), index=False)
    return len(s)


def run():
    model = joblib.load(os.path.join(MODELS, "emotion_detector.joblib"))
    df = score_ads(model)
    res = dict(ads_with_text=int(len(df)), audiences=df.audience.value_counts().to_dict(),
               emotional_targeting=emotional_targeting(df), engagement=engagement(df),
               paid_vs_earned=paid_vs_earned(df), timeline=timeline(df))
    res["validation_sample_size"] = validation_sample(df)
    print("cross-platform scoring ...")
    res["cross_platform_tweets"] = cross_platform(model)
    json.dump(res, open(os.path.join(RES, "analysis.json"), "w"), indent=2, default=float)
    return res


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k not in ("emotional_targeting",)}, indent=1, default=float)[:6000])
