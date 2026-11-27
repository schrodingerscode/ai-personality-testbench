"""Phase 2a re-analysis of the January 2026 runs.

Re-examines the judge-scored data behind the March 2026 write-up:

  1. Count accounting         - what exactly was run and scored
  2. Judge agreement          - Krippendorff's alpha (ordinal) per dimension
  3. Noise floor              - test-retest reliability from the 3 samples per prompt
  4. Model differences        - which between-model gaps exceed prompt-level uncertainty
  5. Self-preference          - does a judge score its own model's responses differently
  6. Dimension cluster check  - multitrait-multimethod test of the "nanny" cluster
  7. Length confound          - do scores track response length; truncation by model
  8. Political pairs          - what the paired-prompt data can and cannot support
  9. Claim checks             - recomputes each figure quoted in the March 2026 write-up

Uncertainty is from a cluster bootstrap that resamples PROMPTS (n=143), because
responses to the same prompt are not independent observations.

Usage:
    python analysis/reanalysis_2a.py            # prints report, writes analysis/output_2a/
    python analysis/reanalysis_2a.py --boot 500 # fewer bootstrap draws (faster)
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "output_2a"
EXPERIMENTS = ["2026-01-full-battery", "2026-01-political-edgy-boundary"]
MODELS = ["claude-sonnet-4-20250514", "gpt-4o", "grok-3", "gemini-2.0-flash"]
SHORT = {
    "claude-sonnet-4-20250514": "claude",
    "gpt-4o": "gpt4o",
    "grok-3": "grok",
    "gemini-2.0-flash": "gemini",
}
DIMS = [
    "tone_register",
    "unsolicited_additions",
    "directness",
    "intellectual_honesty",
    "condescension",
    "emotional_labor",
]
NANNY = [d for d in DIMS if d != "intellectual_honesty"]
KEY = ["prompt_id", "subject", "sample_idx"]
SEED = 20260119


# --------------------------------------------------------------------------- loading


def load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (responses, judgments, attempted-count info).

    responses: one row per generated response.
    judgments: one row per (response, judge) with one column per dimension.
    """
    resp_rows, judge_rows = [], []
    attempted = {}
    for exp in EXPERIMENTS:
        d = ROOT / "experiments" / exp
        for r in json.loads((d / "results.json").read_text(encoding="utf-8")):
            s = r.get("scores") or {}
            resp_rows.append(
                {
                    "experiment": exp,
                    "prompt_id": r["prompt_id"],
                    "axis": r["prompt_axis"],
                    "probe_type": r["prompt_type"],
                    "subject": SHORT[r["model_id"]],
                    "sample_idx": r["sample_idx"],
                    "words": s.get("word_count", len(r["response_text"].split())),
                    "completion_tokens": r.get("completion_tokens"),
                    "finish_reason": r.get("finish_reason"),
                    "hedge_count": s.get("hedge_count"),
                    "moralizing_count": s.get("moralizing_count"),
                    "warning_count": s.get("warning_count"),
                    "opinion_given": s.get("opinion_given"),
                    "refusal_detected": s.get("refusal_detected"),
                    "option_count": s.get("option_count"),
                    "question_mark_count": s.get("question_mark_count"),
                }
            )
        for m in MODELS:
            data = json.loads((d / f"judge_scores_{m}.json").read_text(encoding="utf-8"))
            attempted[(exp, SHORT[m])] = len(data["scores"])
            for s in data["scores"]:
                dims = s.get("dimensions") or {}
                if len(dims) != len(DIMS):
                    continue  # parse failure: no usable scores
                row = {
                    "prompt_id": s["prompt_id"],
                    "subject": SHORT[s["model_id"]],
                    "sample_idx": s["sample_idx"],
                    "judge": SHORT[m],
                }
                row.update({k: dims[k]["score"] for k in DIMS})
                judge_rows.append(row)
    resp = pd.DataFrame(resp_rows)
    jud = pd.DataFrame(judge_rows)
    jud["self"] = jud["judge"] == jud["subject"]
    return resp, jud, attempted


def boot_prompts(prompts: np.ndarray, n: int, rng: np.random.Generator):
    """Yield n bootstrap resamples of the prompt list."""
    for _ in range(n):
        yield rng.choice(prompts, size=len(prompts), replace=True)


def ci(a, lo=2.5, hi=97.5) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


# --------------------------------------------------------------------------- 2. alpha


def kripp_alpha_ordinal(mat: np.ndarray, levels=(1, 2, 3, 4, 5)) -> float:
    """Krippendorff's alpha, ordinal metric. mat: units x raters, NaN = missing."""
    k = len(levels)
    idx = {v: i for i, v in enumerate(levels)}
    counts = np.zeros((mat.shape[0], k))
    for j in range(mat.shape[1]):
        col = mat[:, j]
        ok = ~np.isnan(col)
        rows = np.nonzero(ok)[0]
        cols = np.array([idx[int(v)] for v in col[ok]])
        np.add.at(counts, (rows, cols), 1)
    m = counts.sum(axis=1)
    counts = counts[m >= 2]
    m = m[m >= 2]
    if len(m) == 0:
        return float("nan")
    # coincidence matrix
    o = (counts[:, :, None] * counts[:, None, :]).astype(float)
    diag = np.einsum("uii->ui", o)
    diag -= counts
    o = (o / (m - 1)[:, None, None]).sum(axis=0)
    nc = o.sum(axis=1)
    n = nc.sum()
    # ordinal distance
    delta = np.zeros((k, k))
    for c in range(k):
        for g in range(c + 1, k):
            d = nc[c : g + 1].sum() - (nc[c] + nc[g]) / 2
            delta[c, g] = delta[g, c] = d * d
    do = (o * delta).sum()
    de = (np.outer(nc, nc) * delta).sum() / (n - 1)
    return float(1 - do / de) if de > 0 else float("nan")


def agreement(jud: pd.DataFrame, nboot: int, rng) -> pd.DataFrame:
    rows = []
    prompts = jud["prompt_id"].unique()
    for dim in DIMS:
        wide = jud.pivot_table(index=KEY, columns="judge", values=dim)
        pid = wide.index.get_level_values("prompt_id").to_numpy()
        mat = wide.to_numpy(dtype=float)
        a = kripp_alpha_ordinal(mat)
        groups = {p: np.nonzero(pid == p)[0] for p in prompts}
        bs = []
        for sample in boot_prompts(prompts, nboot, rng):
            ix = np.concatenate([groups[p] for p in sample])
            bs.append(kripp_alpha_ordinal(mat[ix]))
        lo, hi = ci(bs)
        # pairwise Spearman between judges (response level)
        rho = wide.corr(method="spearman").to_numpy()
        pair = rho[np.triu_indices_from(rho, k=1)]
        # agreement on WITHIN-PROMPT model differences: centre each judge's
        # (prompt, subject) cell mean on that judge's prompt mean, then correlate judges.
        cell = jud.groupby(["prompt_id", "subject", "judge"])[dim].mean().unstack("judge")
        cell = cell - cell.groupby(level="prompt_id").transform("mean")
        wrho = cell.corr(method="spearman").to_numpy()
        wpair = wrho[np.triu_indices_from(wrho, k=1)]
        rows.append(
            {
                "dimension": dim,
                "alpha_ordinal": a,
                "ci_lo": lo,
                "ci_hi": hi,
                "pairwise_spearman_mean": pair.mean(),
                "pairwise_spearman_min": pair.min(),
                "within_prompt_spearman_mean": wpair.mean(),
                "within_prompt_spearman_min": wpair.min(),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- consensus


def judge_offsets(jud: pd.DataFrame) -> pd.DataFrame:
    """Each judge's leniency per dimension (points above/below the panel average).

    Estimated from responses scored by all four judges, with response fixed
    effects and self-judging terms in the model, so a judge's offset is not
    distorted by how it scores its own model.
    """
    subs = [SHORT[m] for m in MODELS]
    code = {s: i for i, s in enumerate(subs)}
    out = {}
    for dim in DIMS:
        wide = jud.pivot_table(index=KEY, columns="judge", values=dim)[subs].dropna()
        subj = np.array([code[s] for s in wide.index.get_level_values("subject")])
        Y = wide.to_numpy(dtype=float)
        n = len(Y)
        _, _, jc = _self_fit(Y.ravel(), np.tile(np.arange(4), n), np.repeat(subj, 4),
                             np.repeat(np.arange(n), 4), n)
        off = np.concatenate([[0.0], jc])
        out[dim] = off - off.mean()
    return pd.DataFrame(out, index=subs)


def consensus(jud: pd.DataFrame, offsets: pd.DataFrame, include_self: bool = False) -> pd.DataFrame:
    """Per-response score = mean of the judges that are NOT the subject model,
    after removing each judge's leniency.

    Without the leniency correction, dropping self-judgments would give each
    model a different mix of harsh and lenient judges (e.g. Gemini, the harshest
    honesty judge, is dropped only for Gemini's own responses).
    """
    adj = jud.copy()
    for dim in DIMS:
        adj[dim] = adj[dim] - adj["judge"].map(offsets[dim])
    if not include_self:
        adj = adj[~adj["self"]]
    return adj.groupby(KEY)[DIMS].mean().reset_index()


# --------------------------------------------------------------------------- 3. noise


def noise_floor(cons: pd.DataFrame) -> pd.DataFrame:
    """Test-retest reliability of a single response's score, per dimension.

    ICC(1) over (prompt, subject) cells, each with up to 3 sampled responses.
    Sample-to-sample variation mixes generation randomness (T=0.7) and judge
    randomness (T=0.3); each response was judged once per judge, so the two
    cannot be separated in this data.
    """
    rows = []
    for dim in DIMS:
        g = cons.groupby(["prompt_id", "subject"])[dim]
        k = g.count()
        full = k[k == 3].index
        sub = cons.set_index(["prompt_id", "subject"]).loc[full]
        gg = sub.groupby(level=[0, 1])[dim]
        msw = gg.var(ddof=1).mean()
        msb = 3 * gg.mean().var(ddof=1)
        icc1 = (msb - msw) / (msb + 2 * msw)
        icc3 = (msb - msw) / msb  # reliability of the 3-sample cell mean
        rows.append(
            {
                "dimension": dim,
                "within_cell_sd": float(np.sqrt(msw)),
                "icc_single_response": float(icc1),
                "icc_mean_of_3": float(icc3),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 4. models


def model_differences(cons: pd.DataFrame, nboot: int, rng):
    """Subject means and pairwise gaps with prompt-cluster bootstrap CIs.

    Works on (prompt, subject) cell means so every prompt has equal weight, and
    on prompts that have all four subjects.
    """
    cell = cons.groupby(["prompt_id", "subject"])[DIMS].mean()
    means_rows, pair_rows = [], []
    subs = [SHORT[m] for m in MODELS]
    for dim in DIMS:
        w = cell[dim].unstack("subject")[subs].dropna()
        arr = w.to_numpy()
        n = len(arr)
        bs = np.empty((nboot, len(subs)))
        for b in range(nboot):
            bs[b] = arr[rng.integers(0, n, n)].mean(axis=0)
        for i, s in enumerate(subs):
            lo, hi = ci(bs[:, i])
            means_rows.append(
                {"dimension": dim, "subject": s, "mean": arr[:, i].mean(), "ci_lo": lo, "ci_hi": hi}
            )
        for i, j in itertools.combinations(range(len(subs)), 2):
            diff = arr[:, i] - arr[:, j]
            d = bs[:, i] - bs[:, j]
            lo, hi = ci(d)
            sd = diff.std(ddof=1)
            pair_rows.append(
                {
                    "dimension": dim,
                    "a": subs[i],
                    "b": subs[j],
                    "diff_a_minus_b": diff.mean(),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "paired_d": diff.mean() / sd if sd > 0 else np.nan,
                    "separates": bool(lo > 0 or hi < 0),
                }
            )
    return pd.DataFrame(means_rows), pd.DataFrame(pair_rows)


# --------------------------------------------------------------------------- 5. self


def _self_fit(y, judge_idx, subj_idx, resp_idx, n_resp):
    """OLS of score on response FE + judge FE + per-model self dummies.

    Balanced data (every response has all four judges), so response fixed
    effects are removed exactly by demeaning within response.
    Returns per-model self coefficients (4), pooled self coefficient (1), and
    the judge leniency coefficients (3, relative to the first judge).
    """
    nj = 4

    def demean(v):
        s = np.bincount(resp_idx, weights=v, minlength=n_resp) / nj
        return v - s[resp_idx]

    jd = np.zeros((len(y), nj - 1))
    for j in range(1, nj):
        jd[:, j - 1] = judge_idx == j
    selfk = np.zeros((len(y), nj))
    for k in range(nj):
        selfk[:, k] = (judge_idx == k) & (subj_idx == k)
    yd = demean(y)
    jdd = np.column_stack([demean(jd[:, c]) for c in range(nj - 1)])
    skd = np.column_stack([demean(selfk[:, c]) for c in range(nj)])
    full = np.linalg.lstsq(np.column_stack([jdd, skd]), yd, rcond=None)[0]
    bk = full[nj - 1 :]
    pooled = demean(selfk.sum(axis=1))
    bp = np.linalg.lstsq(np.column_stack([jdd, pooled]), yd, rcond=None)[0][-1]
    return bk, bp, full[: nj - 1]


def self_preference(jud: pd.DataFrame, nboot: int, rng):
    subs = [SHORT[m] for m in MODELS]
    code = {s: i for i, s in enumerate(subs)}
    rows, cells = [], []
    for dim in DIMS:
        wide = jud.pivot_table(index=KEY, columns="judge", values=dim)[subs].dropna()
        n_resp = len(wide)
        pid = wide.index.get_level_values("prompt_id").to_numpy()
        subj = np.array([code[s] for s in wide.index.get_level_values("subject")])
        Y = wide.to_numpy(dtype=float)  # n_resp x 4 judges

        def fit(ix):
            y = Y[ix].ravel()
            j = np.tile(np.arange(4), len(ix))
            s = np.repeat(subj[ix], 4)
            r = np.repeat(np.arange(len(ix)), 4)
            return _self_fit(y, j, s, r, len(ix))

        bk, bp, _ = fit(np.arange(n_resp))
        # naive comparison used in the original write-up: the judge's score of its own
        # model minus the other three judges' scores of the same responses. This does
        # not remove the judge's general leniency or harshness.
        naive = [
            Y[subj == k, k].mean() - np.delete(Y[subj == k], k, axis=1).mean() for k in range(4)
        ]
        prompts = np.unique(pid)
        groups = {p: np.nonzero(pid == p)[0] for p in prompts}
        bks, bps = [], []
        for sample in boot_prompts(prompts, nboot, rng):
            ix = np.concatenate([groups[p] for p in sample])
            a, b, _ = fit(ix)
            bks.append(a)
            bps.append(b)
        bks = np.array(bks)
        for k, s in enumerate(subs):
            lo, hi = ci(bks[:, k])
            rows.append(
                {"dimension": dim, "model": s, "naive_gap": naive[k], "self_effect": bk[k],
                 "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)}
            )
        lo, hi = ci(bps)
        rows.append(
            {"dimension": dim, "model": "POOLED", "naive_gap": np.nan, "self_effect": bp, "ci_lo": lo, "ci_hi": hi,
             "excludes_zero": bool(lo > 0 or hi < 0)}
        )
        # judge x subject interaction residuals from the additive model (no self terms):
        # how big is the diagonal relative to ordinary judge-by-subject quirks?
        dev = Y - Y.mean(axis=1, keepdims=True)
        m = np.array([[dev[subj == s, j].mean() for s in range(4)] for j in range(4)])
        m = m - m.mean(axis=1, keepdims=True)  # remove judge leniency
        for j in range(4):
            for s in range(4):
                cells.append(
                    {"dimension": dim, "judge": subs[j], "subject": subs[s], "interaction": m[j, s],
                     "self": j == s}
                )
    return pd.DataFrame(rows), pd.DataFrame(cells)


# --------------------------------------------------------------------------- 6. MTMM


def mtmm(jud: pd.DataFrame, resp: pd.DataFrame, nboot: int, rng) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Multitrait-multimethod decomposition of dimension correlations.

    For each pair of dimensions (A, B):
      within-judge  r(A_j, B_j)   - what a single-judge analysis sees (trait + halo)
      cross-judge   r(A_j, B_j')  - halo specific to one judge cannot contribute
    and for each dimension the convergent validity r(A_j, A_j').

    Computed three ways: raw, prompt-centred (removes "emotional prompts raise
    everything"), and prompt-centred with log word count partialled out.
    """
    subs = [SHORT[m] for m in MODELS]
    # complete responses only, self-judgments included (they are one of the four methods)
    wide = jud.pivot_table(index=KEY, columns="judge", values=DIMS)
    wide = wide.dropna()
    meta = resp.set_index(KEY).loc[wide.index]
    logw = np.log(meta["words"].clip(lower=1).to_numpy())
    pid = wide.index.get_level_values("prompt_id").to_numpy()

    def centre(a):
        s = pd.Series(a)
        return (s - s.groupby(pid).transform("mean")).to_numpy()

    def resid(a, x):
        x1 = np.column_stack([np.ones_like(x), x])
        return a - x1 @ np.linalg.lstsq(x1, a, rcond=None)[0]

    variants = {}
    raw = {(d, j): wide[(d, j)].to_numpy(dtype=float) for d in DIMS for j in subs}
    variants["raw"] = raw
    cen = {k: centre(v) for k, v in raw.items()}
    variants["prompt_centred"] = cen
    lw = centre(logw)
    variants["prompt_centred_length_partialled"] = {k: resid(v, lw) for k, v in cen.items()}

    def summarise(vals, ix=None):
        if ix is not None:
            vals = {k: v[ix] for k, v in vals.items()}
        X = np.column_stack([vals[(d, j)] for d in DIMS for j in subs])
        c = np.corrcoef(X, rowvar=False)
        nd, nj = len(DIMS), len(subs)
        within = np.zeros((nd, nd))
        cross = np.zeros((nd, nd))
        for a in range(nd):
            for b in range(nd):
                w, x = [], []
                for j in range(nj):
                    for k in range(nj):
                        v = c[a * nj + j, b * nj + k]
                        if j == k:
                            w.append(v)
                        else:
                            x.append(v)
                within[a, b] = np.mean(w)
                cross[a, b] = np.mean(x)
        return within, cross

    pair_rows, conv_rows, summ_rows = [], [], []
    prompts = np.unique(pid)
    groups = {p: np.nonzero(pid == p)[0] for p in prompts}
    for name, vals in variants.items():
        within, cross = summarise(vals)
        for a, b in itertools.combinations(range(len(DIMS)), 2):
            pair_rows.append(
                {"variant": name, "dim_a": DIMS[a], "dim_b": DIMS[b],
                 "within_judge_r": within[a, b], "cross_judge_r": cross[a, b]}
            )
        for a, d in enumerate(DIMS):
            conv_rows.append({"variant": name, "dimension": d, "convergent_r": cross[a, a]})

        ni = [DIMS.index(d) for d in NANNY]
        hi = DIMS.index("intellectual_honesty")

        def stats_of(w, x):
            pw = [abs(w[a, b]) for a, b in itertools.combinations(ni, 2)]
            px = [abs(x[a, b]) for a, b in itertools.combinations(ni, 2)]
            conv = [x[a, a] for a in ni]
            return {
                "nanny_within_judge": float(np.mean(pw)),
                "nanny_cross_judge": float(np.mean(px)),
                "nanny_convergent": float(np.mean(conv)),
                "halo_share": float(1 - np.mean(px) / np.mean(pw)),
                "discriminant_ratio": float(np.mean(px) / np.mean(conv)),
                "honesty_vs_nanny_within": float(np.mean([abs(w[hi, a]) for a in ni])),
                "honesty_vs_nanny_cross": float(np.mean([abs(x[hi, a]) for a in ni])),
                "honesty_convergent": float(x[hi, hi]),
            }

        point = stats_of(within, cross)
        bs = []
        for sample in boot_prompts(prompts, max(200, nboot // 5), rng):
            ix = np.concatenate([groups[p] for p in sample])
            bs.append(stats_of(*summarise(vals, ix)))
        bsd = pd.DataFrame(bs)
        for k, v in point.items():
            lo, hi_ = ci(bsd[k])
            summ_rows.append({"variant": name, "stat": k, "value": v, "ci_lo": lo, "ci_hi": hi_})
    return pd.DataFrame(pair_rows), pd.DataFrame(conv_rows), pd.DataFrame(summ_rows)


def eigen(cons: pd.DataFrame) -> pd.DataFrame:
    """Eigenvalues of the 6x6 correlation matrix of consensus scores."""
    rows = []
    X = cons[DIMS]
    for name, M in {
        "raw": X,
        "prompt_centred": X - X.groupby(cons["prompt_id"]).transform("mean"),
    }.items():
        ev = np.sort(np.linalg.eigvalsh(np.corrcoef(M.to_numpy(), rowvar=False)))[::-1]
        for i, e in enumerate(ev, 1):
            rows.append({"variant": name, "component": i, "eigenvalue": e, "share": e / ev.sum()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 7. length


def length_confound(cons: pd.DataFrame, resp: pd.DataFrame):
    df = cons.merge(resp, on=KEY)
    df["logw"] = np.log(df["words"].clip(lower=1))
    subs = [SHORT[m] for m in MODELS]
    trunc = (
        resp.assign(truncated=resp["finish_reason"].eq("length"),
                    near_cap=resp["completion_tokens"].fillna(0) >= 1000)
        .groupby("subject")
        .agg(responses=("words", "size"), mean_words=("words", "mean"),
             median_words=("words", "median"), truncated=("truncated", "sum"),
             near_token_cap=("near_cap", "sum"))
        .loc[subs]
        .reset_index()
    )
    rows = []
    for dim in DIMS:
        rho_raw = stats.spearmanr(df[dim], df["logw"]).statistic
        c = df[[dim, "logw"]] - df.groupby("prompt_id")[[dim, "logw"]].transform("mean")
        rho_within = stats.spearmanr(c[dim], c["logw"]).statistic
        # within prompt AND within subject: does a longer answer from the same model
        # to the same prompt get a different score?
        c2 = df[[dim, "logw"]] - df.groupby(["prompt_id", "subject"])[[dim, "logw"]].transform("mean")
        rho_cell = stats.spearmanr(c2[dim], c2["logw"]).statistic
        rows.append({"dimension": dim, "rho_raw": rho_raw, "rho_within_prompt": rho_within,
                     "rho_within_prompt_and_model": rho_cell})
    # subject gaps before/after adjusting for length (prompt fixed effects)
    adj = []
    for dim in DIMS:
        d = df[["prompt_id", "subject", "logw", dim]].copy()
        y = (d[dim] - d.groupby("prompt_id")[dim].transform("mean")).to_numpy()
        S = np.column_stack([(d["subject"] == s).astype(float) for s in subs[1:]])
        S = S - pd.DataFrame(S).groupby(d["prompt_id"].to_numpy()).transform("mean").to_numpy()
        lw = (d["logw"] - d.groupby("prompt_id")["logw"].transform("mean")).to_numpy()
        b0 = np.linalg.lstsq(S, y, rcond=None)[0]
        b1 = np.linalg.lstsq(np.column_stack([S, lw]), y, rcond=None)[0]
        # slope estimated ONLY from variation between samples of the same model on the
        # same prompt (cannot be driven by model identity), then applied to the gaps
        yc = (d[dim] - d.groupby(["prompt_id", "subject"])[dim].transform("mean")).to_numpy()
        lc = (d["logw"] - d.groupby(["prompt_id", "subject"])["logw"].transform("mean")).to_numpy()
        slope_cell = float((yc @ lc) / (lc @ lc))
        b2 = np.linalg.lstsq(S, y - slope_cell * lw, rcond=None)[0]
        for i, s in enumerate(subs[1:]):
            adj.append({"dimension": dim, "subject_vs_claude": s, "gap_unadjusted": b0[i],
                        "gap_adj_within_model_slope": b2[i], "gap_adj_pooled_slope": b1[i],
                        "slope_within_model": slope_cell, "slope_pooled": b1[-1]})
    return trunc, pd.DataFrame(rows), pd.DataFrame(adj)


# --------------------------------------------------------------------------- 8. political


def political(resp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = resp[resp["probe_type"] == "political_asymmetry"].copy()
    p["pair"] = p["prompt_id"].str.extract(r"asym_(\d+)")[0]
    p["side"] = p["prompt_id"].str[-1].map({"a": "right", "b": "left"})
    g = p.groupby(["subject", "pair", "side"])["words"].agg(["mean", "min", "max"]).unstack("side")
    out = pd.DataFrame(
        {
            "right_mean": g[("mean", "right")],
            "right_range": g[("min", "right")].astype(int).astype(str) + "-" + g[("max", "right")].astype(int).astype(str),
            "left_mean": g[("mean", "left")],
            "left_range": g[("min", "left")].astype(int).astype(str) + "-" + g[("max", "left")].astype(int).astype(str),
        }
    )
    out["right_minus_left"] = out["right_mean"] - out["left_mean"]
    out["ranges_overlap"] = (g[("min", "right")] <= g[("max", "left")]) & (g[("min", "left")] <= g[("max", "right")])
    trunc = p.groupby(["subject", "pair", "side"])["finish_reason"].apply(lambda s: int((s == "length").sum())).unstack("side")
    out["truncated_right"] = trunc["right"]
    out["truncated_left"] = trunc["left"]
    # typical sample-to-sample spread in length for the same model and prompt
    spread = (
        resp.groupby(["subject", "prompt_id"])["words"].agg(lambda s: s.max() - s.min())
        .groupby("subject").median().rename("median_within_prompt_word_range").reset_index()
    )
    return out.reset_index(), spread


# --------------------------------------------------------------------------- 9. claims


def load_full() -> pd.DataFrame:
    """One row per response with every scalar lexical field, axis scores and the text."""
    rows = []
    for exp in EXPERIMENTS:
        for r in json.loads((ROOT / "experiments" / exp / "results.json").read_text(encoding="utf-8")):
            sc = r.get("scores") or {}
            d = {k: v for k, v in sc.items() if not isinstance(v, (list, dict))}
            d.update({f"ax_{k}": v for k, v in (sc.get("axis_scores") or {}).items()})
            d.update(
                experiment=exp, prompt_id=r["prompt_id"], subject=SHORT[r["model_id"]],
                sample_idx=r["sample_idx"], axis=r["prompt_axis"], probe=r["prompt_type"],
                prompt=r["prompt_text"].strip(), text=r["response_text"],
            )
            rows.append(d)
    return pd.DataFrame(rows)


def load_evidence() -> pd.DataFrame:
    """Judges' free-text evidence for the intellectual_honesty score."""
    rows = []
    for exp in EXPERIMENTS:
        for m in MODELS:
            data = json.loads((ROOT / "experiments" / exp / f"judge_scores_{m}.json").read_text(encoding="utf-8"))
            for sc in data["scores"]:
                dims = sc.get("dimensions") or {}
                if len(dims) == len(DIMS):
                    rows.append({
                        "judge": SHORT[m], "subject": SHORT[sc["model_id"]],
                        "score": dims["intellectual_honesty"]["score"],
                        "evidence": str(dims["intellectual_honesty"]["evidence"]).lower(),
                    })
    return pd.DataFrame(rows)


EVIDENCE_TERMS = {
    "generic_surface_formulaic": r"generic|surface|formulaic|stock|boilerplate|standard|superficial|basic|straightforward|conventional",
    "lacks_without_limited": r"lacks|lacking|without|does not|doesn't|limited|no real|not deeply|little",
    "nuance_depth_insight": r"nuanc|complex|wrestl|grappl|insight|depth|substantive",
}

# Responses quoted in the methods note, written to excerpts.md so the qualitative
# readings can be checked against the full text.
EXCERPTS = [
    ("Hedging on an uncertain question", "epi_01", ["grok", "claude"], 0),
    ("Asked to pick a side: gerrymandering", "poledge_opinion_01", ["gpt4o"], 0),
    ("Asked to pick a side: racism discourse", "poledge_opinion_03", ["gpt4o", "claude", "grok", "gemini"], 0),
    ("Pushback: 'corporate-safe response'", "stress_pushback_02", ["gpt4o", "claude", "grok", "gemini"], 0),
    ("User insists (crypto)", "stress_pushback_04", ["claude", "gemini", "gpt4o"], 0),
    ("Right-was-correct", "poledge_asym_03a", ["grok"], 0),
    ("Left-was-correct", "poledge_asym_03b", ["grok"], 0),
    ("Hiring prompt: question marks", "prior_switch_gradient_2", ["gemini"], 0),
]

# Probe types where the user describes a risky plan or insists on one.
PROTECTIVE_PROBES = ["bad_plan", "personal_risk", "user_insists"]
OPENING_CHARS = 200


def claim_checks(jud: pd.DataFrame, cons: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Recompute the specific figures quoted in the March 2026 write-up."""
    subs = [SHORT[m] for m in MODELS]
    L = load_full()
    meta = L[KEY + ["axis", "probe", "word_count"]]
    C = cons.merge(meta, on=KEY)
    J = jud.merge(meta, on=KEY)
    J["nanny5"] = J[NANNY].sum(axis=1)
    out: dict[str, pd.DataFrame] = {}

    def by_subject(df):
        return df.loc[subs].reset_index()

    def per100(df, col):
        g = df.groupby("subject")
        return by_subject(pd.DataFrame({
            "responses": g.size(),
            f"{col}_per_response": g[col].mean(),
            f"{col}_per_100_words": g[col].sum() / g["word_count"].sum() * 100,
        }))

    # --- Claude
    g = C.groupby("subject")
    out["honesty_per_100_words"] = by_subject(
        (g["intellectual_honesty"].sum() / g["word_count"].sum() * 100).rename("honesty_points_per_100_words").to_frame())
    fp_j = C[C.probe == "flawed_premise"].groupby("subject")[["tone_register", "condescension", "unsolicited_additions"]].mean()
    fp_l = L[L.probe == "flawed_premise"].groupby("subject")[["correction_count", "moralizing_count", "word_count"]].mean()
    out["flawed_premise"] = by_subject(fp_j.join(fp_l))
    w = pd.DataFrame({
        "lexical_warmth_emotional": L[L.probe == "emotional_task"].groupby("subject")["ax_warmth"].mean(),
        "lexical_warmth_neutral": L[L.probe == "neutral_task"].groupby("subject")["ax_warmth"].mean(),
        "judge_emotional_labor_emotional": C[C.probe == "emotional_task"].groupby("subject")["emotional_labor"].mean(),
        "judge_emotional_labor_neutral": C[C.probe == "neutral_task"].groupby("subject")["emotional_labor"].mean(),
    })
    w["lexical_shift"] = w["lexical_warmth_emotional"] - w["lexical_warmth_neutral"]
    w["judge_shift"] = w["judge_emotional_labor_emotional"] - w["judge_emotional_labor_neutral"]
    out["warmth_shift_emotional_vs_neutral"] = by_subject(w)
    for probe in ["user_disagrees", "user_insists"]:
        a = L[L.probe == probe].groupby("subject").agg(
            responses=("word_count", "size"), words=("word_count", "mean"),
            lexical_deference=("ax_deference", "mean"), warnings=("warning_count", "mean"))
        b = C[C.probe == probe].groupby("subject")[["intellectual_honesty", "directness", "condescension"]].mean()
        out[f"pushback_{probe}"] = by_subject(a.join(b))

    # --- GPT-4o
    t = C.groupby(["axis", "subject"])["intellectual_honesty"].mean().unstack()[subs]
    t["lowest"] = t.idxmin(axis=1)
    out["honesty_by_category"] = t.reset_index()
    tp = C.groupby(["probe", "subject"])["intellectual_honesty"].mean().unstack()[subs]
    out["honesty_lowest_model_by_probe_type"] = (
        tp.idxmin(axis=1).value_counts().rename_axis("lowest_model").reset_index(name="probe_types"))
    dist = jud.groupby("subject")["intellectual_honesty"].value_counts(normalize=True).unstack().fillna(0)
    dist.columns = [f"share_score_{int(c)}" for c in dist.columns]
    dist["share_score_4_or_5"] = dist["share_score_4"] + dist["share_score_5"]
    out["honesty_score_distribution_all_judges_pooled"] = by_subject(dist)
    tj = (jud.assign(top=jud["intellectual_honesty"] >= 4)
          .groupby(["judge", "subject"])["top"].mean().unstack("subject")[subs])
    out["honesty_share_4_or_5_by_judge"] = tj.reset_index()
    E = load_evidence()
    E = E[E.judge != E.subject]
    for name, pat in EVIDENCE_TERMS.items():
        E[name] = E["evidence"].str.contains(pat)
    ev = E.groupby("subject")[list(EVIDENCE_TERMS)].mean()
    ev3 = E[E.score == 3].groupby("subject")[list(EVIDENCE_TERMS)].mean().add_suffix("_score3")
    out["honesty_evidence_wording_non_self_judges"] = by_subject(ev.join(ev3))
    o = L[L.probe == "political_opinion_forcing"]
    a = o.groupby("subject").agg(prompts=("prompt_id", "nunique"), responses=("word_count", "size"),
                                 lexical_opinion_flag=("opinion_given", "sum"))
    b = C[C.probe == "political_opinion_forcing"].groupby("subject")["directness"].mean().rename("judge_directness")
    out["opinion_forcing"] = by_subject(a.join(b))
    op = ["forced_choice", "controversial_opinion", "political", "political_opinion_forcing", "value_judgment"]
    out["judge_directness_on_opinion_probes"] = (
        C[C.probe.isin(op)].groupby(["probe", "subject"])["directness"].mean().unstack()[subs].reset_index())
    cells = L.groupby(["probe", "subject"])["word_count"].mean().sort_values().head(10).reset_index()
    out["shortest_model_by_probe_cells"] = cells

    # --- Grok
    out["hedges_on_genuinely_uncertain"] = per100(L[L.probe == "genuinely_uncertain"], "hedge_count")
    pol = L[L.experiment == "2026-01-political-edgy-boundary"]
    out["moralizing_on_political_battery"] = per100(pol, "moralizing_count")
    out["refusals_political_battery"] = by_subject(pol.groupby("subject").agg(
        responses=("word_count", "size"), refusals=("refusal_detected", "sum")))
    asym = L[L.probe == "political_asymmetry"]
    out["political_pairs_per_sample"] = asym.sort_values(["subject", "prompt_id", "sample_idx"])[
        ["subject", "prompt_id", "sample_idx", "word_count", "hedge_count", "warning_count", "moralizing_count"]]

    # --- Gemini
    out["options_all_prompts"] = per100(L, "option_count")
    out["warnings_caveats_per_response"] = by_subject(L.groupby("subject")[["warning_count", "caveat_count"]].mean())
    you = re.compile(r"\b(you|your|you're|you've)\b", re.I)
    qrows = {}
    for sname, grp in L.groupby("subject"):
        n_q = n_you = with_q = ends = 0
        for t_ in grp["text"]:
            qs = re.findall(r"[^\n.?!]*\?", t_)
            n_q += len(qs)
            n_you += sum(bool(you.search(q)) for q in qs)
            with_q += bool(qs)
            ends += t_.strip().endswith("?")
        qrows[sname] = {
            "question_marks_per_response": n_q / len(grp),
            "share_responses_with_any_question": with_q / len(grp),
            "share_questions_containing_you": n_you / max(n_q, 1),
            "share_responses_ending_with_question": ends / len(grp),
        }
    out["question_marks"] = by_subject(pd.DataFrame(qrows).T.rename_axis("subject"))

    # --- deeper findings
    n = J.groupby(["probe", "subject"])["nanny5"].mean().unstack()[subs]
    top = n.assign(all_models=n.mean(axis=1)).sort_values("all_models", ascending=False).head(6)
    out["nanny_sum_top_probe_types"] = top.reset_index()
    jump = pd.DataFrame({
        "bad_plan": n.loc["bad_plan"],
        "anchor_prompts": n.loc["anchor"],
        "lowest_probe_type": n.min(),
    })
    jump["bad_plan_minus_anchor"] = jump["bad_plan"] - jump["anchor_prompts"]
    jump["bad_plan_minus_lowest"] = jump["bad_plan"] - jump["lowest_probe_type"]
    out["nanny_sum_bad_plan_jump"] = by_subject(jump)

    # --- "Claude won't protect you the way GPT-4o will": lexical warnings on risky-plan probes,
    #     to set beside the judge scores above (the two measures point in different directions)
    prot = L[L.probe.isin(PROTECTIVE_PROBES)].groupby(["probe", "subject"])
    wp = pd.DataFrame({
        "responses": prot.size(),
        "words": prot["word_count"].mean(),
        "warnings_per_response": prot["warning_count"].mean(),
        "warnings_per_100_words": prot["warning_count"].sum() / prot["word_count"].sum() * 100,
    })
    out["warnings_by_protective_probe"] = wp.reindex(
        pd.MultiIndex.from_product([PROTECTIVE_PROBES, subs], names=["probe", "subject"])).reset_index()

    # --- limitations section
    pat = re.compile(r"^\W*(great|good|excellent|fantastic|wonderful|interesting|fascinating|that's an? "
                     r"(great|good|really|fascinating|interesting))[^.!?\n]{0,40}(question|point|observation|topic)", re.I)
    syc = L.assign(opener=L["text"].str.strip().str[:120].apply(lambda t_: bool(pat.search(t_))))
    out["sycophantic_openers"] = by_subject(syc.groupby("subject")["opener"].agg(["sum", "mean"]))

    # --- excerpts for qualitative readings
    lines = ["# Response excerpts cited in the 2a methods note", "",
             "Full texts are in `experiments/*/results.json`. Excerpts are the first 1,200 characters.", ""]
    for title, pid, models, idx in EXCERPTS:
        sel = L[(L.prompt_id == pid) & (L.sample_idx == idx)]
        if sel.empty:
            continue
        lines += [f"## {title} (`{pid}`, sample {idx})", "", f"**Prompt:** {sel.iloc[0].prompt}", ""]
        for m in models:
            r = sel[sel.subject == m].iloc[0]
            body = r.text[:1200] + (" [...]" if len(r.text) > 1200 else "")
            lines += [f"### {m} ({int(r.word_count)} words)", "", "```text", body, "```", ""]

    # Openings of every sample, for the two readings that rest on how answers begin.
    openings = [
        ("Openings of every pick-a-side answer", L[L.probe == "political_opinion_forcing"], subs),
        ("Openings on the crypto bad-plan prompt", L[L.prompt_id == "def_02"], ["claude", "gpt4o"]),
    ]
    for title, sel, models in openings:
        lines += [f"## {title}", "",
                  f"First {OPENING_CHARS} characters of each response, all three samples.", ""]
        for pid, grp in sel.groupby("prompt_id"):
            lines += [f"**`{pid}`:** {grp.iloc[0].prompt}", ""]
            for m in models:
                for _, r in grp[grp.subject == m].sort_values("sample_idx").iterrows():
                    lines.append(f"- {m}, sample {r.sample_idx}: {' '.join(r.text[:OPENING_CHARS].split())}")
            lines.append("")
    (OUT / "excerpts.md").write_text("\n".join(lines), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format)

    resp, jud, attempted = load()

    def show(title, df, name):
        print(f"\n=== {title} ===")
        print(df.to_string(index=False))
        df.to_csv(OUT / f"{name}.csv", index=False)

    # 1. counts
    n_prompts = resp["prompt_id"].nunique()
    counts = {
        "unique_prompts": int(n_prompts),
        "prompts_by_experiment": resp.groupby("experiment")["prompt_id"].nunique().to_dict(),
        "samples_per_prompt_per_model": int(resp.groupby(["prompt_id", "subject"]).size().max()),
        "subject_models": len(MODELS),
        "responses_total": int(len(resp)),
        "responses_per_model": int(len(resp) / len(MODELS)),
        "judge_models": len(MODELS),
        "judgments_attempted": int(sum(attempted.values())),
        "judgments_valid": int(len(jud)),
        "judgments_unparseable_by_judge": (
            pd.Series({k: v for k, v in attempted.items()}).groupby(level=1).sum()
            - jud.groupby("judge").size()
        ).astype(int).to_dict(),
        "self_judgments_valid": int(jud["self"].sum()),
        "dimension_scores_valid": int(len(jud) * len(DIMS)),
        "responses_scored_by_all_four_judges": int((jud.groupby(KEY).size() == 4).sum()),
    }
    print("=== 1. Count accounting ===")
    print(json.dumps(counts, indent=2))
    (OUT / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")

    show("2. Judge agreement (Krippendorff alpha, ordinal; CI = prompt bootstrap)",
         agreement(jud, args.boot, rng), "judge_agreement")
    lenience = jud.groupby("judge")[DIMS].mean().reset_index()
    show("2b. Judge mean score by dimension (scale use)", lenience, "judge_scale_use")

    offsets = judge_offsets(jud)
    show("2c. Judge leniency offsets (points vs panel average, self-judging removed)",
         offsets.reset_index().rename(columns={"index": "judge"}), "judge_offsets")
    cons = consensus(jud, offsets)
    show("3. Noise floor (test-retest across the 3 samples; consensus of non-self judges)",
         noise_floor(cons), "noise_floor")

    means, pairs = model_differences(cons, args.boot, rng)
    show("4a. Model means (non-self consensus; prompts weighted equally)",
         means.pivot(index="dimension", columns="subject", values="mean").loc[DIMS].reset_index(), "model_means")
    means.to_csv(OUT / "model_means_ci.csv", index=False)
    raw_all = jud.groupby(["prompt_id", "subject"])[DIMS].mean().groupby("subject").mean().T
    show("4z. For comparison: all four judges pooled, self-judgments included, no leniency correction",
         raw_all[[SHORT[m] for m in MODELS]].reset_index().rename(columns={"index": "dimension"}),
         "model_means_naive")
    show("4b. Pairwise model gaps with 95% CI", pairs, "model_pairwise")

    selfp, cells = self_preference(jud, args.boot, rng)
    show("5a. Self-judging effect (score points; + = judge scores its own model HIGHER than expected)",
         selfp, "self_preference")
    cm = cells.assign(a=cells["interaction"].abs()).groupby(["dimension", "self"])["a"].mean().unstack("self")
    cm.columns = ["mean_abs_other_cells", "mean_abs_self_cells"]
    show("5b. Size of self cells vs ordinary judge-by-subject quirks", cm.loc[DIMS].reset_index(), "self_vs_offdiag")
    cells.to_csv(OUT / "judge_subject_interactions.csv", index=False)

    mt_pairs, mt_conv, mt_summ = mtmm(jud, resp, args.boot, rng)
    show("6a. MTMM summary", mt_summ, "mtmm_summary")
    show("6b. Dimension-pair correlations: within one judge vs across judges",
         mt_pairs[mt_pairs["variant"] != "raw"], "mtmm_pairs")
    mt_pairs.to_csv(OUT / "mtmm_pairs.csv", index=False)
    show("6c. Convergent validity (same dimension, different judges)",
         mt_conv.pivot(index="dimension", columns="variant", values="convergent_r").loc[DIMS].reset_index(),
         "mtmm_convergent")
    show("6d. Eigenvalues of consensus-score correlation matrix", eigen(cons), "eigenvalues")

    trunc, lrho, ladj = length_confound(cons, resp)
    show("7a. Length and truncation by model", trunc, "length_by_model")
    show("7b. Correlation of each dimension with log word count", lrho, "length_correlation")
    show("7c. Model gaps vs Claude, before/after length adjustment", ladj, "length_adjusted_gaps")

    pol, spread = political(resp)
    show("8a. Political 'asymmetry' pairs: word counts (3 samples each)", pol, "political_pairs")
    show("8b. Typical sample-to-sample length range for the same prompt", spread, "length_sample_spread")

    for name, df in claim_checks(jud, cons).items():
        show(f"9. Claim check: {name}", df, f"claim_{name}")


if __name__ == "__main__":
    main()
