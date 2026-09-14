#!/usr/bin/env python3
"""Recompute paper results from copyright-safe released analysis data."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from scipy.special import betaln, digamma, expit, gammaln
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr, t as student_t
from scipy.stats import norm
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUTS = HERE / "outputs"
RIGHT = ["right", "far_right"]
ROLES = ["source", "agent", "patient"]
FRAMES = ["conflict", "human_interest", "economic", "morality", "responsibility"]
SIGNED = {"neg": -1.0, "neu": 0.0, "pos": 1.0}
# A permuted statistic that ties the observed one must count as an exceedance on
# every platform. Comparing exactly lets last-bit rounding decide a tie, which
# moved one exceedance in 20,000 between BLAS builds.
PERMUTATION_EPSILON = 1e-12
# Full-precision fitted parameters used to generate the fixed H3 bootstrap
# stream. The model is re-fitted from the released counts on every run and must
# agree with this vector before it is used. Fixing the generating fit keeps
# tiny platform-level optimizer differences from changing NumPy's rejection-
# sampling path and therefore the entire seeded bootstrap stream.
H3_BOOTSTRAP_GENERATING_FIT = np.array([
    1.0793931955181222,
    0.4120797744561858,
    -1.7528078654236854,
    -0.8761132309507236,
    -1.3813930892793573,
    -1.5051341462774546,
    -1.3542469713161678,
    -0.45420791450401854,
    -1.006832997174138,
    -0.8373886324526401,
    -1.1501597980925884,
    -0.6984349793566555,
    2.844911990112587,
])
H3_BOOTSTRAP_FIT_TOLERANCE = 5e-5


def key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def rho(x, y) -> tuple[float, float]:
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def json_safe(value):
    if isinstance(value, dict): return {key: json_safe(item) for key,item in value.items()}
    if isinstance(value, list): return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
    if isinstance(value, np.integer): return int(value)
    return value


def krippendorff_alpha(matrix: np.ndarray, metric: str = "nominal") -> float:
    """Krippendorff alpha for units x raters, allowing missing cells."""
    values = sorted({v for v in matrix.ravel() if not pd.isna(v)})
    index = {v: i for i, v in enumerate(values)}
    coincidence = np.zeros((len(values), len(values)))
    for row in matrix:
        present = [v for v in row if not pd.isna(v)]
        if len(present) < 2:
            continue
        for i, first in enumerate(present):
            for j, second in enumerate(present):
                if i != j:
                    coincidence[index[first], index[second]] += 1.0 / (len(present) - 1)
    marginals = coincidence.sum(axis=1)
    total = coincidence.sum()

    def distance(i: int, j: int) -> float:
        if metric != "ordinal":
            return 0.0 if i == j else 1.0
        lo, hi = sorted((i, j))
        mass = marginals[lo : hi + 1].sum() - (marginals[lo] + marginals[hi]) / 2.0
        return float(mass**2)

    observed = sum(
        coincidence[i, j] * distance(i, j)
        for i in range(len(values)) for j in range(len(values))
    ) / total
    expected = sum(
        marginals[i] * marginals[j] * distance(i, j)
        for i in range(len(values)) for j in range(len(values))
    ) / (total * (total - 1))
    return float(1 - observed / expected) if expected else float("nan")


def outlet_asymmetry(data, value, ideology, floor=20):
    rows = []
    for outlet, group in data.groupby("outlet"):
        left = group[group.target_bloc == "left"]
        right = group[group.target_bloc.isin(RIGHT)]
        if len(left) < floor or len(right) < floor:
            continue
        rows.append({
            "outlet": outlet, "ideology": ideology[key(outlet)],
            "n_left": len(left), "n_right": len(right),
            "mean_left": float(left[value].mean()), "mean_right": float(right[value].mean()),
            "asymmetry": float(left[value].mean() - right[value].mean()),
        })
    return pd.DataFrame(rows).sort_values(["ideology", "outlet"]).reset_index(drop=True)


def gradient(data, value, ideology, floor=20):
    table = outlet_asymmetry(data, value, ideology, floor)
    statistic, p_value = rho(table.ideology, table.asymmetry)
    return {"n_outlets": len(table), "spearman_rho": statistic, "p_value": p_value}


def fixed_panel_permutation(table, draws=1_000_000, seed=20260911):
    ordered=table.sort_values("outlet"); x=rankdata(ordered.ideology.to_numpy()); y=rankdata(ordered.asymmetry.to_numpy()); x-=x.mean(); y-=y.mean()
    observed=abs(float(np.dot(x,y))); rng=np.random.default_rng(seed); exceed=0
    for _ in range(draws): exceed += abs(float(np.dot(rng.permutation(x),y))) >= observed-PERMUTATION_EPSILON*max(1.0,observed)
    return {"draws":draws,"seed":seed,"exceedances":exceed,"p_value":(exceed+1)/(draws+1),
            "shuffled_unit":"outlet","n_shuffled_units":len(ordered),
            "estimator":"(exceedances+1)/(draws+1)"}


def article_bootstrap_gradient(data,ideology,reps=1000,seed=42):
    sub=data[data.target_bloc.isin(["left"]+RIGHT)].copy(); article_code,_=pd.factorize(sub.article_id); outlet_code,outlets=pd.factorize(sub.outlet); n_articles=article_code.max()+1; n_outlets=len(outlets)
    left=sub.target_bloc.eq("left").to_numpy(); right=sub.target_bloc.isin(RIGHT).to_numpy(); signed=sub.signed.to_numpy(float); ideology_values=np.array([ideology[key(o)] for o in outlets]); rng=np.random.default_rng(seed); values=[]
    for _ in range(reps):
        count=np.bincount(rng.integers(0,n_articles,n_articles),minlength=n_articles).astype(float); wl=count[article_code[left]]; wr=count[article_code[right]]
        sum_left=np.bincount(outlet_code[left],weights=wl*signed[left],minlength=n_outlets); n_left=np.bincount(outlet_code[left],weights=wl,minlength=n_outlets)
        sum_right=np.bincount(outlet_code[right],weights=wr*signed[right],minlength=n_outlets); n_right=np.bincount(outlet_code[right],weights=wr,minlength=n_outlets); keep=(n_left>=20)&(n_right>=20)
        values.append(float(spearmanr(ideology_values[keep],sum_left[keep]/n_left[keep]-sum_right[keep]/n_right[keep]).statistic))
    return {"reps":reps,"seed":seed,"ci95":[float(v) for v in np.percentile(values,[2.5,97.5])],
            "resampled_unit":"article",
            "eligible_panel":("re-applied within each draw: an outlet enters a replicate only if that "
                              "replicate gives it at least 20 left-bloc and 20 right-bloc mentions, so "
                              "the panel can differ between draws")}


def fixed_panel_article_bootstrap(data,ideology,value="signed",reps=1000,seed=42,floor=20):
    """Article-resampling interval on a panel fixed once from the observed data.

    Companion to article_bootstrap_gradient, which re-applies the per-side floor inside every
    draw. Here the eligible outlets are decided once, on the observed sample, and every replicate
    is computed on exactly that panel; all mentions of a sampled article are kept. The interval is
    therefore conditional on the observed outlet panel and reflects only sampling of articles.
    """
    panel=list(outlet_asymmetry(data,value,ideology,floor).outlet)
    sub=data[data.outlet.isin(panel)&data.target_bloc.isin(["left"]+RIGHT)].copy()
    article_code,_=pd.factorize(sub.article_id); outlet_code,outlets=pd.factorize(sub.outlet)
    n_articles=article_code.max()+1; n_outlets=len(outlets)
    left=sub.target_bloc.eq("left").to_numpy(); right=sub.target_bloc.isin(RIGHT).to_numpy()
    values_array=sub[value].to_numpy(float); ideology_values=np.array([ideology[key(o)] for o in outlets])
    rng=np.random.default_rng(seed); draws=[]; degenerate=0
    for _ in range(reps):
        count=np.bincount(rng.integers(0,n_articles,n_articles),minlength=n_articles).astype(float)
        wl=count[article_code[left]]; wr=count[article_code[right]]
        sum_left=np.bincount(outlet_code[left],weights=wl*values_array[left],minlength=n_outlets)
        n_left=np.bincount(outlet_code[left],weights=wl,minlength=n_outlets)
        sum_right=np.bincount(outlet_code[right],weights=wr*values_array[right],minlength=n_outlets)
        n_right=np.bincount(outlet_code[right],weights=wr,minlength=n_outlets)
        usable=(n_left>0)&(n_right>0)
        if usable.sum()<3: degenerate+=1; continue
        asym=sum_left[usable]/n_left[usable]-sum_right[usable]/n_right[usable]
        draws.append(float(spearmanr(ideology_values[usable],asym).statistic))
    return {"reps":reps,"seed":seed,"resampled_unit":"article","n_panel_outlets":len(panel),
            "eligible_panel":("fixed once on the observed data at the per-side floor of "
                              f"{floor}; identical in every draw"),
            "replicates_used":len(draws),"replicates_degenerate":degenerate,
            "ci95":[float(v) for v in np.percentile(draws,[2.5,97.5])],
            "median":float(np.median(draws))}


def pooled_gap_inference(data):
    work=data[data.target_bloc.isin(["left"]+RIGHT)].copy(); x=np.column_stack([np.ones(len(work)),work.target_bloc.eq("left").astype(float)]); y=work.signed.to_numpy(float); inv=np.linalg.inv(x.T@x); beta=inv@x.T@y; residual=y-x@beta; meat=np.zeros((2,2))
    for _,indices in work.groupby("article_id").indices.items():
        score=x[indices].T@residual[indices]; meat+=np.outer(score,score)
    clusters=work.article_id.nunique(); factor=(clusters/(clusters-1))*((len(work)-1)/(len(work)-2)); covariance=factor*inv@meat@inv; se=float(np.sqrt(covariance[1,1])); statistic=float(beta[1]/se); p=float(2*student_t.sf(abs(statistic),clusters-1))
    return {"n_pairs":len(work),"n_article_clusters":clusters,"coefficient_left_minus_right":float(beta[1]),"cluster_robust_se":se,"cr1_factor":factor,"t":statistic,"degrees_of_freedom":clusters-1,"p_value":p}


def mp_weighted_gradient(data, ideology):
    relevant = data[data.target_bloc.isin(["left"] + RIGHT)]
    cells = relevant.groupby(["outlet", "target_bloc", "target_uid"], as_index=False).signed.mean()
    rows = []
    for outlet, group in relevant.groupby("outlet"):
        if (group.target_bloc == "left").sum() < 20 or group.target_bloc.isin(RIGHT).sum() < 20:
            continue
        subset = cells[cells.outlet == outlet]
        left = subset[subset.target_bloc == "left"].signed.mean()
        right = subset[subset.target_bloc.isin(RIGHT)].signed.mean()
        rows.append((ideology[key(outlet)], left - right))
    statistic, p_value = rho(*zip(*rows))
    return {"n_outlets": len(rows), "spearman_rho": statistic, "p_value": p_value}


def robustness(data, votes, ideology):
    primary_table = outlet_asymmetry(data, "signed", ideology)
    leave_one_out = []
    for outlet in primary_table.outlet:
        leave_one_out.append({"dropped": outlet, **gradient(data[data.outlet != outlet], "signed", ideology)})
    leave_one_out.sort(key=lambda row: row["spearman_rho"])
    dominant_uid = data.target_uid.value_counts().idxmax()
    capped = (data.sort_values(["target_uid", "outlet", "pair_id"], kind="stable")
              .groupby(["target_uid", "outlet"], sort=False).head(100))
    ordered = primary_table.sort_values("ideology").outlet.tolist()
    trimmed = data[~data.outlet.isin(ordered[:2] + ordered[-2:])]
    vote_data = data[["pair_id", "outlet", "target_bloc"]].merge(votes, on="pair_id", validate="one_to_one")
    single = {}
    for short, column in [("gpt_oss", "label_gpt_oss"), ("llama", "label_llama"), ("mistral", "label_mistral")]:
        vote_data[short] = vote_data[column].map(SIGNED)
        single[short] = gradient(vote_data, short, ideology)
    ablated = pd.read_csv(DATA / "frideo_scores_no_d3_d7.csv")
    ablated_ideology = {key(row.outlet): float(row.score) for row in ablated.itertuples()}
    primary=gradient(data,"signed",ideology); tau=kendalltau(primary_table.ideology,primary_table.asymmetry)
    primary.update({"kendall_tau":float(tau.statistic),"kendall_p_value":float(tau.pvalue),"fixed_panel_permutation":fixed_panel_permutation(primary_table),"article_clustered_bootstrap":article_bootstrap_gradient(data,ideology)})
    return {
        "primary": primary,
        "drop_most_covered_deputy": {"target_uid": dominant_uid,
            "removed_pairs": int((data.target_uid == dominant_uid).sum()),
            **gradient(data[data.target_uid != dominant_uid], "signed", ideology)},
        "cap_mp_outlet_at_100": {"retained_pairs": len(capped), "removed_pairs": len(data)-len(capped),
            **gradient(capped, "signed", ideology)},
        "mp_weighted": mp_weighted_gradient(data, ideology),
        "floor_30": gradient(data, "signed", ideology, 30),
        "trim_two_per_ideological_end": gradient(trimmed, "signed", ideology),
        "leave_one_out": leave_one_out,
        "leave_one_out_range": {"refits": len(leave_one_out),
            "rho_min": min(row["spearman_rho"] for row in leave_one_out),
            "rho_max": max(row["spearman_rho"] for row in leave_one_out)},
        "single_model_labels": single,
        "frideo_without_d3_d7": gradient(data, "signed", ablated_ideology),
    }


def bloc_contrasts(data, ideology):
    definitions = {"left_vs_right": (["left"], ["right"]),
                   "left_vs_far_right": (["left"], ["far_right"]),
                   "opposition_vs_centre": (["left", "right", "far_right"], ["centre", "centre_right"])}
    output = {}
    for name, (left_blocs, right_blocs) in definitions.items():
        rows = []
        for outlet, group in data.groupby("outlet"):
            left, right = group[group.target_bloc.isin(left_blocs)], group[group.target_bloc.isin(right_blocs)]
            if len(left) >= 20 and len(right) >= 20:
                rows.append((ideology[key(outlet)], left.signed.mean() - right.signed.mean()))
        statistic, p_value = rho(*zip(*rows))
        output[name] = {"n_outlets": len(rows), "spearman_rho": statistic, "p_value": p_value}
    return output


def two_fold_partition(relevant):
    """Kitagawa composition/rate split of the pooled left-minus-right gap at the symmetric reference."""
    def mix(side):
        subset = relevant[relevant.side == side]
        return (subset.role.value_counts(normalize=True).reindex(ROLES).fillna(0),
                subset.groupby("role").signed.mean().reindex(ROLES).fillna(0))
    w_left, p_left = mix("left"); w_right, p_right = mix("right")
    gap = float((w_left*p_left).sum() - (w_right*p_right).sum())
    selection = float(((w_left-w_right)*((p_left+p_right)/2)).sum())
    polarity = float((((w_left+w_right)/2)*(p_left-p_right)).sum())
    return w_left, p_left, w_right, p_right, gap, selection, polarity


def role_decomposition(data, ideology):
    clean = data[data["role(source/patient/agent)"].isin(ROLES)].copy()
    clean["role"] = clean["role(source/patient/agent)"]
    clean["side"] = np.where(clean.target_bloc == "left", "left",
                             np.where(clean.target_bloc.isin(RIGHT), "right", None))
    relevant = clean[clean.side.notna()].copy()
    global_weights = relevant.role.value_counts(normalize=True).reindex(ROLES).fillna(0)
    within_role = {role: gradient(relevant[relevant.role == role], "signed", ideology, 15) for role in ROLES}

    w_left, p_left, w_right, p_right, gap, selection, polarity = two_fold_partition(relevant)
    dominant = data.target_full_name.value_counts().idxmax()
    without = relevant[relevant.target_full_name != dominant]
    _, _, _, _, gap_wo, selection_wo, polarity_wo = two_fold_partition(without)
    fallback = relevant.groupby(["side", "role"]).signed.mean()
    rows = []
    for outlet, group in relevant.groupby("outlet"):
        left, right = group[group.side == "left"], group[group.side == "right"]
        if len(left) < 20 or len(right) < 20: continue
        nets = []
        for side, subset in [("left", left), ("right", right)]:
            observed = subset.groupby("role").signed.mean()
            nets.append(sum(global_weights[r]*observed.get(r, fallback.loc[(side,r)]) for r in ROLES))
        rows.append((ideology[key(outlet)], nets[0]-nets[1]))
    statistic, p_value = rho(*zip(*rows))
    return {
        "analysis_pairs": len(clean), "bloc_relevant_pairs": len(relevant),
        "global_role_mix": {r: float(global_weights[r]) for r in ROLES},
        "left_role_mix": {r: float(w_left[r]) for r in ROLES},
        "right_role_mix": {r: float(w_right[r]) for r in ROLES},
        "bloc_averaged_polarity_by_role": {r: float(((p_left+p_right)/2)[r]) for r in ROLES},
        "pooled_gap": gap, "selection_component": selection, "polarity_component": polarity,
        "selection_share": selection/gap, "polarity_share": polarity/gap,
        "within_role_gradients": within_role,
        "role_standardized_gradient": {"n_outlets": len(rows), "spearman_rho": statistic, "p_value": p_value},
        "excluding_executive_office": gradient(clean[~clean.gov_tenure], "signed", ideology),
        "excluding_most_covered_deputy": {"deputy": dominant, "bloc_relevant_pairs": len(without),
            "pooled_gap": gap_wo, "selection_component": selection_wo, "polarity_component": polarity_wo,
            "selection_share": selection_wo/gap_wo, "polarity_share": polarity_wo/gap_wo},
    }


def temporal_split_gradient(data, ideology):
    """The primary gradient in each half of the period, split at the pair-weighted median date."""
    dated = data.copy(); dated["parsed_date"] = pd.to_datetime(dated.date)
    midpoint = dated.parsed_date.median()
    early = dated[dated.parsed_date <= midpoint]; late = dated[dated.parsed_date > midpoint]
    return {"median_date": str(midpoint.date()),
            "early": {"pairs": len(early), **gradient(early, "signed", ideology)},
            "late": {"pairs": len(late), **gradient(late, "signed", ideology)}}


def within_cell_gradients(data, ideology):
    dated = data.copy(); dated["parsed_date"] = pd.to_datetime(dated.date)
    dated["month"] = dated.parsed_date.dt.to_period("M").astype(str)
    dated["week"] = dated.parsed_date.dt.to_period("W").astype(str)
    dated["day"] = dated.parsed_date.dt.date.astype(str)
    specs = {"deputy_month": ["target_uid","month"], "deputy_week": ["target_uid","week"],
             "deputy_day": ["target_uid","day"], "deputy_all_dates": ["target_uid"],
             "deputy_month_role": ["target_uid","month","role_bucket"]}
    output = {}
    for name, columns in specs.items():
        work = dated.copy(); work["cell"] = list(zip(*[work[c] for c in columns]))
        counts = work.groupby("cell").outlet.nunique()
        work = work[work.cell.isin(set(counts[counts >= 2].index))].copy()
        work["demeaned"] = work.signed - work.groupby("cell").signed.transform("mean")
        output[name] = {"retained_pairs": len(work), "retained_share": len(work)/len(dated),
                        "cells": work.cell.nunique(), **gradient(work, "demeaned", ideology)}
    return output


def frame_results(framed, ideology):
    headline = framed.drop_duplicates("article_id")
    prevalence, conditioned, gradients = [], [], []
    for frame in FRAMES:
        column = f"frame_{frame}"
        prevalence.append({"frame": frame, "pair_prevalence": float(framed[column].mean()),
                           "headline_prevalence": float(headline[column].mean())})
        present, absent = framed[framed[column] == 1], framed[framed[column] == 0]
        conditioned.append({"frame": frame, "n_present": len(present),
            "net_present": float(present.signed.mean()), "net_absent": float(absent.signed.mean()),
            "difference": float(present.signed.mean()-absent.signed.mean())})
        gradients.append({"frame": frame, **gradient(framed, column, ideology)})
    tone = outlet_asymmetry(framed, "signed", ideology)
    conflict = outlet_asymmetry(framed, "frame_conflict", ideology)
    coupled = tone[["outlet","asymmetry"]].rename(columns={"asymmetry":"tone"}).merge(
        conflict[["outlet","asymmetry"]].rename(columns={"asymmetry":"conflict"}), on="outlet")
    statistic, p_value = rho(coupled.tone, coupled.conflict)
    permutation={}
    for frame in ["conflict","morality"]:
        permutation[frame]=fixed_panel_permutation(outlet_asymmetry(framed,f"frame_{frame}",ideology))
    # A paired-outlet permutation for the tone/conflict coupling, so the reported p is not only
    # the parametric Spearman reference. Outlets are the exchangeable unit: one asymmetry pair
    # each, conflict asymmetries shuffled against tone asymmetries.
    coupling_permutation=fixed_panel_permutation(
        coupled.rename(columns={"tone":"ideology","conflict":"asymmetry"}), draws=1_000_000, seed=20260913)
    coupling_permutation["shuffled_unit"]="outlet (paired tone/conflict asymmetries)"
    frame_bootstrap={frame:fixed_panel_article_bootstrap(framed,ideology,f"frame_{frame}",seed=20260913+index)
                     for index,frame in enumerate(FRAMES)}
    return {"pairs": len(framed), "headlines": framed.article_id.nunique(), "prevalence": prevalence,
            "tone_conditioned_on_frame": conditioned, "outlet_gradients": gradients,
            "outlet_gradient_article_bootstrap": frame_bootstrap,
            "fixed_panel_permutation":permutation,
            "tone_conflict_coupling": {"n_outlets": len(coupled), "spearman_rho": statistic, "p_value": p_value,
              "test":("two-sided Spearman rank correlation between each outlet's tone asymmetry and its "
                      "conflict-prevalence asymmetry; p_value is SciPy's parametric reference against zero "
                      "rank correlation, NOT a permutation result"),
              "paired_outlet_permutation": coupling_permutation},
            "bloc_prevalence": records(framed.groupby("target_bloc")[[f"frame_{f}" for f in FRAMES]].mean().reset_index())}


def fit_beta(successes, totals):
    total = totals.sum(); mean = successes.sum()/total; rates = successes/totals
    observed = (totals*(rates-mean)**2).sum()/total
    sampling = mean*(1-mean)*(len(totals)/total)
    variance = max(1e-5, observed-sampling)
    kappa = max(1.0, mean*(1-mean)/variance-1.0)
    return {"alpha": mean*kappa, "kappa": kappa, "mean": mean}


def cross_outlet_spread(data, floor=16):
    cells = [(float(g.signed.mean()), outlet) for outlet,g in data.groupby("outlet") if len(g)>=floor]
    if len(cells) < 3: return None
    low, high = min(cells,key=lambda item:item[0]), max(cells,key=lambda item:item[0])
    return {"divisive_spread": high[0]-low[0], "divisive_lo": f"{low[0]:+.2f} {low[1]}",
            "divisive_hi": f"{high[0]:+.2f} {high[1]}"}


def reconstruct_leaderboards(framed):
    deputy_n = framed.groupby("target_uid").size()
    indicators = {"neg": framed["label(pos/neg/neu)"].eq("neg"),
                  "pos": framed["label(pos/neg/neu)"].eq("pos"),
                  "conflict": framed.frame_conflict.eq(1), "patient": framed.role_bucket.eq("patient_like")}
    priors = {name: fit_beta(values.groupby(framed.target_uid).sum().reindex(deputy_n.index).fillna(0), deputy_n)
              for name,values in indicators.items()}
    def shrunk(k,n,name):
        p=priors[name]; return float((k+p["alpha"])/(n+p["kappa"]))
    def base(data, unit, name):
        n=len(data); neg=int(data["label(pos/neg/neu)"].eq("neg").sum()); pos=int(data["label(pos/neg/neu)"].eq("pos").sum())
        spread=cross_outlet_spread(data) or {"divisive_spread":None,"divisive_lo":None,"divisive_hi":None}
        return {unit:name,"n":n,"net":float(data.signed.mean()),"shrunk_net":shrunk(pos,n,"pos")-shrunk(neg,n,"neg"),
                "targeted_share":float(data.role_bucket.eq("patient_like").mean()),
                "targeted_shrunk":shrunk(int(data.role_bucket.eq("patient_like").sum()),n,"patient"),
                "conflict_prev":float(data.frame_conflict.mean()),
                "conflict_shrunk":shrunk(int(data.frame_conflict.eq(1).sum()),n,"conflict"),**spread}
    mp_rows=[]
    for uid,data in framed.groupby("target_uid"):
        if len(data)<30: continue
        row=base(data,"uid",uid); row.update(name=data.target_full_name.iloc[0],group=data.target_group.iloc[0],bloc=data.target_bloc.iloc[0]); mp_rows.append(row)
    mp_cols=["uid","name","group","bloc","n","net","shrunk_net","targeted_share","targeted_shrunk","conflict_prev","conflict_shrunk","divisive_spread","divisive_lo","divisive_hi"]
    mp=pd.DataFrame(mp_rows)[mp_cols].sort_values("uid").reset_index(drop=True)
    group_rows=[]
    for group,data in framed.groupby("target_group"):
        if pd.isna(group) or str(group) in {"nan", ""}: continue
        row=base(data,"group",group); row["targeted_net"]=float(data.loc[data.role_bucket=="patient_like","signed"].mean()); row["responsibility_prev"]=float(data.frame_responsibility.mean()); group_rows.append(row)
    group_cols=["group","n","net","shrunk_net","targeted_share","targeted_shrunk","targeted_net","conflict_prev","conflict_shrunk","responsibility_prev","divisive_spread","divisive_lo","divisive_hi"]
    all_groups=pd.DataFrame(group_rows)[group_cols].sort_values("group").reset_index(drop=True)
    return mp,all_groups[all_groups.group!="NI"].reset_index(drop=True),all_groups[all_groups.group=="NI"].reset_index(drop=True),{n:{"mean":float(p["mean"]),"kappa":float(p["kappa"])} for n,p in priors.items()}


def table_comparison(rebuilt, shipped_path, keys):
    shipped=pd.read_csv(shipped_path)
    shipped=shipped[list(rebuilt.columns)].sort_values(keys).reset_index(drop=True)
    rebuilt=rebuilt.sort_values(keys).reset_index(drop=True); failures=[]
    for column in shipped.columns:
        if pd.api.types.is_numeric_dtype(shipped[column]):
            left = pd.to_numeric(shipped[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(rebuilt[column], errors="coerce").to_numpy(float)
            if not np.allclose(left,right,rtol=1e-10,atol=1e-12,equal_nan=True): failures.append(column)
        elif not shipped[column].fillna("").astype(str).equals(rebuilt[column].fillna("").astype(str)): failures.append(column)
    return {"rows":len(rebuilt),"matching":not failures,"mismatched_columns":failures}


def within_group_vectors(data,x,y):
    xs,ys=[],[]
    for _,group in data.groupby("group",sort=True):
        n=len(group); xr=rankdata(group[x].to_numpy(),method="average")/n; yr=rankdata(group[y].to_numpy(),method="average")/n
        xs.append(xr-xr.mean()); ys.append(yr-yr.mean())
    return np.concatenate(xs),np.concatenate(ys)


def h3_mp_table(data,votes,gov_override=None):
    mp=(data.groupby("target_uid").agg(n=("pair_id","size"),net=("signed","mean"),conflict=("frame_conflict","mean"),group=("target_group","first"),name=("target_full_name","first"),gov=("gov_tenure","max")).reset_index().merge(votes[["target_uid","alignment_pct","compared_votes"]],on="target_uid",how="left"))
    if gov_override is not None: mp["gov"]=mp.target_uid.map(gov_override).fillna(False)
    return mp[(mp.group!="NI")&(~mp.gov)&(mp.compared_votes>=50)&(mp.n>=10)].dropna(subset=["alignment_pct"]).copy()


def h3_uncertainty(data,y,seed=20260831):
    groups=[g.copy() for _,g in data.groupby("group",sort=True)]; rng=np.random.default_rng(seed); bootstrap=[]
    for _ in range(4000):
        sample=pd.concat([g.iloc[rng.integers(0,len(g),len(g))] for g in groups],ignore_index=True); x,v=within_group_vectors(sample,"alignment_pct",y); bootstrap.append(float(pearsonr(x,v).statistic))
    x,v=within_group_vectors(data,"alignment_pct",y); observed=abs(float(np.corrcoef(x,v)[0,1])); arrays=[]
    for group in groups: arrays.append(within_group_vectors(group.assign(group="one"),"alignment_pct",y))
    rng=np.random.default_rng(seed); exceed=0
    for _ in range(20000):
        permuted=np.corrcoef(np.concatenate([a[0] for a in arrays]),np.concatenate([rng.permutation(a[1]) for a in arrays]))[0,1]; exceed += abs(permuted)>=observed-PERMUTATION_EPSILON*max(1.0,observed)
    return [float(v) for v in np.percentile(bootstrap,[2.5,97.5])],(exceed+1)/20001


def h3_results(framed,votes):
    mp=h3_mp_table(framed,votes); output={"n_deputies":len(mp),"outcomes":{},"party_estimates":[],"temporal_split":{}}
    for outcome in ["conflict","net"]:
        x,y=within_group_vectors(mp,"alignment_pct",outcome); estimate=float(pearsonr(x,y).statistic); ci,p=h3_uncertainty(mp,outcome)
        output["outcomes"][outcome]={"normalized_within_party_rank_r":estimate,"bootstrap_ci95":ci,"permutation_p_value":p}
    for group,subset in mp.groupby("group"):
        if len(subset)>=4:
            statistic,p=rho(subset.alignment_pct,subset.conflict); output["party_estimates"].append({"group":group,"n":len(subset),"spearman_rho":statistic,"p_value":p})
    dated=framed.copy(); dated["parsed_date"]=pd.to_datetime(dated.date); midpoint=dated.parsed_date.median(); full_gov=dated.groupby("target_uid").gov_tenure.max()
    for label,subset,seed in [("early",dated[dated.parsed_date<=midpoint],20260832),("late",dated[dated.parsed_date>midpoint],20260833)]:
        temporal=h3_mp_table(subset,votes,full_gov); x,y=within_group_vectors(temporal,"alignment_pct","conflict"); ci,p=h3_uncertainty(temporal,"conflict",seed)
        output["temporal_split"][label]={"n_deputies":len(temporal),"normalized_within_party_rank_r":float(pearsonr(x,y).statistic),"bootstrap_ci95":ci,"permutation_p_value":p}
    output["temporal_split"]["median_date"]=str(midpoint.date())
    return output,mp


def beta_binomial_terms(theta,matrix,successes,totals):
    mean=np.clip(expit(matrix@theta[:-1]),1e-10,1-1e-10); concentration=np.exp(theta[-1])
    return mean,concentration,mean*concentration,(1-mean)*concentration


def beta_binomial_nll(theta,matrix,successes,totals):
    mean,concentration,alpha,beta=beta_binomial_terms(theta,matrix,successes,totals)
    ll=gammaln(totals+1)-gammaln(successes+1)-gammaln(totals-successes+1)+betaln(successes+alpha,totals-successes+beta)-betaln(alpha,beta)
    return float(-ll.sum())


def gradient_hessian(theta,matrix,successes,totals,step=1e-6):
    """Hessian by central differences of the exact gradient."""
    size=len(theta); hessian=np.zeros((size,size))
    for i in range(size):
        offset=np.zeros(size); offset[i]=step
        _,high=beta_binomial_objective(theta+offset,matrix,successes,totals); _,low=beta_binomial_objective(theta-offset,matrix,successes,totals)
        hessian[i]=(high-low)/(2*step)
    return (hessian+hessian.T)/2


def beta_binomial_objective(theta,matrix,successes,totals):
    """Negative log-likelihood with its exact gradient.

    Supplying the analytic gradient instead of letting the optimiser form
    finite differences is what makes the fit converge to the same point on
    every BLAS implementation: the finite-difference step size interacts with
    platform-specific rounding in the matrix products, and on a likelihood this
    flat that moved the bootstrap percentiles in the second decimal.
    """
    mean,concentration,alpha,beta=beta_binomial_terms(theta,matrix,successes,totals)
    value=beta_binomial_nll(theta,matrix,successes,totals)
    shared=digamma(totals+concentration)-digamma(concentration)
    d_alpha=digamma(successes+alpha)-shared-digamma(alpha); d_beta=digamma(totals-successes+beta)-shared-digamma(beta)
    slope=(d_alpha-d_beta)*concentration*mean*(1-mean)
    return value,np.r_[-(matrix*slope[:,None]).sum(axis=0),-float((d_alpha*alpha+d_beta*beta).sum())]


def fit_beta_binomial_model(matrix,successes,totals,start=None):
    """Maximum-likelihood beta-binomial fit.

    The fit always starts from the same neutral point, never from a
    caller-supplied warm start, so that the optimiser follows one trajectory
    that does not depend on the order in which replicates were fitted. With the
    exact gradient and a gradient-norm stop, L-BFGS-B converges to the same
    optimum on every platform we have tested; a derivative-free polish is kept
    only as a fallback for the rare replicate it cannot close.
    """
    neutral=np.r_[np.zeros(matrix.shape[1]),np.log(20.0)]
    result=minimize(beta_binomial_objective,neutral,args=(matrix,successes,totals),jac=True,method="L-BFGS-B",options={"maxiter":10000,"maxfun":10000,"ftol":1e-15,"gtol":1e-10})
    if result.success and np.max(np.abs(result.jac))<1e-6: return result.x,result
    polish=minimize(beta_binomial_nll,result.x,args=(matrix,successes,totals),method="Nelder-Mead",options={"maxiter":50000,"maxfev":50000,"xatol":1e-12,"fatol":1e-12})
    best=polish if polish.fun<result.fun else result
    if not np.isfinite(best.fun): raise RuntimeError("beta-binomial optimization failed")
    return best.x,best


def h3_precision_models(framed,alignment,h3_points):
    deputies=h3_points.reset_index(drop=True).copy(); deputies["align_rank"]=deputies.groupby("group").alignment_pct.rank(method="average",pct=True); deputies["align_rank_c"]=deputies.align_rank-deputies.groupby("group").align_rank.transform("mean"); rank_map=deputies.set_index("target_uid").align_rank_c
    pair_data=framed[framed.target_uid.isin(rank_map.index)].copy(); pair_data["align_rank_c"]=pair_data.target_uid.map(rank_map); pair_data["conflict"]=pair_data.frame_conflict.astype(int)
    successes=pair_data.groupby("target_uid").conflict.sum().reindex(deputies.target_uid).to_numpy(int); totals=pair_data.groupby("target_uid").size().reindex(deputies.target_uid).to_numpy(int); dummy=pd.get_dummies(deputies.group,drop_first=True,dtype=float); matrix=np.column_stack([np.ones(len(deputies)),deputies.align_rank_c,dummy])
    fitted,result=fit_beta_binomial_model(matrix,successes,totals); hessian=gradient_hessian(fitted,matrix,successes,totals); covariance=np.linalg.inv(hessian); se=float(np.sqrt(covariance[1,1])); coefficient=float(fitted[1]); low,high=coefficient-1.96*se,coefficient+1.96*se
    if fitted.shape != H3_BOOTSTRAP_GENERATING_FIT.shape:
        raise RuntimeError("registered H3 bootstrap generating fit has the wrong shape")
    fit_difference=float(np.max(np.abs(fitted-H3_BOOTSTRAP_GENERATING_FIT)))
    if fit_difference > H3_BOOTSTRAP_FIT_TOLERANCE:
        raise RuntimeError(f"H3 fit differs from the registered bootstrap generating fit by {fit_difference:.3g}")
    rng=np.random.default_rng(20260831); boot=[]; failures=0
    mean=expit(matrix@H3_BOOTSTRAP_GENERATING_FIT[:-1]); concentration=np.exp(H3_BOOTSTRAP_GENERATING_FIT[-1])
    for _ in range(1000):
        probability=rng.beta(mean*concentration,(1-mean)*concentration); simulated=rng.binomial(totals,probability)
        try: theta,_=fit_beta_binomial_model(matrix,simulated,totals); boot.append(float(theta[1]))
        except RuntimeError: failures+=1
    loo=[]
    for index in range(len(totals)):
        keep=np.arange(len(totals))!=index; reduced=matrix[keep]; active=np.r_[np.ptp(reduced,axis=0)>0,True]; active[0]=True; theta,_=fit_beta_binomial_model(reduced[:,active[:-1]],successes[keep],totals[keep]); loo.append(float(theta[1]))
    glm=smf.glm("conflict ~ align_rank_c + C(target_group)",pair_data,family=sm.families.Binomial()).fit(); mp_codes=pd.Categorical(pair_data.target_uid).codes; article_codes=pd.Categorical(pair_data.article_id).codes; two_way,_,_=cov_cluster_2groups(glm,mp_codes,article_codes); position=glm.params.index.get_loc("align_rank_c"); glm_coefficient=float(glm.params.iloc[position]); glm_se=float(np.sqrt(two_way[position,position])); glm_low,glm_high=glm_coefficient-1.96*glm_se,glm_coefficient+1.96*glm_se
    return {"design":{"n_mps":len(deputies),"n_pairs":len(pair_data),"n_articles":pair_data.article_id.nunique(),"mentions_min":int(totals.min()),"mentions_median":float(np.median(totals)),"mentions_max":int(totals.max())},
      "beta_binomial":{"coefficient_log_odds":coefficient,"standard_error":se,"wald_p":float(2*norm.sf(abs(coefficient/se))),"coefficient_ci95":[low,high],"odds_ratio":float(np.exp(coefficient)),"odds_ratio_ci95":[float(np.exp(low)),float(np.exp(high))],"concentration":float(np.exp(fitted[-1])),"log_likelihood":float(-result.fun)},
      "parametric_bootstrap":{"reps_requested":1000,"reps_completed":len(boot),"failures":failures,"seed":20260831,"coefficient_ci95":[float(v) for v in np.percentile(boot,[2.5,97.5])],"coefficient_median":float(np.median(boot)),"share_positive":float(np.mean(np.asarray(boot)>0))},
      "leave_one_mp_out":{"refits":len(loo),"coefficient_range":[float(min(loo)),float(max(loo))],"all_positive":bool(min(loo)>0)},
      "headline_binomial_glm":{"coefficient_log_odds":glm_coefficient,"odds_ratio":float(np.exp(glm_coefficient)),"standard_error":glm_se,"p_value":float(2*norm.sf(abs(glm_coefficient/glm_se))),"coefficient_ci95":[glm_low,glm_high],"odds_ratio_ci95":[float(np.exp(glm_low)),float(np.exp(glm_high))]}}


def tie_sensitivity(data,votes,ideology):
    label_cols=["label_gpt_oss","label_mistral","label_llama"]; role_cols=["role_gpt_oss","role_mistral","role_llama"]
    label_ties=votes[label_cols].nunique(axis=1).eq(3); valid_roles=votes[role_cols].where(votes[role_cols].isin(ROLES))
    role_ties=valid_roles.nunique(axis=1).gt(1)&valid_roles.apply(lambda row:row.value_counts().max(),axis=1).lt(2); off_vocab=(~votes[role_cols].isin(ROLES)).any(axis=1)
    merged=data.merge(votes,on="pair_id",validate="one_to_one"); merged["probabilistic_signed"]=merged[label_cols].apply(lambda column:column.map(SIGNED)).mean(axis=1); merged["label_tie"]=label_ties.to_numpy(); merged["role_tie"]=role_ties.to_numpy()
    def role_probs(frame):
        p=pd.DataFrame(0.0,index=frame.index,columns=ROLES)
        for c in role_cols:
            for role in ROLES:p[role]+=frame[c].eq(role).astype(float)
        return p.div(p.sum(axis=1),axis=0)
    def standardized(frame,probs):
        mask=frame.target_bloc.isin(["left"]+RIGHT); work=frame.loc[mask].copy().reset_index(drop=True); probs=probs.loc[mask].reset_index(drop=True); work["side"]=np.where(work.target_bloc=="left","left","right"); weights=probs.sum()/probs.to_numpy().sum()
        def weighted(indices,role):
            w=probs.loc[indices,role].to_numpy(float); v=work.loc[indices,"signed"].to_numpy(float); return float(np.sum(w*v)/np.sum(w)) if w.sum() else np.nan
        fallback={s:{r:weighted(work.index[work.side==s],r) for r in ROLES} for s in ["left","right"]}; rows=[]
        for outlet,subset in work.groupby("outlet"):
            left,right=subset.index[subset.side=="left"],subset.index[subset.side=="right"]
            if len(left)<20 or len(right)<20:continue
            nets=[]
            for side,indices in [("left",left),("right",right)]:
                vals=[]
                for role in ROLES:
                    value=weighted(indices,role); vals.append(weights[role]*(fallback[side][role] if np.isnan(value) else value))
                nets.append(sum(vals))
            rows.append((ideology[key(outlet)],nets[0]-nets[1]))
        statistic,p=rho(*zip(*rows)); return {"n_outlets":len(rows),"spearman_rho":statistic,"p_value":p}
    majority_probs=pd.DataFrame({r:merged["role(source/patient/agent)"].eq(r).astype(float) for r in ROLES}); keep=merged[~merged.role_tie].copy().reset_index(drop=True); drop_probs=pd.DataFrame({r:keep["role(source/patient/agent)"].eq(r).astype(float) for r in ROLES})
    return {"sentiment_three_way_ties":int(label_ties.sum()),"role_no_majority":int(role_ties.sum()),"role_units_with_offvocab":int(off_vocab.sum()),
            "tone_gradients":{"baseline":gradient(merged,"signed",ideology),"drop_sentiment_ties":gradient(merged[~merged.label_tie],"signed",ideology),"probabilistic_votes":gradient(merged,"probabilistic_signed",ideology)},
            "role_standardized_gradients":{"baseline":standardized(merged,majority_probs),"drop_role_ties":standardized(keep,drop_probs),"fractional_votes":standardized(merged,role_probs(merged))}}


def inter_model_agreement(votes):
    labels=votes[["label_gpt_oss","label_mistral","label_llama"]]; codes=labels.apply(lambda column:column.map({"neg":0.0,"neu":1.0,"pos":2.0})).to_numpy(float)
    roles=votes[["role_gpt_oss","role_mistral","role_llama"]].where(lambda f:f.isin(ROLES))
    return {"sentiment":{"n_units":len(labels),"ordinal_alpha":krippendorff_alpha(codes,"ordinal"),"nominal_alpha":krippendorff_alpha(codes),"three_way_agreement":float(labels.nunique(axis=1).eq(1).mean()),"marginals":{c.removeprefix("label_"):labels[c].value_counts().sort_index().to_dict() for c in labels}},
            "role":{"n_units":len(roles),"complete_cases":int(roles.notna().all(axis=1).sum()),"nominal_alpha":krippendorff_alpha(roles.to_numpy(object)),"three_way_agreement_complete_cases":float(roles.dropna().nunique(axis=1).eq(1).mean()),"marginals":{c.removeprefix("role_"):roles[c].value_counts().sort_index().to_dict() for c in roles}}}


def complete_nominal_alpha(matrix):
    """Krippendorff nominal alpha for a complete units x raters matrix (vectorised)."""
    units,raters=matrix.shape; categories=np.unique(matrix)
    counts=np.stack([(matrix==category).sum(axis=1) for category in categories],axis=1).astype(float); total=units*raters
    observed=float((raters*raters-(counts**2).sum(axis=1)).sum())/(raters-1); marginals=counts.sum(axis=0)
    expected=float(total*total-(marginals**2).sum())/(total-1)
    return float(1-observed/expected)


def fleiss_kappa(matrix,categories):
    units,raters=matrix.shape; counts=np.stack([(matrix==category).sum(axis=1) for category in categories],axis=1).astype(float)
    agreement=((counts**2).sum(axis=1)-raters)/(raters*(raters-1)); shares=counts.sum(axis=0)/(units*raters)
    return float((agreement.mean()-(shares**2).sum())/(1-(shares**2).sum()))


def cohen_kappa(first,second):
    categories=sorted(set(first)|set(second)); index={c:i for i,c in enumerate(categories)}; table=np.zeros((len(categories),len(categories)))
    for a,b in zip(first,second): table[index[a],index[b]]+=1.0
    observed=float(np.trace(table))/len(first); expected=float((table.sum(axis=0)*table.sum(axis=1)).sum())/len(first)**2
    return float((observed-expected)/(1-expected))


def pairwise_kappas(matrix,raters):
    return {f"{a}|{b}":cohen_kappa(matrix[:,i],matrix[:,j]) for i,a in enumerate(raters) for j,b in enumerate(raters) if i<j}


def alpha_bootstrap_ci(matrix,reps=1000,seed=42):
    rng=np.random.default_rng(seed); units=len(matrix)
    draws=[complete_nominal_alpha(matrix[rng.integers(0,units,units)]) for _ in range(reps)]
    return [float(v) for v in np.percentile(draws,[2.5,97.5])]


def frame_inter_model_agreement(framed):
    """Recompute frame agreement from the released per-model frame votes."""
    raters=["gpt_oss","mistral","llama"]; votes=pd.read_csv(DATA/"frame_rater_votes.csv",dtype={"article_id":str})
    binary=[f"frame_{frame}_{rater}" for frame in FRAMES for rater in raters]; dominant=[f"dominant_{rater}" for rater in raters]
    incomplete=votes[binary+dominant].isna().any(axis=1); complete=votes[~incomplete].reset_index(drop=True)
    supplied=json.loads((DATA/"model_iaa_frames.json").read_text()); deviations=[]
    per_frame={}; majority={}
    for frame in FRAMES:
        matrix=complete[[f"frame_{frame}_{rater}" for rater in raters]].to_numpy(int)
        majority[frame]=(matrix.sum(axis=1)>=2).astype(int); unanimous=int((matrix.min(axis=1)==matrix.max(axis=1)).sum())
        alpha=complete_nominal_alpha(matrix); reference=supplied["per_frame"][frame]
        deviations.append(abs(alpha-reference["krippendorff_alpha_nominal"]))
        per_frame[frame]={"krippendorff_alpha_nominal":alpha,"alpha_nominal_ci95_own_bootstrap":alpha_bootstrap_ci(matrix),
            "fleiss_kappa":fleiss_kappa(matrix,[0,1]),"pairwise_cohen_unweighted":pairwise_kappas(matrix,raters),
            "raw_3way_agreement":float(unanimous/len(complete)),"unanimous_3_0":unanimous,"split_2_1":len(complete)-unanimous,
            "prevalence_per_rater":{rater:float(complete[f"frame_{frame}_{rater}"].mean()) for rater in raters},
            "prevalence_majority":float(majority[frame].mean())}
    labels=complete[dominant].to_numpy(object); categories=FRAMES+["none"]
    counted=np.array([np.unique(row,return_counts=True) for row in labels],dtype=object)
    dominant_majority=np.array([values[counts.argmax()] if counts.max()>=2 else None for values,counts in counted],dtype=object)
    deviations.append(abs(complete_nominal_alpha(labels)-supplied["dominant"]["krippendorff_alpha_nominal"]))
    # The frame annotators ran over the whole headline universe, and their agreement is
    # reported over it. The analytic set is a strict subset of that universe, so the
    # majority-vote reconstruction is checked on the articles that actually carry a
    # released mention, and the shortfall is reported.
    released_all=framed.drop_duplicates("article_id").set_index("article_id")
    in_release=complete.article_id.isin(released_all.index).to_numpy()
    released=released_all.loc[complete.article_id[in_release]]
    released_dominant=released.dominant.to_numpy(object)
    frame_columns_match={frame:int((released[f"frame_{frame}"].to_numpy(int)==majority[frame][in_release]).sum()) for frame in FRAMES}
    resolved_all=dominant_majority!=None  # noqa: E711 - object array comparison
    resolved=resolved_all[in_release]
    dominant_majority_released=dominant_majority[in_release]
    inconsistent=sum(1 for i,label in enumerate(released_dominant) if label!="none" and majority[label][in_release][i]==0)
    all_zero=np.sum([majority[frame][in_release] for frame in FRAMES],axis=0)==0
    return {
      "alignment":{"n_units_per_rater":{rater:int(votes[f"dominant_{rater}"].notna().sum()) for rater in raters},"n_units_joined":len(votes),
        "n_dropped_incomplete":int(incomplete.sum()),"dropped_article_ids":sorted(votes.loc[incomplete,"article_id"]),"n_units":len(complete)},
      "per_frame":per_frame,
      "dominant":{"categories":categories,"krippendorff_alpha_nominal":complete_nominal_alpha(labels),"fleiss_kappa":fleiss_kappa(labels,np.array(categories,dtype=object)),
        "pairwise_cohen_unweighted":pairwise_kappas(labels,raters),"raw_3way_agreement":float(np.mean([len(set(row))==1 for row in labels])),
        "n_two_of_three_majority":int(resolved.sum()),"n_all_three_differ":int((~resolved).sum()),
        "per_rater_marginals":{rater:complete[f"dominant_{rater}"].value_counts().reindex(categories).astype(int).to_dict() for rater in raters}},
      "majority_vote_reconstruction":{"n_articles":int(in_release.sum()),
        "n_articles_annotated":len(complete),
        "n_annotated_articles_outside_analytic_set":int((~in_release).sum()),
        "released_frame_columns_matched":frame_columns_match,
        "all_released_frame_columns_reproduced":all(value==int(in_release.sum()) for value in frame_columns_match.values()),
        "released_dominant_matched_where_majority_exists":int((released_dominant[resolved]==dominant_majority_released[resolved]).sum()),
        "released_dominant_disagreements_where_majority_exists":int((released_dominant[resolved]!=dominant_majority_released[resolved]).sum()),
        "n_dominant_ties_resolved_outside_these_votes":int((~resolved_all).sum()),
        "n_dominant_inconsistent_with_majority_vector":int(inconsistent),
        "n_dominant_set_but_all_frames_zero":int(((released_dominant!="none")&all_zero).sum()),
        "majority_prevalence":{frame:float(majority[frame].mean()) for frame in FRAMES},
        "majority_prevalence_in_analytic_set":{frame:float(majority[frame][in_release].mean()) for frame in FRAMES},
        "majority_dominant_distribution":pd.Series(released_dominant).value_counts().reindex(categories).astype(int).to_dict()},
      "max_abs_deviation_from_supplied_record":float(max(deviations))}


def confusion_metrics(matrix,labels):
    precision=[]; recall=[]; f1=[]; per_class={}
    for i,label in enumerate(labels):
        p=matrix[i,i]/matrix[:,i].sum() if matrix[:,i].sum() else np.nan; r=matrix[i,i]/matrix[i,:].sum() if matrix[i,:].sum() else np.nan; score=2*p*r/(p+r) if p+r else np.nan
        precision.append(p); recall.append(r); f1.append(score); per_class[label]={"precision":float(p),"recall":float(r),"f1":float(score)}
    return {"confusion_human_rows_model_columns":matrix.tolist(),"accuracy":float(np.trace(matrix)/matrix.sum()),"macro_precision":float(np.nanmean(precision)),"macro_recall":float(np.nanmean(recall)),"macro_f1":float(np.nanmean(f1)),"per_class":per_class}


def soft_confusion(humans,model,weights,labels):
    index={label:i for i,label in enumerate(labels)}; matrix=np.zeros((len(labels),len(labels)))
    for pair_id in model.index:
        m=index[str(model.loc[pair_id]).strip().lower()]
        for human in humans: matrix[index[str(human.loc[pair_id]).strip().lower()],m]+=float(weights.loc[pair_id])/len(humans)
    return matrix


def human_analysis(pairs):
    coded=pd.read_csv(DATA/"human_labels.csv",dtype={"pair_id":str}).set_index("pair_id"); population=pairs.drop_duplicates("pair_id").set_index("pair_id"); sample=population.loc[coded.index].copy(); sample["stratum"]=sample["label(pos/neg/neu)"].astype(str)+" | "+sample.dominant.astype(str); population_strata=pairs.drop_duplicates("pair_id").assign(stratum=lambda d:d["label(pos/neg/neu)"].astype(str)+" | "+d.dominant.astype(str)).stratum.value_counts(); sample_counts=sample.stratum.value_counts(); weights=sample.stratum.map(population_strata/sample_counts); weights=weights/weights.mean()
    tasks={"sentiment":("label(pos/neg/neu)","human_label(neg/neu/pos)",["neg","neu","pos"],"ordinal"),"role":("role(source/patient/agent)","human_role(source/patient/agent)",["source","patient","agent"],"nominal")}
    for frame in FRAMES: tasks[f"frame_{frame}"]=(f"frame_{frame}",f"human_frame_{frame}(0/1)",["0","1"],"nominal")
    output={"n_units":len(coded),"weight_range":[float(weights.min()),float(weights.max())],"tasks":{}}
    for name,(model_col,human_base,labels,metric) in tasks.items():
        model=sample[model_col].astype(int).astype(str) if name.startswith("frame_") else sample[model_col].astype(str).str.lower(); humans=[coded[f"{human_base}_coder{i}"].astype(int).astype(str) if name.startswith("frame_") else coded[f"{human_base}_coder{i}"].astype(str).str.lower() for i in [1,2]]
        matrix=soft_confusion(humans,model,weights,labels); result=confusion_metrics(matrix,labels); agreed=humans[0]==humans[1]; result["n_human_agreed"]=int(agreed.sum()); result["agreed_only_accuracy"]=float((model[agreed]==humans[0][agreed]).mean()); human_codes=np.column_stack([[labels.index(v) for v in human] for human in humans]); result["human_alpha"]=krippendorff_alpha(human_codes,metric)
        if name=="sentiment":
            agreed_codes=np.column_stack([[labels.index(v) for v in humans[0][agreed]],[labels.index(v) for v in model[agreed]]]); result["agreed_only_model_human_alpha"]=krippendorff_alpha(agreed_codes,"ordinal")
        output["tasks"][name]=result
    score={"neg":-1.0,"neu":0.0,"pos":1.0}; sample["side"]=np.where(sample.target_bloc=="left","left",np.where(sample.target_bloc.isin(RIGHT),"right","other")); model_sent=sample["label(pos/neg/neu)"].map(score); human_sent=(coded["human_label(neg/neu/pos)_coder1"].map(score)+coded["human_label(neg/neu/pos)_coder2"].map(score))/2; sample["sentiment_error"]=model_sent-human_sent; side={}
    for name,subset in sample.groupby("side"):
        w=weights.loc[subset.index]; side[name]={"n":len(subset),"model_minus_human_tone_bias":float(np.average(subset.sentiment_error,weights=w))}
    difference=side["left"]["model_minus_human_tone_bias"]-side["right"]["model_minus_human_tone_bias"]; rng=np.random.default_rng(20260831); cells=[cell.index.to_numpy() for _,cell in sample.groupby("stratum",sort=True)]; boot=[]
    for _ in range(4000):
        indices=np.concatenate([rng.choice(cell,size=len(cell),replace=True) for cell in cells]); draw=sample.loc[indices]; draw_weights=weights.loc[indices]; left=draw.side.eq("left"); right=draw.side.eq("right"); boot.append(float(np.average(draw.loc[left,"sentiment_error"],weights=draw_weights[left])-np.average(draw.loc[right,"sentiment_error"],weights=draw_weights[right])))
    output["group_context"]={"by_side":side,"left_minus_right_tone_bias":difference,"bootstrap_reps":4000,"seed":20260831,"left_minus_right_ci95":[float(v) for v in np.quantile(boot,[0.025,0.975])]}
    return output


def prompt_sensitivity(data,ideology,votes=None,reps=1000,seed=20260913):
    labels=pd.read_csv(DATA/"prompt_paraphrase_labels.csv"); merged=data[["pair_id","article_id","outlet","target_bloc","signed","label(pos/neg/neu)"]].merge(labels,on="pair_id",validate="one_to_one"); merged["paraphrased_signed"]=merged.paraphrased_label.map(SIGNED)
    merged["changed"]=merged["label(pos/neg/neu)"].ne(merged.paraphrased_label)
    frozen=gradient(merged,"signed",ideology); paraphrased=gradient(merged,"paraphrased_signed",ideology)
    # Article-clustered bootstrap on the label-change rate, and the paired difference between the
    # two gradients on the same resampled articles, so the attenuation carries an interval.
    article_code,_=pd.factorize(merged.article_id); n_articles=article_code.max()+1
    rng=np.random.default_rng(seed); change=[]; difference=[]
    index_by_article=[np.flatnonzero(article_code==a) for a in range(n_articles)]
    for _ in range(reps):
        picked=np.concatenate([index_by_article[a] for a in rng.integers(0,n_articles,n_articles)])
        draw=merged.iloc[picked]
        change.append(float(draw.changed.mean()))
        try:
            a=gradient(draw,"signed",ideology)["spearman_rho"]; b=gradient(draw,"paraphrased_signed",ideology)["spearman_rho"]
            difference.append(float(a-b))
        except Exception:
            pass
    output={"pairs":len(merged),"label_change_rate":float(merged.changed.mean()),
      "label_change_rate_ci95":[float(v) for v in np.percentile(change,[2.5,97.5])],
      "bootstrap":{"reps":reps,"seed":seed,"resampled_unit":"article","replicates_used":len(difference)},
      "frozen_gradient":frozen,"paraphrased_gradient":paraphrased,
      "frozen_minus_paraphrased_rho":frozen["spearman_rho"]-paraphrased["spearman_rho"],
      "frozen_minus_paraphrased_rho_ci95":[float(v) for v in np.percentile(difference,[2.5,97.5])],
      "frozen_net":float(merged.signed.mean()),"paraphrased_net":float(merged.paraphrased_signed.mean())}
    if votes is not None:
        # the closer same-model control: the frozen gpt-oss labels on exactly these rows
        same=merged.merge(votes[["pair_id","label_gpt_oss"]],on="pair_id",validate="one_to_one")
        same["gpt_oss_signed"]=same.label_gpt_oss.map(SIGNED)
        output["frozen_gpt_oss_same_rows_gradient"]=gradient(same,"gpt_oss_signed",ideology)
    return output


def group_field_ablation(ideology):
    paired=pd.read_csv(DATA/"group_field_ablation.csv"); paired["with_signed"]=paired.label_with_group.map(SIGNED); paired["without_signed"]=paired.label_without_group.map(SIGNED); paired["changed"]=paired.label_with_group!=paired.label_without_group
    left=paired[paired.target_bloc=="left"]; right=paired[paired.target_bloc.isin(RIGHT)]; rng=np.random.default_rng(20260912); change_boot=[]; gap_boot=[]
    for _ in range(4000):
        li=rng.integers(0,len(left),len(left)); ri=rng.integers(0,len(right),len(right)); change_boot.append(left.iloc[li].changed.mean()-right.iloc[ri].changed.mean())
    for _ in range(4000):
        li=rng.integers(0,len(left),len(left)); ri=rng.integers(0,len(right),len(right)); ld=left.iloc[li]; rd=right.iloc[ri]
        gap_boot.append((ld.with_signed.mean()-rd.with_signed.mean())-(ld.without_signed.mean()-rd.without_signed.mean()))
    with_gap=float(left.with_signed.mean()-right.with_signed.mean()); without_gap=float(left.without_signed.mean()-right.without_signed.mean()); cells=paired.groupby(["outlet","sampling_side"]).size()
    return {"pairs":len(paired),"outlets":paired.outlet.nunique(),"outlet_side_cell_min":int(cells.min()),"outlet_side_cell_max":int(cells.max()),"change_rates":{b:float(paired.loc[paired.target_bloc==b,"changed"].mean()) for b in ["left","right","far_right"]},"left_minus_other_change_rate":float(left.changed.mean()-right.changed.mean()),"left_minus_other_change_ci95":[float(v) for v in np.percentile(change_boot,[2.5,97.5])],"gap_with_group":with_gap,"gap_without_group":without_gap,"with_minus_without_gap":with_gap-without_gap,"with_minus_without_gap_ci95":[float(v) for v in np.percentile(gap_boot,[2.5,97.5])],"gradient_with_group":gradient(paired,"with_signed",ideology),"gradient_without_group":gradient(paired,"without_signed",ideology)}


def outlet_inventory(data,ideology):
    frozen=pd.read_csv(DATA/"outlet_inventory.csv"); monthly=pd.read_csv(DATA/"outlet_month_counts.csv")
    analytic=data.groupby("outlet").agg(analytic_pairs=("pair_id","size"),analytic_headlines=("article_id","nunique")); blocs=pd.crosstab(data.outlet,data.target_bloc)
    kept=["outlet","outlet_id","outlet_type","first_record","last_record","months_present","low_volume_months","enters_primary_correlation"]
    rebuilt=frozen[kept].copy().set_index("outlet").join(monthly.groupby("outlet").agg(raw_articles=("articles","sum"))).join(analytic).join(blocs).reset_index()
    columns=["outlet","outlet_id","raw_articles","months_present","outlet_type","analytic_pairs","analytic_headlines","centre","centre_right","far_right","left","other","right","enters_primary_correlation"]
    comparison=table_comparison(rebuilt[columns],DATA/"outlet_inventory.csv",["outlet"])
    monthly["month"]=monthly.month.astype(str); interior=monthly[monthly.month.between("2024-08","2026-05")].copy(); medians=interior.groupby("outlet").articles.median(); interior["median"]=interior.outlet.map(medians); interior["ratio"]=interior.articles/interior["median"]; low=interior[interior.ratio<0.5].copy(); low_keys=set(zip(low.outlet,low.month))
    dated=data.copy(); dated["month"]=pd.to_datetime(dated.date).dt.to_period("M").astype(str); without_low=dated[~pd.Series(list(zip(dated.outlet,dated.month)),index=dated.index).isin(low_keys)]; common=dated[dated.month.between("2024-08","2026-05")]; outlet_types=frozen.set_index("outlet").outlet_type.to_dict()
    type_sensitivity={name:gradient(dated[dated.outlet.map(outlet_types)!=name],"signed",ideology) for name in sorted(set(outlet_types.values()))}
    return {"raw_article_records":int(monthly.articles.sum()),"outlet_month_cells":len(monthly),"outlets":len(rebuilt),"reconstructed_columns_match_shipped_inventory":comparison,"sensitivity":{"baseline":gradient(dated,"signed",ideology),"excluding_low_volume_months":gradient(without_low,"signed",ideology),"common_complete_months":gradient(common,"signed",ideology),"leave_one_outlet_type_out":type_sensitivity},"low_volume_cells":records(low[["outlet","month","articles","median","ratio"]].sort_values(["outlet","month"])),"rows":records(rebuilt[columns].sort_values("outlet").reset_index(drop=True))}


# Cessation and resumption boundaries checked against the released interval table.
MANDATE_BOUNDARIES=[
  ("PA717161","2025-01-23",True,"Borne: last day of her first 17th-legislature mandate"),
  ("PA717161","2025-01-24",False,"Borne: seat vacated for government service"),
  ("PA717161","2025-06-01",False,"Borne: mid-absence"),
  ("PA717161","2025-11-12",False,"Borne: day before resumption"),
  ("PA717161","2025-11-13",True,"Borne: resumption of the mandate of a former government member"),
  ("PA719372","2024-10-21",True,"Kasbarian: last day before ministerial cessation"),
  ("PA719372","2024-12-01",False,"Kasbarian: mid-absence"),
  ("PA719372","2025-01-24",True,"Kasbarian: resumption"),
  ("PA759832","2025-11-12",False,"Pannier-Runacher: day before resumption"),
  ("PA759832","2025-11-13",True,"Pannier-Runacher: resumption"),
  ("PA842147","2024-10-21",False,"Mongardien: replacement deputy, seat not yet taken"),
  ("PA842147","2024-10-22",True,"Mongardien: takes the seat of a deputy named to government"),
  ("PA842147","2025-01-14",False,"Mongardien: seat returned to its holder"),
  ("PA368","2025-09-28",False,"Barnier: day before his by-election mandate begins"),
  ("PA368","2025-09-29",True,"Barnier: mandate begins"),
  ("PA720480","2026-03-27",True,"Parmentier-Lecocq: mandate resumed"),
]


def mandate_gate(pairs):
    """Re-apply the active-seat rule to the released rows, from the released mandate table.

    Three conditions are checked, and any failure aborts:

      1. no deputy has two overlapping seat intervals;
      2. every released mention falls inside one of its target's intervals;
      3. sixteen published cessation and resumption boundaries reproduce exactly.
    """
    table=pd.read_csv(DATA/"mandate_intervals.csv",dtype=str).fillna({"mandate_end":""})
    OPEN="9999-12-31"; spans={}
    for uid,group in table.groupby("target_uid"):
        spans[uid]=sorted((row.mandate_start,row.mandate_end or OPEN) for row in group.itertuples())
    overlaps=[]
    for uid,ivs in spans.items():
        for i in range(len(ivs)):
            for j in range(i+1,len(ivs)):
                if ivs[i][0]<=ivs[j][1] and ivs[j][0]<=ivs[i][1]:
                    overlaps.append({"target_uid":uid,"a":list(ivs[i]),"b":list(ivs[j])})
    if overlaps:
        raise RuntimeError(f"overlapping seat intervals in mandate_intervals.csv: {overlaps[:5]}")
    def inside(date,uid):
        return any(a<=date<=b for a,b in spans.get(uid,()))
    dates=pairs.date.astype(str).to_numpy(); uids=pairs.target_uid.to_numpy()
    outside=[(u,d) for u,d in zip(uids,dates) if not inside(d,u)]
    if outside:
        raise RuntimeError(f"{len(outside)} released mentions fall outside their target's mandate: {outside[:5]}")
    boundary_failures=[]
    for uid,date,expected,why in MANDATE_BOUNDARIES:
        if uid not in spans: boundary_failures.append(f"{uid} absent from mandate_intervals.csv ({why})")
        elif inside(date,uid)!=expected: boundary_failures.append(f"{uid} on {date}: expected {expected} -- {why}")
    if boundary_failures:
        raise RuntimeError("published mandate boundaries did not reproduce: "+"; ".join(boundary_failures))
    exclusions=pd.read_csv(DATA/"tenure_exclusions.csv")
    multi=[uid for uid,ivs in spans.items() if len(ivs)>1]
    return {"deputies_with_intervals":len(spans),"intervals":len(table),
      "deputies_with_more_than_one_interval":len(multi),
      "released_mentions_checked":len(pairs),"released_mentions_outside_mandate":0,
      "overlapping_interval_pairs":0,"published_boundaries_checked":len(MANDATE_BOUNDARIES),
      "source":"Assemblee nationale, base Sycomore; one fiche per roster member, retrieved 2026-09-13",
      "exclusions":{"individuals":len(exclusions),"pairs":int(exclusions.excluded_pairs.sum()),
        "by_individual":records(exclusions[["target_full_name","target_group","excluded_pairs","first_date","last_date"]])}}


def group_concentration(pairs):
    """Leave-one-deputy-out group tone, so a group level cannot rest on one person."""
    output={}
    for group,data in pairs.groupby("target_group"):
        if pd.isna(group) or str(group) in {"nan",""}: continue
        counts=data.target_full_name.value_counts()
        if counts.empty: continue
        top=counts.index[0]; rest=data[data.target_full_name!=top]
        output[group]={"n":len(data),"net":float(data.signed.mean()),
          "most_covered_deputy":top,"most_covered_mentions":int(counts.iloc[0]),
          "most_covered_share":float(counts.iloc[0]/len(data)),
          "n_excluding_most_covered":len(rest),
          "net_excluding_most_covered":float(rest.signed.mean()) if len(rest) else None,
          "net_of_most_covered":float(data[data.target_full_name==top].signed.mean())}
    return output


def cross_outlet_duplicates(pairs,ideology):
    """Exact-title cross-outlet copies: prevalence, and the all-copy exclusion sensitivity.

    Denominator is the analytic article records -- one row per (article, outlet) in the analytic
    set -- not raw crawl records. Only verbatim identical titles count; paraphrased syndication is
    not measured here and is not claimed to be.
    """
    groups=pd.read_csv(DATA/"duplicate_title_groups.csv",dtype={"article_id":str})
    articles=pairs.drop_duplicates("article_id")
    affected=set(groups.article_id)
    removed=pairs[~pairs.article_id.isin(affected)]
    return {"analytic_article_records":int(articles.article_id.nunique()),
      "analytic_mention_pairs":len(pairs),
      "distinct_titles_at_two_or_more_outlets":int(groups.title_group.nunique()),
      "article_records_affected":int(groups.article_id.nunique()),
      "article_record_share":float(groups.article_id.nunique()/articles.article_id.nunique()),
      "mention_pairs_affected":int(pairs.article_id.isin(affected).sum()),
      "definition":("verbatim identical title strings appearing at two or more outlets; "
                    "paraphrased syndication is not measured"),
      "all_copies_removed":{"remaining_pairs":len(removed),**gradient(removed,"signed",ideology)}}


def parliamentary_takeaways(pairs,framed,mp):
    """Guard the named parliamentary quantities promoted into the abstract and Results."""
    volume=pairs.groupby(["target_uid","target_full_name"],as_index=False).size().sort_values(["size","target_full_name"],ascending=[False,True])
    top=volume.head(5); groups={}
    for name in ["LFI-NFP","RN"]:
        subset=framed[framed.target_group==name]
        groups[name]={"net_tone":float(subset.signed.mean()),"conflict_prevalence":float(subset.frame_conflict.mean())}
    groups["centre_conflict_prevalence"]=float(framed[framed.target_bloc.isin(["centre","centre_right"])].frame_conflict.mean())
    conflict=mp.sort_values("conflict_shrunk",ascending=False).head(5)
    ciotti=pairs[pairs.target_full_name=="Éric Ciotti"].groupby("outlet").agg(n=("pair_id","size"),net=("signed","mean"))
    ciotti=ciotti[ciotti.n>=16].sort_values("net"); low=ciotti.iloc[0]; high=ciotti.iloc[-1]
    return {"volume_top_five":[{"name":r.target_full_name,"mentions":int(r.size)} for r in top.itertuples()],
      "marine_le_pen_share":float(pairs.target_full_name.eq("Marine Le Pen").mean()),
      "group_levels":groups,
      "conflict_top_five":[{"name":r.name,"shrunk_prevalence":float(r.conflict_shrunk)} for r in conflict.itertuples()],
      "ciotti_floor_16":{"floor":16,"low_outlet":low.name,"low_n":int(low.n),"low_net":float(low.net),"high_outlet":high.name,"high_n":int(high.n),"high_net":float(high.net)}}


def reproduce():
    print("  [1/7] Checking the active-seat gate against the released mandate record...",flush=True)
    pairs=pd.read_parquet(DATA/"mention_labels.parquet"); pairs=pairs[pairs.in_analysis].copy(); pairs["signed"]=pairs["label(pos/neg/neu)"].map(SIGNED); framed=pairs.dropna(subset=["frame_conflict"]).copy()
    gate=mandate_gate(pairs)
    print("  [2/7] Loading released data and rebuilding leaderboards...",flush=True)
    votes=pd.read_parquet(DATA/"rater_votes.parquet"); alignment=pd.read_csv(DATA/"votes_alignment_leg17.csv").rename(columns={"acteur_uid":"target_uid"}); scores=pd.read_csv(DATA/"frideo_scores.csv"); ideology={key(r.outlet):float(r.score) for r in scores.itertuples()}
    mp,groups,ni,priors=reconstruct_leaderboards(framed); checks={"mp":table_comparison(mp,DATA/"mp_leaderboard.csv",["uid"]),"groups":table_comparison(groups,DATA/"group_leaderboard.csv",["group"]),"non_attached":table_comparison(ni,DATA/"group_non_attached.csv",["group"])}
    if not all(x["matching"] for x in checks.values()):raise RuntimeError(f"reconstructed leaderboard mismatch: {checks}")
    print("  [3/7] Recomputing exploratory H3 rank checks...",flush=True)
    h3,h3_points=h3_results(framed,alignment)
    print("  [4/7] Running the 1,000-replicate H3 precision bootstrap...",flush=True)
    h3_precision=h3_precision_models(framed,alignment,h3_points)
    print("  [5/7] Recomputing frame results and agreement...",flush=True)
    frames=frame_results(framed,ideology)
    print("  [6/7] Running primary-gradient robustness and permutation checks...",flush=True)
    robust=robustness(pairs,votes,ideology)
    permutation_ps={"directed_tone":robust["primary"]["fixed_panel_permutation"]["p_value"],"conflict":frames["fixed_panel_permutation"]["conflict"]["p_value"],"morality":frames["fixed_panel_permutation"]["morality"]["p_value"]}
    ordered=sorted(permutation_ps.items(),key=lambda item:item[1]); adjusted={}; running=0.0
    for position,(name,value) in enumerate(ordered): running=max(running,min(1.0,(len(ordered)-position)*value)); adjusted[name]=running
    return {
      "coverage":{"independently_recomputed":["active-seat mandate gate, interval overlap and published cessation/resumption boundaries","corpus and analytic-set composition","leave-one-deputy-out group tone concentration","exact-title cross-outlet duplication and its all-copy exclusion sensitivity","outlet inventory analytic columns","primary and robustness gradients","bloc contrasts","role decomposition","within-deputy period checks","frame prevalence and tone associations","complete group and deputy leaderboards","sentiment and role inter-model alpha","frame inter-model agreement and majority-vote reconstruction","human audit agreement, weighted task metrics and group-context bias","rank-based H3 estimates","prompt-paraphrase sensitivity","group-field ablation"],"aggregate_only_checked":["per-individual tenure-exclusion counts, whose excluded rows are outside the released analytic set","frame alpha bootstrap confidence intervals as published (own resampling scheme agrees to within the registered tolerance)","H3 lagged-vote, leadership and tenure covariate extensions","outlet-sampling and joint-placement audits"],"not_reconstructed_from_release":["crawler recall and mention-extraction attrition, which need the upstream crawl logs","standalone Cointet hyperlink placement, whose hyperlink graph is not ours to redistribute","FrIdeo source-model fitting and its placement uncertainty, which need the source corpus","any label re-derivation from headline text, which is withheld for copyright"]},
      "mandate_gate":gate,
      "group_concentration":group_concentration(pairs),
      "cross_outlet_duplicates":cross_outlet_duplicates(pairs,ideology),
      "primary_gradient_fixed_panel_article_bootstrap":fixed_panel_article_bootstrap(pairs,ideology),
      "corpus":{"pairs":len(pairs),"headlines":pairs.article_id.nunique(),"outlets":pairs.outlet.nunique(),"deputies":pairs.target_uid.nunique(),"negative_share":float(pairs["label(pos/neg/neu)"].eq("neg").mean()),"neutral_share":float(pairs["label(pos/neg/neu)"].eq("neu").mean()),"positive_share":float(pairs["label(pos/neg/neu)"].eq("pos").mean()),"overall_net_tone":float(pairs.signed.mean()),"full_name_matches":int(pairs.match_type.isin(["full_name","full_name_ambiguous"]).sum()),"surname_only_matches":int(pairs.match_type.isin(["surname_only","surname_ambiguous"]).sum()),"frame_pairs":len(framed),"frame_headlines":framed.article_id.nunique()},
      "outlet_inventory":outlet_inventory(pairs,ideology),"outlet_tone_table":records(outlet_asymmetry(pairs,"signed",ideology)),"robustness":robust,"fixed_panel_multiplicity":{"raw_permutation_p":permutation_ps,"holm_adjusted_p":adjusted},"pooled_gap_inference":pooled_gap_inference(pairs),"bloc_contrasts":bloc_contrasts(pairs,ideology),"role_decomposition":role_decomposition(pairs,ideology),"within_cell_gradients":within_cell_gradients(pairs,ideology),"temporal_split_gradient":temporal_split_gradient(pairs,ideology),"frames":frames,"inter_model_agreement":inter_model_agreement(votes),"frame_inter_model_agreement":frame_inter_model_agreement(framed),"human_validation":human_analysis(framed),
      "parliamentary_takeaways":parliamentary_takeaways(pairs,framed,mp),"leaderboards":{"priors":priors,"checks_against_shipped_tables":checks,"groups":records(groups),"non_attached":records(ni),"deputies":records(mp)},"exploratory_h3":h3,"h3_precision_models":h3_precision,"tie_sensitivity":tie_sensitivity(pairs,votes,ideology),"prompt_paraphrase":prompt_sensitivity(pairs,ideology,votes),"group_field_ablation":group_field_ablation(ideology),
      "figure_data":{"validation":records(outlet_asymmetry(pairs,"signed",ideology)),"framing":frames["outlet_gradients"],"loyalty":records(h3_points[["target_uid","name","group","n","alignment_pct","conflict","net"]].sort_values("target_uid")),"party_tone":records(framed.groupby(["target_uid","target_full_name","target_group"],as_index=False).agg(n=("pair_id","size"),net=("signed","mean")).query("n >= 10 and target_group != 'NI'").sort_values(["target_group","target_uid"]))},
      "supplied_aggregate_records":{
        "frame_inter_model_agreement":json.loads((DATA/"model_iaa_frames.json").read_text()),
        "human_agreement":json.loads((DATA/"human_agreement.json").read_text()),
        "human_model_validity":json.loads((DATA/"human_model_validity.json").read_text()),
        "group_context_bias":json.loads((DATA/"group_context_bias.json").read_text()),
        "h3_precision_models":json.loads((DATA/"h3_precision_audit.json").read_text()),
        "h3_estimand_extensions":json.loads((DATA/"h3_estimand_audit.json").read_text()),
        "joint_placement_uncertainty":json.loads((DATA/"joint_placement_uncertainty.json").read_text()),
        "multiplicity":json.loads((DATA/"multiplicity_audit.json").read_text()),
        "outlet_sampling":json.loads((DATA/"outlet_sampling_audit.json").read_text()),
        "clustered_pooled_gap":json.loads((DATA/"cluster_gap_inference.json").read_text())}}


# Numerical tolerances, applied to the first matching rule by path substring.
# Everything not matched is recomputed by deterministic arithmetic from the
# released rows and must agree to near machine precision, so drift fails loudly.
TOLERANCES=(
  ("/h3_precision_models/",0.0,1e-4,"iterative maximum-likelihood and clustered-covariance fits; optimiser paths differ slightly across BLAS builds"),
  ("alpha_nominal_ci95_own_bootstrap",0.0,5e-3,"1,000-replicate unit bootstrap; interval endpoints carry Monte Carlo error"),
  ("max_abs_deviation_from_supplied_record",0.0,5e-5,"agreement of recomputed alphas with the four-decimal supplied record"),
  ("/supplied_aggregate_records/",0.0,0.0,"verbatim contents of shipped records; must match exactly"),
  ("/fixed_panel_permutation/p_value",0.0,2e-6,"a permutation p is resolved only to one exceedance in 1,000,000 draws"),
  ("permutation_p_value",0.0,1e-4,"a permutation p is resolved only to one exceedance in 20,000 draws"),
  ("",1e-9,1e-12,"deterministic recomputation from released rows, including seeded resampling with a fixed bit generator"),
)


def tolerance(path):
    for marker,rel_tol,abs_tol,_ in TOLERANCES:
        if marker in path: return rel_tol,abs_tol
    return 1e-9,1e-12


def compare(expected,actual,path=""):
    failures=[]
    if isinstance(expected,dict):
        if not isinstance(actual,dict):return [f"{path}: expected object, got {type(actual).__name__}"]
        for name in sorted(set(expected)|set(actual)):
            if name not in actual: failures.append(f"{path}/{name}: missing from reproduced output")
            elif name not in expected: failures.append(f"{path}/{name}: present in reproduced output but not registered in expected_results.json")
            else: failures.extend(compare(expected[name],actual[name],f"{path}/{name}"))
    elif isinstance(expected,list):
        if not isinstance(actual,list):return [f"{path}: expected list, got {type(actual).__name__}"]
        if len(expected)!=len(actual):failures.append(f"{path}: expected {len(expected)} items, got {len(actual)}")
        else:
            for i,(e,a) in enumerate(zip(expected,actual)):failures.extend(compare(e,a,f"{path}[{i}]"))
    elif isinstance(expected,float):
        rel_tol,abs_tol=tolerance(path)
        try:
            if not math.isclose(expected,float(actual),rel_tol=rel_tol,abs_tol=abs_tol):failures.append(f"{path}: expected {expected!r}, got {actual!r} (rel_tol={rel_tol}, abs_tol={abs_tol})")
        except (TypeError,ValueError):failures.append(f"{path}: expected {expected!r}, got {actual!r}")
    elif expected!=actual:failures.append(f"{path}: expected {expected!r}, got {actual!r}")
    return failures


def count_values(node):
    if isinstance(node,dict): return sum(count_values(v) for v in node.values())
    if isinstance(node,list): return sum(count_values(v) for v in node)
    return 1


def main():
    actual=json_safe(reproduce()); OUTPUTS.mkdir(exist_ok=True); path=OUTPUTS/"reproduced_results.json"; path.write_text(json.dumps(actual,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("  [7/7] Comparing every result with the registered reference...",flush=True)
    registry=HERE/"expected_results.json"
    if not registry.exists():
        print(f"FAIL: {registry.name} is missing; nothing to check against"); raise SystemExit(1)
    failures=compare(json.loads(registry.read_text()),actual)
    if failures:
        print("FAIL: registered results drifted")
        for failure in failures[:100]:print(f"  - {failure}")
        if len(failures)>100:print(f"  - ... {len(failures)-100} more")
        raise SystemExit(1)
    print(json.dumps({"independently_recomputed_families":len(actual["coverage"]["independently_recomputed"]),
      "aggregate_only_families":len(actual["coverage"]["aggregate_only_checked"]),
      "not_reconstructed_families":len(actual["coverage"]["not_reconstructed_from_release"]),
      "registered_values_checked":count_values(actual),
      "tolerances":[{"paths_containing":marker or "(default)","rel_tol":rel_tol,"abs_tol":abs_tol,"reason":reason} for marker,rel_tol,abs_tol,reason in TOLERANCES],
      "registered_output":str(path.relative_to(HERE))},indent=2)); print("PASS: all registered results reproduced")


if __name__=="__main__":main()
