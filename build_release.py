#!/usr/bin/env python3
"""Rebuild `data/` in this reviewer archive from the analysis inputs.

The script copies and projects the final analytic data: it strips withheld
columns (headline text, free-text audit notes, collision ids), restricts paired
experiments and the human audit sample to the released population, and writes
the aggregate analysis records. It computes no paper result of its own;
`reproduce.py` recomputes those results independently from the released rows.

Usage:  python3 build_release.py [--check]
        --check rebuilds into a temporary directory and diffs, changing nothing.
"""
from __future__ import annotations

import argparse, filecmp, json, shutil, sys, tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MPS = HERE.parent
PILOT = MPS / "pilot"

WITHHELD = ["title", "notes", "collision_id"]          # never leave the private tree
FRAMES = ["conflict", "human_interest", "economic", "morality", "responsibility"]
RATERS = {"gpt_oss": "population_study_annotated.parquet",
          "mistral": "population_study_annotated_mistral-small-4-119b.parquet",
          "llama":   "population_study_annotated_Llama-3.3-70B-Instruct.parquet"}
FRAME_RATERS = {"gpt_oss": "population_frames_gpt-oss-120b.parquet",
                "mistral": "population_frames_mistral-small-4-119b.parquet",
                "llama":   "population_frames_Llama-3.3-70B-Instruct.parquet"}

# Aggregate analysis records: released name -> analysis input and producing script.
SOURCES = {
    "model_iaa_sentiment_role.json": ("pilot/iaa_report.json", "build_majority_vote.py"),
    "model_iaa_frames.json": ("pilot/frames_iaa_report.json", "build_frames_majority.py"),
    "human_agreement.json": ("pilot/human_ceiling/per_bloc_agreement.json", "human_ceiling_per_bloc.py"),
    "human_model_validity.json": ("pilot/human_validity_out/human_model_validity.json", "human_model_validity_audit.py"),
    "group_context_bias.json": ("pilot/group_context_out/group_context_bias.json", "group_context_bias_audit.py"),
    "h3_precision_audit.json": ("pilot/precision_audit_out/h3_precision_audit.json", "analysis_h3_precision_audit.py"),
    "h3_estimand_audit.json": ("pilot/estimand_audit_out/h3_estimand_audit.json", "analysis_h3_estimand_audit.py"),
    "joint_placement_uncertainty.json": ("pilot/joint_placement_uncertainty_out/joint_placement_uncertainty.json", "analysis_joint_placement_uncertainty.py"),
    "multiplicity_audit.json": ("pilot/multiplicity_audit_out/multiplicity_audit.json", "multiplicity_audit.py"),
    "outlet_sampling_audit.json": ("pilot/outlet_sampling_out/outlet_sampling_audit.json", "outlet_sampling_audit.py"),
    "cluster_gap_inference.json": ("pilot/cluster_gap_inference_out/cluster_gap_inference.json", "cluster_gap_inference_audit.py"),
    "outlet_inventory.csv": ("pilot/outlet_sampling_out/outlet_inventory.csv", "outlet_sampling_audit.py"),
    "outlet_month_counts.csv": ("pilot/outlet_sampling_out/outlet_month_counts_2024-07-18_2026-06-24.csv", "outlet_sampling_audit.py"),
    "mp_leaderboard.csv": ("pilot/leaderboards_out/mp_all.csv", "analysis_leaderboards.py"),
    "group_leaderboard.csv": ("pilot/leaderboards_out/group_all.csv", "analysis_leaderboards.py"),
    "group_non_attached.csv": ("pilot/leaderboards_out/group_non_attached.csv", "analysis_leaderboards.py"),
    "votes_alignment_leg17.csv": ("pilot/votes_alignment_leg17.csv", "fetch_votes_alignment.py"),
    # group_field_ablation_full.csv is filtered to the final analytic population.
}
# Fixed inputs used by the released analysis.
CARRIED = ["frideo_scores.csv", "frideo_scores_no_d3_d7.csv", "conflict_error_sensitivity.md"]


def clean_release_record(name: str, value):
    """Map an analysis record onto the released schema: estimand-specific field names, released fields only."""
    rename = {
        "corrected_net_tone": "bias_adjusted_net_tone",
        "outlet_ideology_rho_side_corrected": "outlet_ideology_rho_side_adjusted",
        "global_corrected": "global_error_adjusted",
        "side_corrected": "side_error_adjusted",
        "n_dropped_unrecovered_na": "n_dropped_missing_vote_vector",
    }
    omit = {
        "historical_pooled_centered_raw",
        "historical_recomputed_ci95",
        "historical_recomputed_median",
    }

    def visit(node):
        if isinstance(node, dict):
            return {rename.get(key, key): visit(item) for key, item in node.items() if key not in omit}
        if isinstance(node, list):
            return [visit(item) for item in node]
        return node

    value = visit(value)
    if name == "human_agreement.json":
        sample_key = next(key for key, item in value.items() if isinstance(item, dict) and "n_key_pairs" in item)
        sample = value.pop(sample_key)
        value["audit_sample"] = {"n_units": sample["n_key_pairs"]}
    if name in {"human_model_validity.json", "group_context_bias.json"}:
        value["audit_sample"] = {"n_units": value["audit_sample"]["n_used"]}
    if name == "human_model_validity.json":
        value["conflict_error_propagation"]["interpretation"] = (
            "Nondifferential error adjustment is affine within each party, so H3 ranks are "
            "unchanged; target-side-specific adjustment can change outlet asymmetry ordering."
        )
    return value


def same_released_content(generated: Path, live: Path) -> bool:
    """Compare logical table content, while leaving byte integrity to MANIFEST.sha256.

    Parquet writers can encode identical frames with different file metadata across
    PyArrow versions. Treating that serialization difference as data drift made
    ``--check`` fail even when every cell, dtype, column, and row order matched.
    """
    if not generated.exists() or not live.exists():
        return False
    if generated.suffix == ".parquet":
        try:
            pd.testing.assert_frame_equal(
                pd.read_parquet(generated),
                pd.read_parquet(live),
                check_dtype=True,
                check_exact=True,
                check_like=False,
            )
            return True
        except (AssertionError, OSError, ValueError):
            return False
    return filecmp.cmp(generated, live, shallow=False)


def build(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    notes = {}

    # ---- mention labels: the analytic spine, plus the headline-level frame vector.
    pairs = pd.read_parquet(PILOT / "population_study_majority.parquet")
    pairs = pairs[pairs.in_analysis].copy()
    frames = pd.read_parquet(PILOT / "population_frames_majority.parquet")
    pairs["article_id"] = pairs["article_id"].astype(str)
    frames["article_id"] = frames["article_id"].astype(str)
    frame_cols = [f"frame_{f}" for f in FRAMES] + ["dominant"]
    labels = pairs.drop(columns=[c for c in WITHHELD if c in pairs.columns]) \
                  .merge(frames[["article_id"] + frame_cols], on="article_id", how="left")
    columns = ["pair_id", "article_id", "outlet_id", "outlet", "date", "category",
               "target_uid", "target_full_name", "target_group", "target_bloc",
               "role_bucket", "match_type", "rank", "role(source/patient/agent)",
               "label(pos/neg/neu)", "held_seat", "gov_tenure", "surname_misattrib",
               "disambig_drop", "in_analysis"] + frame_cols
    labels = labels[columns]
    labels.to_parquet(out / "mention_labels.parquet", index=False)
    labels.to_csv(out / "mention_labels.csv", index=False)
    analytic = set(labels.pair_id)
    notes["mention_labels"] = {"rows": len(labels),
                               "rows_without_frame_vector": int(labels.frame_conflict.isna().sum()),
                               "articles": int(labels.article_id.nunique())}

    # ---- per-model sentiment/role votes for exactly those pairs.
    votes = None
    for rater, filename in RATERS.items():
        one = pd.read_parquet(PILOT / filename,
                              columns=["pair_id", "label(pos/neg/neu)", "role(source/patient/agent)"])
        one = one.rename(columns={"label(pos/neg/neu)": f"label_{rater}",
                                  "role(source/patient/agent)": f"role_{rater}"})
        votes = one if votes is None else votes.merge(one, on="pair_id", how="inner")
    votes = votes[votes.pair_id.isin(analytic)].reset_index(drop=True)
    votes = votes[["pair_id"] + [f"{kind}_{r}" for kind in ("label", "role") for r in RATERS]]
    if len(votes) != len(labels):
        raise SystemExit(f"rater_votes has {len(votes)} rows for {len(labels)} analysis pairs")
    votes.to_parquet(out / "rater_votes.parquet", index=False)
    notes["rater_votes"] = {"rows": len(votes)}

    # ---- per-model frame votes over the whole frame-annotation universe (not population-filtered:
    # frame agreement is a property of the headline universe, and the paper reports it as such).
    frame_votes = None
    for rater, filename in FRAME_RATERS.items():
        one = pd.read_parquet(PILOT / filename)
        one["article_id"] = one["article_id"].astype(str)
        keep = ["article_id"] + [f"frame_{f}" for f in FRAMES] + ["dominant"]
        one = one[keep].rename(columns={**{f"frame_{f}": f"frame_{f}_{rater}" for f in FRAMES},
                                        "dominant": f"dominant_{rater}"})
        frame_votes = one if frame_votes is None else frame_votes.merge(one, on="article_id", how="outer")
    order = ["article_id"] + [f"frame_{f}_{r}" for r in FRAME_RATERS for f in FRAMES] \
            + [f"dominant_{r}" for r in FRAME_RATERS]
    frame_votes = frame_votes[order].sort_values("article_id").reset_index(drop=True)
    frame_votes.to_csv(out / "frame_rater_votes.csv", index=False)
    notes["frame_rater_votes"] = {"rows": len(frame_votes)}

    # ---- blind double-coded human audit, stripped to labels and restricted to the population.
    human_dir = PILOT / "human_ceiling"
    coders = [pd.read_csv(human_dir / f"sample_blind_coder{i}_annotated.csv", dtype=str).set_index("pair_id")
              for i in (1, 2)]
    cols = ["human_role(source/patient/agent)", "human_label(neg/neu/pos)"] + \
           [f"human_frame_{f}(0/1)" for f in FRAMES]
    coded = coders[0][cols].add_suffix("_coder1").join(coders[1][cols].add_suffix("_coder2"), how="inner")
    n_coded = len(coded)
    coded = coded[coded.index.isin(analytic)]
    coded.reset_index().to_csv(out / "human_labels.csv", index=False)
    notes["human_labels"] = {"coded": n_coded, "released": len(coded),
                             "outside_final_population": n_coded - len(coded)}

    # ---- paired experiments, restricted to the final population.
    para = pd.read_parquet(PILOT / "prompt_perturb_out" / "prompt_perturb_labels.parquet",
                           columns=["pair_id", "label_para"]).rename(columns={"label_para": "paraphrased_label"})
    n_para = len(para)
    para = para[para.pair_id.isin(analytic)].reset_index(drop=True)
    para.to_csv(out / "prompt_paraphrase_labels.csv", index=False)
    notes["prompt_paraphrase_labels"] = {"annotated": n_para, "released": len(para),
                                         "outside_final_population": n_para - len(para)}

    # Read the complete experiment input so repeated builds remain idempotent.
    ablation = pd.read_csv(PILOT / "group_field_ablation_full.csv", dtype={"pair_id": str})
    n_abl = len(ablation)
    ablation = ablation[ablation.pair_id.isin(analytic)].reset_index(drop=True)
    ablation.to_csv(out / "group_field_ablation.csv", index=False)
    notes["group_field_ablation"] = {"annotated": n_abl, "released": len(ablation),
                                     "outside_final_population": n_abl - len(ablation)}

    # ---- final mandate intervals, so the active-seat gate is checkable from the release.
    mandate_candidates = list(MPS.glob("mandates_leg17_seat_*.json"))
    if len(mandate_candidates) != 1:
        raise SystemExit(f"expected one mandate-interval input, found {len(mandate_candidates)}")
    mandate_source = mandate_candidates[0]
    mandates = json.loads(mandate_source.read_text())
    rows = []
    for uid, record in mandates["records"].items():
        for start, end in record["intervals"]:
            rows.append({"target_uid": uid, "name": record["name"], "mandate_start": start,
                         "mandate_end": end or "", "sycomore_fid": record["sycomore_fid"],
                         "source_url": record["source_url"], "retrieved": record["retrieved"]})
    intervals = pd.DataFrame(rows).sort_values(["target_uid", "mandate_start"]).reset_index(drop=True)
    intervals.to_csv(out / "mandate_intervals.csv", index=False)
    notes["mandate_intervals"] = {"deputies": int(intervals.target_uid.nunique()),
                                  "intervals": len(intervals),
                                  "source": mandates["source"].split(";")[0].strip()}

    # ---- per-individual tenure exclusions: the rows the gate removes are NOT released
    # (they are outside the analytic population), so this aggregate is what makes the
    # manuscript's exclusion breakdown checkable.
    gated = pd.read_parquet(PILOT / "population_study_gated.parquet")
    false_match = gated["notes"].fillna("").str.contains("false match", case=False)
    candidates = gated[~false_match]
    excluded = candidates[~candidates.held_seat]
    breakdown = (excluded.groupby(["target_uid", "target_full_name", "target_group"])
                 .agg(excluded_pairs=("pair_id", "size"), first_date=("date", "min"), last_date=("date", "max"))
                 .reset_index().sort_values(["excluded_pairs", "target_full_name"], ascending=[False, True]))
    breakdown.to_csv(out / "tenure_exclusions.csv", index=False)
    notes["tenure_exclusions"] = {"individuals": len(breakdown),
                                  "excluded_pairs": int(breakdown.excluded_pairs.sum()),
                                  "raw_candidate_pairs": len(gated),
                                  "false_match_pairs": int(false_match.sum())}

    # ---- cross-outlet exact-duplicate titles, as group ids only. The headline text itself is
    # withheld for copyright, so duplicates are released as a shared integer key per identical
    # normalised title, which is all the published prevalence and sensitivity checks need.
    titles = pd.read_parquet(PILOT / "population_study_majority.parquet", columns=["article_id", "outlet", "title"])
    titles = titles.drop_duplicates("article_id").copy()
    titles["article_id"] = titles["article_id"].astype(str)
    titles = titles[titles.article_id.isin(set(labels.article_id))]
    # EXACT title strings, as the published prevalence figure is defined; no casefolding or
    # whitespace normalisation, so this counts verbatim copies and not paraphrased syndication.
    spread = titles.groupby("title").outlet.nunique()
    shared = set(spread[spread > 1].index)
    dup = titles[titles.title.isin(shared)].copy()
    dup["title_group"] = pd.factorize(dup.title)[0] + 1
    dup[["article_id", "title_group"]].sort_values(["title_group", "article_id"]).to_csv(
        out / "duplicate_title_groups.csv", index=False)
    notes["duplicate_title_groups"] = {"distinct_titles_at_two_or_more_outlets": int(dup.title_group.nunique()),
                                       "article_records": len(dup),
                                       "analytic_article_records": int(titles.article_id.nunique())}

    # ---- aggregate records and fixed inputs.
    for name, (relative, _script) in SOURCES.items():
        source = MPS / relative
        if not source.exists():
            raise SystemExit(f"missing private source for {name}: {source}")
        if source.suffix == ".json":
            record = clean_release_record(name, json.loads(source.read_text()))
            (out / name).write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=True) + "\n")
        else:
            shutil.copyfile(source, out / name)
    for name in CARRIED:
        current = HERE / "data" / name
        if current.resolve() != (out / name).resolve():
            shutil.copyfile(current, out / name)
    return notes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="build into a temporary directory and report differences only")
    args = parser.parse_args()
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            notes = build(Path(tmp))
            live = HERE / "data"
            names = sorted({p.name for p in Path(tmp).iterdir()} | {p.name for p in live.iterdir()
                                                                    if p.is_file() and not p.name.startswith(".")})
            differing = [n for n in names
                         if not same_released_content(Path(tmp) / n, live / n)]
            print(json.dumps({"notes": notes, "differing_files": differing}, indent=2))
            sys.exit(1 if differing else 0)
    notes = build(HERE / "data")
    print(json.dumps(notes, indent=2))
    print("rebuilt data/ -- now refresh MANIFEST.sha256 and re-register expected_results.json")


if __name__ == "__main__":
    main()
