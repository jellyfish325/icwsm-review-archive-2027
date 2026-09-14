# Anonymous review archive

Reproduction package for *How French News Outlets Cover Deputies of the 17th
National Assembly: A Headline-Level Study of Directed Sentiment and Framing*. It
contains derived labels, public covariates, protocols, audit records, and an
executable result check. Headlines are withheld for copyright reasons.

## Run

First run: internet access, about 500 MB of disk space, and
[`uv`](https://docs.astral.sh/uv/). Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Run the launcher from any directory:

```bash
/path/to/review_archive/run_review.sh
```

The launcher verifies file integrity, installs the pinned Python 3.13.5
environment, runs the analysis, and checks 5,572 values. It takes 5-10 minutes
and writes `outputs/reproduced_results.json`.

Success ends with:

```text
PASS: all registered results reproduced
```

Failures exit non-zero and identify the file or value that failed.

## Coverage

Recomputed from released rows:

- the active-seat gate: every released mention is re-tested against the Assembly's own
  mandate intervals in `data/mandate_intervals.csv`, no deputy may hold two overlapping
  intervals, and sixteen published cessation and resumption boundaries must reproduce;
- corpus composition, outlet inventory, primary gradient, and robustness checks;
- leave-one-deputy-out group tone, and exact-title cross-outlet duplication with its
  all-copy exclusion sensitivity;
- bloc contrasts, role decomposition, within-deputy checks, and frame analyses;
- group and MP leaderboards, model agreement, and human validation;
- H3 rank and precision models, sensitivity experiments, and figure data.

Checked against included aggregate records:

- per-individual tenure-exclusion counts, whose excluded rows lie outside the released
  analytic set;
- published frame-alpha bootstrap intervals;
- extended H3, outlet-sampling, and outlet-placement uncertainty audits.

Not reproduced because the required source material cannot be distributed:

- crawl and mention-extraction recall;
- the Cointet hyperlink graph and FrIdéo source-model fit;
- labels re-derived from withheld headline text.

The H3 bootstrap re-fits the released MP counts, verifies the fixed
full-precision generating fit, then runs 1,000 seeded replicates. These temporary
simulations estimate uncertainty; they are not released observations.

`build_release.py` regenerates `data/` from the analysis inputs;
`build_release.py --check` rebuilds into a temporary directory and reports any
file that differs, changing nothing.

See `SCHEMA.md` for fields and final row counts. The package does not query the news
database or call annotation models. Data are CC BY 4.0 and code is MIT; see
`LICENSES.md` for third-party terms.
