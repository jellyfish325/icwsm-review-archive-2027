# Exploratory H3 analysis protocol

H3 examines whether within-party legislative alignment is associated with the
conflict framing of individual deputies. This analysis is exploratory.

## Eligibility

A deputy enters the analysis when all of the following hold:

1. the deputy belongs to a parliamentary group rather than the non-attached
   category;
2. the deputy does not hold executive office during the analysis window;
3. the alignment measure is based on at least 50 compared votes; and
4. the deputy has at least 10 eligible headline mentions.

These rules yield 90 deputies in the final release.

## Estimator

Within each parliamentary group, the analysis ranks alignment and conflict
prevalence, divides each rank by group size, centres both normalized ranks on
their group means, and computes the Pearson correlation of the pooled centred
vectors. The final estimate is `r_rank = 0.1775878756`.

The two-sided uncertainty interval comes from 4,000 party-stratified bootstrap
draws that rerank deputies within every draw. The within-party permutation test
uses 20,000 permutations. Both procedures use fixed seeds reported in
`expected_results.json` and reproduced by `reproduce.py`.

## Precision model and interpretation

The count-model sensitivity uses each deputy's conflict count and total eligible
mentions, with group effects and the specified covariates. The release includes
the fixed generating parameter vector and verifies it against a fit to the
released counts before running 1,000 seeded bootstrap replicates.

Results are reported as an exploratory association with its uncertainty. The
analysis does not support a confirmatory or causal claim.
