# Release schema

The final analytic population contains 19,981 `(headline, deputy)` pairs over
18,981 headlines. Every released pair falls within the target deputy's seat
interval in the National Assembly's Sycomore records. Those intervals are
released in `data/mandate_intervals.csv`. `reproduce.py` checks every pair,
rejects overlapping intervals, and verifies sixteen cessation and resumption
boundaries.

## `data/mention_labels.parquet` / `data/mention_labels.csv`

The same 19,981 rows and columns in both formats.

| Field | Meaning |
|---|---|
| `pair_id` | Stable identifier for a `(headline, deputy)` analysis unit. |
| `article_id` | Stable source-record identifier; no source text or URL is included. |
| `outlet_id`, `outlet` | Stable outlet code and public outlet name. |
| `date`, `category` | Publication date and source category metadata. |
| `target_uid`, `target_full_name` | Public National Assembly identifier and deputy name. |
| `target_group`, `target_bloc` | Parliamentary group and analysis bloc. |
| `role_bucket`, `role(source/patient/agent)` | Extraction-side and majority-vote discursive-role labels. |
| `match_type`, `rank` | Mention-matching metadata. |
| `label(pos/neg/neu)` | Majority-vote directed-sentiment label. |
| `held_seat`, `gov_tenure`, `surname_misattrib`, `disambig_drop`, `in_analysis` | Eligibility and audit flags. |
| `frame_conflict`, `frame_human_interest`, `frame_economic`, `frame_morality`, `frame_responsibility` | Majority-vote binary generic-frame labels. |
| `dominant` | Derived dominant-frame label. |

One analysis pair has missing frame fields because its article identifier has no
row in the released frame-majority table. Tone analyses therefore use 19,981 pairs;
frame analyses use the 19,980 pairs with a frame vector, over 18,980 articles.
The reproduction script enforces these observed populations rather than imputing
the missing frame.

## `data/rater_votes.parquet`

One row per analysis pair.

| Field | Meaning |
|---|---|
| `pair_id` | Join key to `mention_labels`. |
| `label_gpt_oss`, `label_mistral`, `label_llama` | Each model's directed-sentiment vote. |
| `role_gpt_oss`, `role_mistral`, `role_llama` | Each model's discursive-role vote; off-vocabulary strings are retained verbatim and treated as missing. |

## `data/frame_rater_votes.csv`

One row per article in the frame-annotation universe (19,976 rows), derived from
the three per-model frame tables.

| Field | Meaning |
|---|---|
| `article_id` | Join key to `mention_labels.article_id`. |
| `frame_<frame>_<rater>` | That rater's binary vote on that frame, for each of the five frames and the three raters (15 columns). |
| `dominant_<rater>` | That rater's dominant-frame vote, one of the five frames or `none` (3 columns). |

One article (`4555807`) has an empty vote vector from one rater and is dropped by
the documented alignment rule, leaving the 19,975 units on which the published
frame agreement is computed. Frame agreement is a property of the annotation
universe and is reported over all of it; 995 of those articles carry no mention that
passes the active-seat gate, so the majority-vote reconstruction is
checked on the 18,980 articles that do, and the shortfall is reported alongside it. Taking the per-frame majority over the three raters
reproduces the `frame_*` columns of `mention_labels` exactly; the dominant-frame
majority reproduces `dominant` wherever a two-of-three majority exists. The 1,311
articles on which all three raters name a different dominant frame were resolved
by a fallback that cannot be re-derived from these votes alone, so the script
checks the resolved labels rather than recomputing them.

## `data/human_labels.csv`

The final blind double-coded audit sample, stripped to labels, contains 440
eligible units. Inverse-probability weights use the final analytic-population
strata.

| Field | Meaning |
|---|---|
| `pair_id` | Join key to `mention_labels`. |
| `human_label(neg/neu/pos)_coder1`, `..._coder2` | Each coder's sentiment label. |
| `human_role(source/patient/agent)_coder1`, `..._coder2` | Each coder's role label. |
| `human_frame_<frame>(0/1)_coder1`, `..._coder2` | Each coder's binary frame labels. |

Coder identities, free-text notes, and the headline shown to the coders are not
included. The coding sheets withheld parliamentary group and bloc, so these
labels are the group-blind reference used for the group-context bias check.

## `data/prompt_paraphrase_labels.csv`, `data/group_field_ablation.csv`

Paired experiment outputs, stripped to stable pair identifiers and derived
labels. `prompt_paraphrase_labels.csv` carries `pair_id` and `paraphrased_label`
for 3,851 eligible pairs. `group_field_ablation.csv` contains 1,940 eligible
pairs and carries `pair_id`, the public strata needed to reproduce the reported contrasts
(`outlet`, `target_bloc`, `sampling_side`) and the label from each arm
(`label_with_group`, `label_without_group`). No prompt text,
raw response, headline, note, or serving metadata is included.

## `data/mandate_intervals.csv`

Every roster member's 17th-legislature Assembly seat intervals (599 rows over 577
deputies), as recorded in the Assembly's Sycomore database and retrieved on 2026-09-13.

| Field | Meaning |
|---|---|
| `target_uid` | National Assembly actor identifier, the join key to `mention_labels`. |
| `name` | Deputy name as carried in the roster. |
| `mandate_start`, `mandate_end` | Inclusive interval; an empty end means the mandate was still open at retrieval. |
| `sycomore_fid`, `source_url` | The Sycomore record the interval was read from. |
| `retrieved` | Retrieval date. |

22 deputies have more than one interval, because a deputy who is appointed to government
loses the seat one month later and resumes it one month after leaving, and the deputy who
replaced them holds it in between.

## `data/tenure_exclusions.csv`

One row per individual whose mentions the active-seat gate removes (26 individuals,
5,336 pairs), with the first and last excluded date. The excluded rows themselves are not
released, because they are outside the analytic population; this aggregate is what makes
the manuscript's exclusion breakdown checkable.

## `data/duplicate_title_groups.csv`

`article_id` and a shared integer `title_group` for the article records whose headline
string appears verbatim at two or more outlets (130 records in 60 title groups). The
headline text itself is withheld, so duplication is released as a group key. This measures
verbatim copies only, not paraphrased syndication.

## `data/outlet_inventory.csv`, `data/outlet_month_counts.csv`

The 29-outlet inventory (29 rows) and the outlet-by-month record counts
behind it (696 rows, one per outlet-month cell).
The analytic columns of the inventory are rebuilt from `mention_labels` and
checked cell by cell; the raw monthly counts cover the fixed study window and
are not re-derivable from the released rows.

## Row counts at a glance

| File | Rows |
|---|---|
| `mention_labels.parquet` / `.csv` | 19,981 |
| `rater_votes.parquet` | 19,981 |
| `frame_rater_votes.csv` | 19,976 |
| `human_labels.csv` | 440 |
| `prompt_paraphrase_labels.csv` | 3,851 |
| `group_field_ablation.csv` | 1,940 |
| `mandate_intervals.csv` | 599 |
| `tenure_exclusions.csv` | 26 |
| `duplicate_title_groups.csv` | 130 |
| `outlet_inventory.csv` | 29 |
| `outlet_month_counts.csv` | 696 |
| `frideo_scores.csv`, `frideo_scores_no_d3_d7.csv` | 30 |
| `votes_alignment_leg17.csv` | 577 |
| `group_leaderboard.csv` | 11 |
| `mp_leaderboard.csv` | 57 |
| `group_non_attached.csv` | 1 |

## Deliberately excluded fields

The private analysis table also contains the source headline and free-text audit
notes. Both are excluded because they can reproduce copyrighted text. URLs, model
prompts containing source text, raw model responses, credentials and serving
metadata are likewise excluded from every file in this release.
