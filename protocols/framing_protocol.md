# Generic news-frame annotation protocol

This protocol defines how each French political headline is labelled for the
five generic news frames adapted from Semetko and Valkenburg (2000).

## Unit and analysis population

The annotation unit is one unique headline (`article_id`). Frames describe the
headline, not an individual deputy. A headline naming several deputies receives
one frame vector, which is joined to every `(headline, deputy)` pair sharing its
`article_id`.

The final release contains 19,981 pairs over 18,981 headlines. Frame analyses
use the 19,980 pairs over 18,980 headlines with a frame vector. The released
per-model frame-vote table contains 19,976 annotated headlines; agreement is
computed on the 19,975 with a usable vote vector.

Each frame is judged independently as present (`1`) or absent (`0`), so several
frames may apply to one headline. Annotators also select one dominant frame or
`none`. Decisions use only the headline text and do not infer a frame from a
deputy's identity, party, or outside knowledge. Tone and framing are separate:
evaluative language alone does not establish a frame.

## Frame definitions

1. **Conflict** indicates opposition, clash, rivalry, reproach, attack,
   contradiction, defeat, rupture, or sanction between identifiable actors or
   camps. The annotator must be able to name the opposing sides. A lone regret,
   criticism, or wish without an opposing actor is not conflict.
2. **Human interest** gives the story a personal or emotional face through an
   individual narrative, emotion, drama, suffering, or a personal vignette.
3. **Economic consequences** identifies financial gains or losses, costs,
   taxation, debt, purchasing power, employment, or other monetary consequences
   of action or inaction.
4. **Morality** contains an explicit moral or ethical judgment, religious
   reference, appeal to honour or shame, statement of values, or accusation of
   hypocrisy. Ordinary criticism or blame is not sufficient.
5. **Responsibility** assigns responsibility for a public problem, demands
   accountability, or advocates or rejects a policy or collective course of
   action. A personal opinion about an individual does not qualify.

## Decision rules

Policy prescriptions inside quotations count as responsibility. A regret about
an agenda without an opposing actor does not count as conflict. Accusatory
political quotations count as conflict when they identify opposing actors.
Personal assessments such as “one should not underestimate X” do not count as
responsibility. Ambiguous cases are resolved by three-model majority vote and
evaluated against the human audit sample.

## Annotation and output

Three models annotate every eligible headline at temperature 0. Majority vote
is computed separately for each frame. The released output fields are
`article_id`, the five binary `frame_*` fields, and `dominant`, whose permitted
values are `conflict`, `human_interest`, `economic`, `morality`,
`responsibility`, and `none`.
