# Target-directed sentiment and role annotation protocol

This protocol defines how a French political headline is labelled for sentiment
toward a named deputy and for that deputy's discursive role.

## Unit of annotation

The unit is one `(headline, target deputy)` pair. A headline naming two deputies
produces two items, each judged independently for its own target.

The sentiment question is: reading the headline as a French news reader, does
the headline place the target deputy in a positive, negative, or neutral light?
The judgment concerns valence toward the target, not the overall mood of the
sentence.

## Sentiment labels

- **Positive:** the headline casts the target favourably through success,
  vindication, praise, or sympathy.
- **Negative:** the headline casts the target unfavourably through scandal,
  failure, accusation, ridicule, or defeat.
- **Neutral:** the mention is factual, procedural, balanced, or evaluative
  language is directed at someone or something other than the target.

## Role labels

- **Source:** the deputy speaks, accuses, demands, proposes, defends, warns, or
  introduces a quotation. The content of the deputy's statement does not by
  itself determine sentiment toward the deputy.
- **Patient:** the deputy receives an action or evaluation, such as being
  accused, investigated, convicted, criticised, or defeated.
- **Agent:** the deputy performs a substantive non-speech action. Sentiment is
  determined by how that action reflects on the deputy.

Role is assigned before sentiment because it distinguishes evaluative language
about the deputy from language merely attributed to the deputy.

## Decision rules

Sarcasm and scare quotes count as their implied valence. Purely procedural
mentions are neutral. Coalition or party blame transfers to the deputy only when
the headline names the deputy as the locus of that blame. Conflict alone does
not make every named participant negative. When valence is genuinely not about
the target, the label is neutral.

## Annotation and output

Three models annotate every eligible pair at temperature 0. Majority vote is
computed separately for sentiment and role, and the final labels are evaluated
against the blind human audit sample. Released fields include `pair_id`,
`label(pos/neg/neu)`, and `role(source/patient/agent)`; per-model votes are
released in `data/rater_votes.parquet`.
