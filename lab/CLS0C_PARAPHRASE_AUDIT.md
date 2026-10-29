# CLS-0C paraphrase audit (frozen 2026-10-29, Claude + Sol, before any Phase B result)

Each held-out paraphrase in `being/v3/lifegen.py` `PARAPHRASE` was judged against all four stored phrasings in `TRAIN`.
The question: does the experienced text **establish** what the paraphrase asks? This test is about what the memory
supports, not what is plausible, so the labels are deliberately conservative.

**How the labels are used:**
- Primary paraphrase accuracy in CLS-0C counts ENTAILED paraphrases only.
- BORDERLINE and NOT ENTAILED paraphrases are reported separately and not gated. Both answering with the stored value
  and abstaining are recorded; neither is penalised.
- CLS-0B's historical scores are unchanged.

| Attribute | Paraphrase | Label | Reason |
|---|---|---|---|
| allergy | p0 "The substance {e} cannot be near because of an allergy is" | NOT ENTAILED | an allergy does not establish that mere proximity is unsafe (Sol) |
| allergy | p1 "{e} has to avoid" | NOT ENTAILED | introduces an avoidance requirement |
| birth month | p0 "The month of {e}'s birthday is" | ENTAILED | |
| birth month | p1 "{e} turns a year older every year in the month of" | ENTAILED | |
| childhood fear | p0 "The thing that scared {e} most as a young child was" | BORDERLINE | introduces an intensity ranking |
| childhood fear | p1 "In childhood, {e} feared" | ENTAILED | |
| city | p0 "If you visited {e} at home, you would be in" | ENTAILED | |
| city | p1 "The name of the city where {e} resides is" | ENTAILED | |
| favorite object | p0 "The object {e} would save first from a fire is a" | NOT ENTAILED | a favorite does not establish emergency behaviour |
| favorite object | p1 "Nothing {e} owns is loved more than their" | ENTAILED | stored: "the thing {e} treasures most" |
| hobby | p0 "What {e} likes to do in their spare time is" | ENTAILED | |
| hobby | p1 "The activity {e} most enjoys outside work is" | BORDERLINE | introduces a comparative claim |
| occupation | p0 "Asked what they do for a living, {e} would answer that they are a" | ENTAILED | |
| occupation | p1 "The job {e} holds is that of a" | ENTAILED | |
| pet species | p0 "The animal {e} keeps as a pet is a" | ENTAILED | |
| pet species | p1 "Visitors to {e}'s home meet their pet, a" | NOT ENTAILED | owning a pet does not establish that visitors meet it (Sol) |
| sister | p0 "The name of {e}'s sister is" | ENTAILED | |
| sister | p1 "If you asked {e} who their sister is, they would say" | ENTAILED | |
| workplace | p0 "The place {e} goes to work is called" | ENTAILED | |
| workplace | p1 "If you wanted to find {e} at work, you would go to" | ENTAILED | |

**Totals:** 14 ENTAILED, 2 BORDERLINE, 4 NOT ENTAILED.

**Consequence:** allergy facts are tested only through their template probe on the primary paraphrase measure.
Machine-readable copy: `AUDIT` in `scripts/cls0c_b.py` (unit-tested against this table).
