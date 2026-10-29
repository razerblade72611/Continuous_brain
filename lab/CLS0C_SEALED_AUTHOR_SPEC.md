# CLS-0C sealed test: specification for the independent author (Grok session A)

This document is everything the author receives. The author is NOT given any information about the memory system's
prompts, its development results, or its known errors.

## What is being tested
A memory system stores **experiences**, which are short texts tagged with the life (1–5) in which they happened. It is
then given a **statement to complete**: an unfinished sentence such as `The role Liora held during the spring audit
was`. It must either:
- complete the statement with words copied **verbatim** from one experience, naming that experience; or
- **abstain** when no experience establishes the completion.

The test checks whether the system answers only what its experiences actually establish.

## Labels (definitions)
- **SUPPORTED:** one experience states the completion, so that a careful reader needs no assumption beyond the text.
  - The gold answer is a short span (1–4 words) that occurs verbatim in that experience.
  - Give the exact evidence sentence.
- **IMPLICATION:** the experiences make a completion plausible but do not establish it. The completion requires an
  assumption about preferences, intentions, hypothetical behaviour, causes or circumstances. Correct behaviour is to
  abstain.
  - Give the plausible-but-unsupported completion.
  - Say in one sentence why it is not established.
- **NEAR-MISS:** the person is mentioned in the experiences, but nothing at all is said about the asked property.
  Correct behaviour is to abstain.
- **UNFAMILIAR:** the person never appears in any experience. Correct behaviour is to abstain.
- **TEMPORAL:** the property changes across lives.
  - Statements that begin "At life N, …" ask for the state at life N.
  - Statements with no life prefix ask for the most recent state.
  - Gold answer: as for SUPPORTED, from the experience of the right life.

## Part 1: narrative worlds (10 worlds)
- **Characters:** each world is a small group of 3–6 invented people. Names must be invented and must not be real
  people or famous fictional characters.
- **Experiences:** 8–12 per world, spread over lives 1–5, 2–5 sentences each. They should read as natural
  autobiographical prose:
  - varied sentence structure;
  - incidental details;
  - several people per experience;
  - events that are described without stating a neat "attribute = value".
- **Include across the worlds:**
  - intentions that are announced but never confirmed;
  - claims made by one person about another;
  - two different descriptions of the same event;
  - roles or places that change across lives;
  - facts buried among unrelated details.
- **Statements to complete, per world:**

  | Label | Count |
  |---|---|
  | SUPPORTED | 10 |
  | IMPLICATION | 10 |
  | NEAR-MISS | 5 |
  | UNFAMILIAR | 5 |
  | TEMPORAL | 4 (at least one "At life N, …" and one unqualified) |

- **Form:** every statement is an unfinished sentence that ends exactly where the answer would begin. No question
  marks.
- **Optional:** also give each SUPPORTED item as an ordinary question, in a separate field.

## Part 2: new paraphrase statements for ten fixed attributes
The synthetic part of the test uses facts of the form "PERSON's ATTRIBUTE is VALUE". The ten attributes are: allergy,
birth month, childhood fear, city, favorite object, hobby, occupation, pet species, sister, workplace.

For EACH attribute, write templates using the placeholder `{e}` for the person, each ending where the value would
begin:
- **4 SUPPORTED templates:** different wordings that ask exactly for that attribute, with no extra assumption.
- **3 IMPLICATION templates:** wordings whose natural completion is the attribute's value, but which ask for something
  the fact does not establish (behaviour, frequency, intensity, other people's experience, and so on).

Avoid these wordings, which are already in use:
- "{e} is allergic to"
- "The month of {e}'s birthday is"
- "In childhood, {e} feared"
- "The name of the city where {e} resides is"
- "Nothing {e} owns is loved more than their"
- "What {e} likes to do in their spare time is"
- "The job {e} holds is that of a"
- "The animal {e} keeps as a pet is a"
- "The name of {e}'s sister is"
- "The place {e} goes to work is called"

## Output format (strict JSON, nothing else)
```json
{"worlds": [{"world_id": "W01",
             "experiences": [{"xid": "W01-X01", "life": 1, "text": "..."}],
             "items": [{"iid": "W01-Q01", "label": "SUPPORTED", "stem": "...", "answer": "...", "xid": "W01-X01",
                        "evidence": "...", "question": "..."},
                       {"iid": "W01-Q11", "label": "IMPLICATION", "stem": "...", "plausible": "...",
                        "why_not": "..."},
                       {"iid": "W01-Q21", "label": "NEAR-MISS", "stem": "..."},
                       {"iid": "W01-Q26", "label": "UNFAMILIAR", "stem": "..."},
                       {"iid": "W01-Q31", "label": "TEMPORAL", "stem": "...", "answer": "...", "xid": "W01-X07",
                        "evidence": "..."}]}],
 "attribute_templates": {"allergy": {"supported": ["..."], "implication": ["..."]}}}
```
