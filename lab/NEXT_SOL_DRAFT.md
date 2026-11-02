# Sol's independent next-design draft

Status: independent design proposal, not an approved protocol or launch instruction. Basis: closed ABST-0 evidence through `cf21cf988546a461db72b9ddf30dc7099cf5a97e`. Written without reading Claude's next-design draft. Freeze this file before exchanging proposals; reconcile the two drafts only afterward. Progress updates disclosed the dose/coverage observations and a broad supervision-strength hypothesis before Claude's hash was verified here. Sol has not read Claude's proposal; verify commitment ordering before claiming that Claude was also blind to those updates.

## 1. What we are trying to build

The primary goal remains a single persistent learner whose acquired experiences make a durable, causally useful difference to its knowledge, choices, and subsequent learning. Its history should accumulate into a coherent individual rather than a sequence of sessions reconstructed by a prompt. Accurate autobiographical binding, retention, learning from new experience, and distinguishing supported knowledge from invention are components of that goal. None alone establishes consciousness, and this experiment does not measure consciousness.

ABST-0 asked whether one component could emerge through cortical consolidation: answer supported questions, abstain on unsupported ones, and later learn a formerly unsupported fact without damaging recall or general competence. The externally supplied verbatim archive removed episodic reconstruction as a confound. The fresh-process fp32 checks removed the earlier merge-equivalence ambiguity. This was a useful component test, not an integrated episodic-store demonstration.

It closed cleanly as NO_ADVANCE: factual acquisition/retention survived, but the required support-sensitive boundary did not emerge. That is informative. It does not establish that a persistent individual is impossible, that this cortex cannot ever learn a support boundary, or that its small observed UNKNOWN effect contains no grounding. The U–G result failed the preregistered grounding test; four development seeds with a weak intervention are not an equivalence proof about all mechanisms.

My recommendation is one bounded, diagnostic attempt to make the support-learning intervention identifiable. Do not spend the remaining budget on unstructured abstention tuning. If a clean, strongly supervised boundary still fails, change the representation/interface rather than keep increasing repetition.

## 2. Three distinctions in the shared premises

1. **Audit quality:** 20/231 = 8.7% were flagged **supported or ambiguous**. Clearly supported items were 8/231 = 3.5%; ambiguous items were 12/231 = 5.2%. Treating every ambiguous item as definitely answerable overstates the evidence. Delivered flagged exposure mass was roughly 10%, with seed/sleep variation. That percentage alone does not bound its causal effect on learning.
2. **Unique QA versus delivered supervision:** across the 20 seed/sleep records there were 482 new QA items, presented 1,928 times at E=4, plus 1,928 replay/background items, 482 A-negatives, and 482 T-negatives: 4,820 delivered items. Negatives were 20% of item presentations, split 10% A and 10% T. In a representative 27-new-QA sleep, the schedule was 108 new presentations + 108 replay/background + 27 A + 27 T, not 27 new presentations against 54 negatives. Item proportions still do not establish loss-token or gradient proportions.
3. **Direction versus size:** at sleep 5 U had lower HNM confabulation than G in 3/4 seeds, but the pooled reduction was only 0.04375. The size condition failed. This is weak positive directional evidence, not a useful demonstrated grounding benefit.

An additional coverage concern is visible without any new run: 54/231 audited questions literally ask for a favorite color and 13 ask about siblings. Those two topics alone account for 29.0% of the audit and are outside the nine non-relational evaluation attributes. This is an audit-sample observation, not a delivered-mass estimate or a complete semantic annotation. Nevertheless, nominally holding the evaluation attribute domain fixed did not guarantee that negative supervision covered that domain.

## 3. Leading hypotheses and discriminating evidence

These mechanisms can coexist. The table orders my current priorities, not established probabilities.

| Hypothesis | Why it remains plausible | Observation that separates it |
|---|---|---|
| Insufficient effective UNKNOWN supervision relative to positive/replay learning | Only 20% of delivered items were negatives; broad questions disperse that budget. UNKNOWN initially dominates N, then positive training overwhelms it. | At fixed positive exposures and total update budget, increasing only clean negative-versus-background exposure improves exact trained negatives first, then held-out support discrimination. |
| Wrong coverage or construction, rather than failure to learn UNKNOWN itself | Many audited negatives ask outside the evaluation attribute domain; Question-to-Statement is an additional holdout. | Exact supervised Question probes succeed while matched missing-cell Statement probes fail; or clean in-domain supervision transfers when broad free-generation did not. |
| Generic answer suppression without an experience-specific boundary | U and name-swapped G had nearly the same weak effect. | U and G improve similarly, including on unseen names; U has no additional advantage on held-out familiar entities once dose and wording are matched. |
| The cortical representation/readout does not expose a usable support state | Fact values can be recalled while absence of experience is not represented in a form this objective can use. | Clean, high-dose U fits its exact A-negatives but still cannot generalize to H missing cells or respond correctly to controlled history changes. This is stronger evidence for an interface/representation change than ABST-0 alone. |
| Incorrect/ambiguous targets poison the intended distinction | The audit found real answerable negatives, though not 8.7% all-definitely-answerable. | A later matched label-flip intervention changes the boundary. A clean-task rescue alone would not isolate label noise from coverage and construction changes. |

The lack of accumulation across sleeps is compatible with repeatedly relearning a weak generic constraint while positive/replay supervision continues to reinforce answering. It is also compatible with a transfer failure. It does not by itself show that consolidation forgot a previously acquired support rule. We need exact supervised-item acquisition and before/after trajectories to distinguish those statements.

## 4. Best next experiment: an oracle-defined support curriculum, grounding by dose

Working name: **SUPPORT-1**. Development experiment. Allocate four unused seeds before freezing; use 20–23 only if the shared experiment registry confirms they are unused. Keep the base model, cortical rank, optimizer, competence monitors, five-sleep structure, decoding and numerical-integrity requirements fixed unless a separately reviewed implementation necessity requires otherwise.

### 4.1 A deliberately narrow, exact teacher

Use synthetic episode streams with the same nine attributes and explicitly defined acquired cell `(entity, attribute)` histories. The teacher is a deterministic program over those histories:

- If a cell has been observed, its target is its current value.
- If the cell has not been observed, its target is UNKNOWN. This means unsupported by that checkpoint's experience, not false in the world.
- Updates select the latest acquired value. Future episodes never enter teacher labels.

Use fixed, independently reviewed query/answer templates, not generated QA or free-generated negatives. Positives and negatives share the same attribute-specific query grammar. Every allowed missing attribute is represented; no favorite-color/sibling detour. This removes generator/verifier uncertainty and gives an exact support curriculum, but it is external supervision. Success would establish learnability under that supervision, not autonomous discovery of an epistemic rule.

Split core entities into A and H, balancing introductions and attribute frequencies. Only A entities can receive UNKNOWN training. H receives factual experiences/positive consolidation but no UNKNOWN target, in any arm. Keep distinct fresh-name pools T (training only) and V (evaluation only), disjoint by name word. Balance value reuse and known/unknown probe attributes so a global attribute prior cannot solve the task. Independently validate cell masks, name-pool disjointness and answer uniqueness.

Keep Question as the training construction and Statement as the primary transfer construction. Also evaluate a small, frozen diagnostic set in the exact Question format. Report matched trained A-negatives, held-out A missing cells, H missing cells, and V fresh names separately. A successful exact-question test followed by a failed Statement test is a construction failure; do not call it failure to learn UNKNOWN.

This replaces several historical scaffolding choices at once. Therefore a rescue cannot be attributed specifically to removing 8.7% flagged targets. The causal contrasts inside SUPPORT-1 are grounding and relative dose; comparison with ABST-0 is descriptive feasibility evidence.

### 4.2 Five matched arms

Let n be the number of new positive training items in a sleep. Keep E=4 for positive exposure. Every arm receives 4n new positive presentations and 4n identical replay/background presentations. Fix an additional 8n third-slot positions in all arms, giving 16n items total:

| Arm | Third-slot UNKNOWN items | Remaining third-slot content |
|---|---:|---|
| P | 0 | 8n matched background items |
| U1 | 2n: n A + n T | 6n matched background items |
| G1 | 2n, same positions as U1; A names swapped to T | 6n matched background items |
| U4 | 8n: 4n A + 4n T | none |
| G4 | 8n, same positions as U4; A names swapped to T | none |

U1/U4 use oracle-unsupported A cells. In G1/G4 those questions have a fresh T name of matched framed token length; T-half items are unchanged. Use identical selected item order within each dose, with deterministic repetitions. No arm's output enters selection.

Match item, input-token and loss-token budgets across all five arms, including the low-dose background fillers. Keep positive exposures and their positions identical. Record actual delivered per-cell counts and loss-token counts. The intervention is explicitly **replacement of matched background with UNKNOWN supervision**, not a claim to alter only a mathematical gradient scalar. All arms share the same added background/negative capacity, so total training steps do not explain their contrast.

Increasing E alone would increase positive learning and UNKNOWN exposure together; it is not the clean dose intervention we need. Do not also change rank, optimizer, decoding, memory architecture, or temperature in this experiment.

### 4.3 Guaranteed reversal cohort, with an acquisition check

Reserve four A and four H transition cells, with the same introduction balance as ABST-0. They acquire their first value at sleep 4.

- Every A transition cell must receive at least four explicit UNKNOWN presentations at each of sleeps 1–3, in U1 and U4, from its known exact cell/query construction. Give the remaining eligible A cells exposure as well; freeze the allocation before running and ensure the low-dose quota is sufficient.
- If an entity has not yet been introduced, report that exposure as unfamiliar; guarantee the familiar exposures after introduction and before sleep 4. Do not silently treat unfamiliar-name learning as the familiar transition test.
- H transition cells receive no UNKNOWN target. This is checked mechanically against every delivered item.
- Stop assigning UNKNOWN to an A transition cell as soon as its episode arrives; replace its slot with another eligible negative, identically across the paired arm schedule.

Report delivered exposure directly, not a verifier's guess that a broad question might concern the later fact. Also require a measured pre-transition UNKNOWN acquisition criterion on the exact trained A questions before calling the later result a test of **learned** UNKNOWN reversal. Delivered UNKNOWN supervision without learned abstention is not successful pre-transition acquisition.

Track both familiar TPRE at sleeps 2–3 and TPOST at sleeps 4–5. Separate A-exposed and H cohorts. If pre-transition UNKNOWN was not learned, the reversal endpoint is not established; still report the failure rather than redefining the cohort after seeing results.

### 4.4 Small paired-history fork

At sleep 3, save checkpoint and optimizer state for U4 and G4 in each seed. The ordinary sleep-4 continuation teaches the reserved H transition facts. One additional, matched control continuation withholds those H facts and teaches the same number of comparable distractor facts elsewhere. H target cells receive no UNKNOWN supervision in either continuation. Hold A-negative selection, budget, construction and all other training inputs fixed between the two continuations as far as the target intervention permits.

Evaluate the **same target questions** after the two histories. In the observed-history branch they should be answered with the new value; in the withheld-history branch they should remain UNKNOWN. Randomize/counterbalance target-to-distractor assignment before training. The replication unit is the seed's paired branch, not each probe or each branch separately.

This tests causal sensitivity to receiving the fact while controlling age and starting parameters. It does not by itself prove grounding specific to U; G can also learn new positive facts. The useful evidence is the paired known/unknown tradeoff together with U4 versus G4's held-out missing-cell behavior. Do not equate ordinary fact acquisition with a support mechanism.

## 5. Readouts, frozen decisions, and stopping rules

Before a paid run, write a complete protocol with exact templates, allocation, exposure quotas, sampler keys, normalization, counts and statistical unit. Extend both analyzers independently. Unit tests must force the reversal cohort nonempty, reject shared forged gold, detect malformed targets, and check every comparison's expected checkpoint identity. The closed ABST-0 analyzers and outcomes remain unchanged.

For each arm, seed and sleep report numerator/denominator and equal-weight seed means for correct, substantive-incorrect, exact UNKNOWN and empty; answer rate and selective accuracy jointly; acquisition templates, held-out constructions, retention, transition cohorts, updates and both competence ratios. Include exact trained-negative loss/log-prob and decoded acquisition on the fixed Question diagnostics. Log actual exposure mass, not just candidate counts.

Use the existing functional standards for the primary **U4** claim: D1 known/first acquisition, D2 HNM and FRESH with confabulation <=0.10 and UNKNOWN >=0.90, D3 familiar TPRE >=0.80 plus TPOST recall and P-relative constraints, and D4 retention/competence. Keep all sleeps and the separate A/H requirements. Do not lower thresholds to make a cleaner task pass.

The primary mechanism contrast is U4 versus G4 at sleep 5: HNM confabulation reduction >=0.30 pooled and U4 lower in at least three of four seeds, conditional on valid arms and functional success. Report U1/G1 and all earlier contrasts regardless of outcome. The low/high dose contrast is a preregistered diagnostic, not a selector for the headline arm after seeing results. The fork and exact-Question diagnostics are additional mechanistic readouts; freeze their success rules before running, rather than inventing a combined score afterward.

Interpretation table:

- **Both U1 and U4 fail exact trained negatives:** investigate target loss, masking, optimization or support capacity. Do not claim a held-out grounding failure from an unlearned training rule.
- **U4 fits exact negatives but misses Statement aliases:** construction transfer is the bottleneck. A later diagnostic may train both constructions while holding out different aliases; it is a new experiment.
- **U4 and G4 both improve strongly and similarly:** generic contract/answer suppression is sufficient for that improvement. Functional progress may exist, but experience-grounding benefit is not demonstrated.
- **U4 exceeds U1, and U4 exceeds G4 while preserving positives:** relative dose enables an experience-sensitive benefit on this clean curriculum. Return to noisy/distilled supervision only in a later matched experiment.
- **U4 fits A queries but fails H and the paired-history boundary despite intact positive learning:** stop cortical abstention-only tuning. Next investigate an explicit learned support state or episode/provenance representation, with retrieval and oracle-mask controls separated from learned ability.
- **High dose damages acquisition or competence:** the tradeoff is a failure under the functional criteria; do not report increased UNKNOWN alone as progress.

Stop only for frozen integrity, competence/calibration, data-construction or budget violations, never because an interim scientific score is disappointing. All seeds and all arms remain reported. Hardware failures follow the existing preserve-before-removal and resume-provenance rules. David decides and launches.

## 6. Alternatives I would not choose first

- **Repeat ABST-0 with more epochs only:** keeps the same relative supervision, coverage ambiguity and empty reversal cohort. It would not isolate the leading cause.
- **A broad learning-rate/rank/dose search:** consumes the budget and makes a winning configuration hard to interpret. One fixed fourfold negative-versus-background contrast is enough for the next diagnostic.
- **Immediately add a prompted episodic retriever or an external known-cell mask as the solution:** those are useful engineering baselines for the eventual integrated learner, but they can implement the answer/abstain decision outside the cortical learner. They do not answer this component question. An explicit-memory route is the planned fallback if the clean experiment fails.
- **Immediately blame or repair only the flagged generated negatives:** no causal attribution is available, and removing 20 sampled candidates would be selective repair. A quality-only study would need matched, randomized label corruption on a controlled corpus.
- **Train both Question and Statement immediately:** may be a useful later engineering step but removes the current transfer diagnostic before we know whether construction is the limiting factor.
- **Start a larger brain, agent swarm, or consciousness demonstration:** ABST-0 did not test those alternatives. Adding mechanisms now would make the observed component failure harder to understand.

## 7. Budget: one L40S, approximately $90 remaining

Use David's planning rate of about $10 per 12 hours; these are estimates, not a live price quote.

| Work | Planning allocation | Rule |
|---|---:|---|
| Closed-artifact accounting and protocol/template tests | CPU; at most $1 of GPU plumbing if needed | No scientific model sweep |
| Four seeds, five arms, five sleeps; primary and twin evaluations | about $16–21 | 200 trained evaluations + 20 N evaluations |
| U4/G4 paired-history control continuations | about $1–3 | 16 additional evaluations; ordinary branches already counted |
| Overrun/recovery allowance | remainder within $30 total ceiling | Retain evidence; do not loosen checks |
| Reserve after this proposal | at least $60 | For replication/integrated-representation work, not automatic spending |

The main schedule is 220 evaluations, plus 16 fork evaluations: 236 versus ABST-0's 140. Scaling the approximately 10–12-hour reference gives roughly 17–20 hours before extra diagnostic probe and training overhead. Limit exact-Question diagnostics to a frozen small set, e.g. <=64 rows per primary evaluation, rather than duplicating every probe. Treat approximately 20–28 total hours as the working envelope and **$30 as the proposed total spending ceiling**. David decides whether to fund and launch it; this draft authorizes no expenditure.

Before launching the full matrix, measure throughput on a fixed plumbing subset with no outcome-dependent choices. If the complete planned run cannot fit the ceiling, revise the protocol before inspecting scientific scores: first reduce diagnostic probe volume or postpone the paired-history fork; do not drop unsuccessful arms or seeds. Do not begin a partial matrix and later pretend it is the preregistered complete experiment.

This spends at most one third of the remaining budget on a discriminating clean test. A positive result earns a replication/transfer study; a negative result earns a representation change. Neither result earns another open-ended hyperparameter search by default.
