# The language organ (exploration lane)

**Status:** engineering result on `explore/w1-language`, not preregistered. Code: `continuous_brain_organs/`
(`language.py`, `language_eval.py`, `model_store.py`, `model_cards.json`). Scripts: `scripts/explore/
fetch_models.py`, `reassemble_model.py`, `eval_language_organ.py`. Tests: `tests/test_language_organ.py`
(16 tests, about 25 s). Evidence: `evidence/explore/language/dev_parser_eval.json` and the labelled corpus it
was computed on, `evidence/explore/language/w1_language_corpus.json.gz`.

The persistent individual hears W1's caretaker (`docs/explore/W1_WORLD.md`) only as English text. It will use
open-weight language models as an **organ**: a fixed function, pinned by weight hashes, that turns what it hears
into structure, and turns its own memory into speech. This document gives the interface, the event schema, the
two backends and how honest each one's numbers are, the model ladder, pinning, the commands for the owner's
machine, and what has and has not been measured.

## 1. Interface

```python
from continuous_brain_organs.language import LanguageOrgan
organ = LanguageOrgan.from_config({"backend": "dev"})          # or {"backend": "hf", "model_id": ..., ...}
events = organ.interpret(message, context)                     # -> list of events (validated)
text   = organ.answer(question_event, memory)                  # -> "answer q7: 5 east 2 north"
story  = organ.narrate(life_events)                            # -> first-person paragraph
organ.provenance()                                             # backend, model id, revision, per-file SHA-256
```

- **`interpret(message, context)`** turns one caretaker message into events. `context` is only what the
  individual itself knows: `step` (required, the step of the observation that carried the message) and,
  optionally, `inventory` (its inventory strip), `achievements` (its own counters), `player_pos` (its own
  position estimate) and `commitments` (`{id: event}`, its own earlier outputs). Never privileged state.
- **`answer(question_event, memory)`** composes W1's exact answer grammar from the caller's memory
  (`MEMORY_SCHEMA`: `position`, `last_seen{kind: {step, positions}}`, `counts_today{event: n}`,
  `requests[{id, issue_step}]`, `cancellations[{id, step}]`, optional `location_format` = `offset` | `bucket`).
  It is deterministic for every backend: the grammar is exact and the facts are the memory's, so a model would
  add only risk. It raises when the memory lacks what the question needs (it never guesses) and round-trips its
  own output through `continuous_brain_world.utterance.parse` before returning it.
- **`narrate(events)`** writes a short first-person narrative of a span of life. Input: any list of dicts with
  `type` and `step` (interpret events, plus e.g. `{"type": "utterance", "text": ...}` or `{"type": "note",
  "text": ...}`). The organ first turns them into factual first-person sentences (`narration_facts`); the dev
  backend joins them, the HF backend has the model rewrite them into one paragraph.
- **Division of labour.** A backend only *reads*: it returns a raw event (`RAW_EVENT_SCHEMA`) with the type, id,
  family and the fields stated in words, and the deadline phrase copied verbatim. The organ then *completes* it
  deterministically (`complete_event`): the absolute deadline step from the phrase with W1's daylight rules, the
  state-derived fields W1 verifies against (collect `base`/`target`, craft and place `base`, visit `start`) from
  the context, question windows and days, answer types. It also rejects raw events whose id is not the
  message's tag, whose phrase is not in the message, or whose deadline is not after the current step.
  Arithmetic and bookkeeping are never left to a model.
- **Provenance.** Every event carries `provenance = {backend, model_id, revision, weights_digest, call_id,
  attempts}`; `organ.provenance()` has the full record (per-file SHA-256 of the weights, library versions,
  decoding settings); `organ.records` (and `record_path`, JSONL) logs every call's input, output and latency.

## 2. Event schema (`cb-language-events/1`)

JSON Schema draft 2020-12, in `language.py` as `EVENT_SCHEMA` (one event), `EVENTS_SCHEMA` (a list) and
`RAW_EVENT_SCHEMA` (what a backend returns). Every output is validated; unknown keys are errors. Common fields of
every organ event: `type`, `message` (verbatim), `step`, `provenance`.

| type | fields | example message |
|---|---|---|
| `new_commitment` | `id` (`o\d+`), `family`, `value` (1..3 from the marker; 2 if none), `deadline_phrase` (verbatim), `deadline_step`, `spec` | "[o4] Top priority: make me a stone pickaxe within 380 steps." |
| `amendment` | `id`, `changes` ⊆ {`quantity`+`target`, `tool`+`base`, `threshold`, `deadline_phrase`+`deadline_step`} (only what changed; "same time frame" = no deadline change) | "[o3] Change of plan: I need five more logs after all (counting from when I first asked), by step 640." |
| `cancellation` | `id`, `reference` (the noun named, or null) | "[o8] Never mind the coal." |
| `question` | `id` (`q\d+`), `spec` (W1's own report spec), `answer_type` (`location_or_never` / `int` / `request_or_nothing`), `deadline_phrase`, `deadline_step` | "[q10] Which way, and how far, is the last cow you saw? (Answer within 120 steps.)" |
| `feedback` | `id`, `outcome` (`fulfilled` `late` `violated` `correct` `wrong` `no_answer`), `step_mentioned`, `revealed_answer` | "[q5] Not quite. The answer was: 5 east 2 north (east mid)." |
| `info` | — | the greeting (no tag) |
| `unparsed` | `error` | anything the backend could not turn into a valid event |

`spec` per family uses W1's own keys, so it compares directly with the layer's records:

| family | spec |
|---|---|
| collect | `item` (wood sapling stone coal iron diamond), `quantity`, `base`, `target` (= base + quantity) |
| craft | `tool` (wood/stone/iron × pickaxe/sword), `base` (the `make_<tool>` counter at issue) |
| place | `what` (table plant furnace stone) + either `near` (water / table) and `radius`, or `onto` (water); `base` |
| maintain | `stat` (food drink energy health), `threshold` |
| avoid | `source` (zombie arrow any) |
| visit | `feature` (water sand tree stone), `direction` (8-way), `stated_distance` ("about N tiles", else null), `start` |
| question | `last_seen{kind}`, `count_today{event, day}`, `first_request{window, which}`, `count_requests{window}`, `last_cancelled` |

`base`, `target` and `start` are null when the context does not provide them. W1's visit rule also uses
`min_dist`, computed from the exact (unstated) distance, and fixed 3-tile radii: those are not in the message
and the organ does not invent them. Deadline phrases resolve exactly as in W1 (`resolve_deadline`): "by/until
step N" → N; "within N steps"/"for the next N steps" → step + N; "before nightfall"/"before it gets
dark"/"until nightfall" → next nightfall − 1; "by (tomorrow) morning"/"until morning"/"through the night until
morning" → next dawn; "by noon" → next noon.

## 3. Backends and their honest status

### DevelopmentParser — a development stand-in, not a language model

A deterministic parser that **knows W1's caretaker templates**: it compiles the caretaker's own template strings
(imported from `continuous_brain_world.caretaker`), the eight change frames and the feedback frames into
anchored regular expressions, and refuses a message two templates would read differently. It exists so the
individual can run on this 2-CPU machine without weights, and as an upper reference for parsing accuracy on
W1's closed language. Its numbers say nothing about how a model, or the individual, would do on open language.

Measured (`evidence/explore/language/dev_parser_eval.json`, commit `f65696f`, corpus items_sha256
`5e2d128c…`):

| corpus | messages (distinct) | exact match | deadline step | answers grammar-valid / correct with oracle memory |
|---|---|---|---|---|
| main: default W1Config, seeds 72001–72020, 3000 steps | 959 (801) | 959 / 959 | 508 / 508 | 209 / 209 and 209 / 209 |
| stress: high-change caretaker, seeds 72101–72110, 900 steps | 1180 (1106) | 1180 / 1180 | 772 / 772 | 159 / 159 and 159 / 159 |

By event type (main / stress): new 324 / 306, amendment 35 / 362, cancellation 23 / 191, question 150 / 116,
feedback 407 / 195, info 20 / 10 — all exact. Latency about 0.5 ms per message. The test suite also runs it on
more than 1500 synthetic instances covering every template × synonym × deadline phrasing × value marker.

### HFBackend — a transformers causal LM

`HFBackend(model_id, revision, local_dir, device, dtype)` loads the model and tokenizer from `local_dir` with
`local_files_only=True` (no network, ever). Before loading, every file is hashed (`hash_weights="full"`) or
size-checked (`"lock"`, for multi-GB weights fetch_models already verified) and compared with
`models.lock.json` next to the model directory; a different file, an extra or missing file, or a different
revision refuses the load.

- Prompt: the model's **own chat template** over a system message (the raw schema, the value markers, the
  deadline phrases, the synonyms) + 16 few-shot user/assistant turns (`FEW_SHOT`: real caretaker message
  families, one per family and event type; the tests check the DevelopmentParser reads each of them exactly as
  given) + the message. About 2 000 tokens for Qwen-class tokenizers (unmeasured; `n_shots` shortens it).
- Decoding: greedy (`do_sample=False`, one beam, repetition penalty 1.0), `max_new_tokens=192`.
- Output: the first JSON array/object in the reply (code fences and prose tolerated) → raw schema → the organ's
  completion checks. On failure the reply and the error are appended to the conversation and the model is asked
  again, at most `max_attempts=3` times; then the event is `unparsed`.
- Audit: one record per attempt (prompt prefix SHA-256 + the suffix text, each distinct prefix stored once, the
  reply, prompt/new token counts, latency, ok/error), in `backend.audit` and optionally JSONL (`audit_path`).

**Status:** the plumbing is tested here without any download, on a tiny randomly initialised Qwen2 model (2
layers, hidden 32) with a byte-level BPE tokenizer trained offline on the organ's own prompt text and a ChatML
chat template: prompt construction, tokenisation, real greedy `generate`, decoding, failed extraction, the
retry with the error fed back, determinism, audit and provenance. Because a random model cannot be made to emit
valid JSON, one test **mocks `model.generate`** with scripted replies (garbage, then a non-verbatim deadline,
then valid JSON inside prose) to prove extraction, validation, retry, completion (identical to the
DevelopmentParser's event) and narration; everything around `generate` is real. **No real model has been run.**
Its accuracy, latency and memory use are not measured yet.

Device strings are those of torch: `cpu`, `cuda`, or `auto`. DirectML (`torch-directml`) is not supported by
HFBackend; at home use the CPU, on a rented pod use CUDA. The vision-language models in the cards are not
loadable by HFBackend (they need an image-text-to-text loader and a processor).

## 4. The model ladder

All facts read from the Hugging Face Hub API on 2026-10-04 (`model_cards.json`). Disk = what fetch_models
downloads (ONNX exports and Meta's `original/` checkpoint skipped).

| rung | model | role | params | licence | disk |
|---|---|---|---|---|---|
| 0 | HuggingFaceTB/SmolLM2-360M-Instruct | language | 0.36 B | Apache-2.0 | 0.73 GB |
| 0 | Qwen/Qwen2.5-0.5B-Instruct | language | 0.49 B | Apache-2.0 | 1.00 GB |
| 1 | Qwen/Qwen2.5-1.5B-Instruct | language | 1.54 B | Apache-2.0 | 3.10 GB |
| 2 | Qwen/Qwen2.5-3B-Instruct | language | 3.09 B | **Qwen Research License — non-commercial only, not Apache** | 6.18 GB |
| 3 | Qwen/Qwen2.5-7B-Instruct | language | 7.62 B | Apache-2.0 | 15.24 GB |
| 3 | meta-llama/Llama-3.1-8B-Instruct | language | 8.03 B | **Llama 3.1 Community License, GATED** (accept on the Hub, then HF_TOKEN) | 16.07 GB |
| VL 0 | HuggingFaceTB/SmolVLM-256M-Instruct | vision-language | 0.26 B | Apache-2.0 | 0.52 GB |
| VL 1 | Qwen/Qwen2-VL-2B-Instruct | vision-language | 2.21 B | Apache-2.0 | 4.43 GB |

Climb a rung only when the eval on the rung below says the organ is the bottleneck. Start with the two rung-0
models on the owner's CPU.

### Expected RAM and latency — ESTIMATES, NOT MEASURED

Weights in memory are parameters × 4 bytes in float32 (the CPU default) or × 2 bytes in bfloat16; add roughly
0.5–1.5 GB for torch, the tokenizer and a ~2 k-token KV cache. Latency per `interpret` call is dominated by the
~2 000-token prompt prefill plus ~60–120 generated tokens. The latency figures are order-of-magnitude guesses
for an 8-core desktop CPU and a 24 GB-class GPU; the eval report measures the real ones (`interpret_latency`).

| model | RAM, CPU float32 | RAM, CPU/GPU bfloat16 | per message, CPU (guess) | per message, GPU (guess) |
|---|---|---|---|---|
| SmolLM2-360M | ~2.0 GB | ~1.3 GB | 3–8 s | < 0.5 s |
| Qwen2.5-0.5B | ~2.5 GB | ~1.5 GB | 4–10 s | < 0.5 s |
| Qwen2.5-1.5B | ~7 GB | ~4 GB | 10–30 s | ~0.5–1 s |
| Qwen2.5-3B | ~13 GB | ~7 GB | 25–60 s | ~1 s |
| Qwen2.5-7B | ~31 GB | ~16 GB | 1–3 min (bf16) | ~1–2 s |
| Llama-3.1-8B | ~33 GB | ~17 GB | 1–3 min (bf16) | ~1–2 s |

Rungs 2–3 are impractical on a CPU at one message per W1 step; on CPU use `--dtype bfloat16` for them if RAM
is short (slower on CPUs without native bf16).

## 5. Pinning: trust on first use

`model_cards.json` leaves `revision` and `files_sha256` **null on purpose**. `fetch_models.py` fills them on the
first download, in `models.lock.json` next to the weights:

1. resolves the requested revision (default `main`) to its 40-hex commit SHA and downloads exactly that commit;
2. SHA-256-hashes every file and cross-checks the large (LFS) files against the Hub's own LFS SHA-256;
3. writes `{revision, files: {path: {size, sha256}}, ...}` for the model into the lock.

Every later run re-uses the locked revision (a different `--revision` is refused without `--repin`), re-hashes
every file and refuses on any difference (exit 2, lock untouched); `--verify-only` does that without network.
`HFBackend` checks the same lock on every load and records the revision and per-file hashes in the provenance
of every output. Moving weights through a channel with a per-file size limit: `--split-mb N` writes each file
as ≤ N-MiB chunks plus `split_manifest.json` (per-chunk and per-file SHA-256, revision);
`reassemble_model.py` verifies every chunk, rebuilds, verifies every file and the whole directory, and adds the
same pin to the destination lock (refusing if that lock pins the model differently).

## 6. Commands for the owner's machine (WSL Ubuntu-24.04)

The exploration checkout (`LANES.md`), never the protected one. Once `explore/w1-language` is on GitHub
(`razerblade72611/Continuous_brain`):

```bash
# from Windows PowerShell
wsl -d Ubuntu-24.04

# code
cd ~/projects/continuous_brain_explore
git fetch origin
git switch explore/w1-language
git pull --ff-only

# venv (once); the versions are the ones the tests ran with here
python3 -m venv ~/venvs/cb_explore
source ~/venvs/cb_explore/bin/activate
pip install --upgrade pip
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==5.17.0 tokenizers==0.23.2 safetensors==0.8.0 huggingface_hub==1.32.0 \
    jsonschema==4.26.0 crafter==1.8.3 numpy pytest

# tests (no network; about half a minute)
cd ~/projects/continuous_brain_explore
python -m pytest -q -p no:cacheprovider tests/test_language_organ.py

# download and pin the two small models (about 1.7 GB), then re-verify without network
python scripts/explore/fetch_models.py --list --models-dir ~/cb_models
python scripts/explore/fetch_models.py --set small --models-dir ~/cb_models
python scripts/explore/fetch_models.py --set small --models-dir ~/cb_models --verify-only

# measure Qwen2.5-0.5B on the CPU, on exactly the corpus the dev evidence used
# (150 messages of each corpus, about 300 calls; start with --max-messages 20 to see the speed)
mkdir -p runs/explore/language
python scripts/explore/eval_language_organ.py --backend hf --model Qwen/Qwen2.5-0.5B-Instruct \
    --local-dir ~/cb_models/Qwen--Qwen2.5-0.5B-Instruct --device cpu \
    --corpus-in evidence/explore/language/w1_language_corpus.json.gz --max-messages 150 --progress 10 \
    --audit-out runs/explore/language/qwen2.5-0.5b_cpu_audit.jsonl \
    --out runs/explore/language/eval_qwen2.5-0.5b_cpu.json

# the same for SmolLM2
python scripts/explore/eval_language_organ.py --backend hf --model HuggingFaceTB/SmolLM2-360M-Instruct \
    --local-dir ~/cb_models/HuggingFaceTB--SmolLM2-360M-Instruct --device cpu \
    --corpus-in evidence/explore/language/w1_language_corpus.json.gz --max-messages 150 --progress 10 \
    --audit-out runs/explore/language/smollm2-360m_cpu_audit.jsonl \
    --out runs/explore/language/eval_smollm2-360m_cpu.json

# hand the reports, audits and the lock back through the Windows exchange folder
mkdir -p /mnt/c/Users/davei/cb_exchange/language
cp runs/explore/language/eval_*_cpu.json runs/explore/language/*_audit.jsonl ~/cb_models/models.lock.json \
    /mnt/c/Users/davei/cb_exchange/language/

# moving weights through a channel with a per-file size limit (here: 300 MiB chunks)
python scripts/explore/fetch_models.py --model Qwen/Qwen2.5-0.5B-Instruct --models-dir ~/cb_models --split-mb 300
cp -r ~/cb_models/_split/Qwen--Qwen2.5-0.5B-Instruct /mnt/c/Users/davei/cb_exchange/
# ...and on the receiving machine (any checkout of this branch):
python scripts/explore/reassemble_model.py /mnt/c/Users/davei/cb_exchange/Qwen--Qwen2.5-0.5B-Instruct \
    --models-dir ~/cb_models
```

The gated model: accept the licence on https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct, create a read
token, then `export HF_TOKEN=hf_...` and `python scripts/explore/fetch_models.py --model
meta-llama/Llama-3.1-8B-Instruct --models-dir ~/cb_models`. On a rented CUDA pod: install the default (CUDA)
torch wheel (`pip install torch==2.14.0`), then use `--device cuda --dtype bfloat16`.

## 7. Measured, and not yet measured

Measured here:

- DevelopmentParser on 2 139 labelled W1 messages (959 default + 1 180 high-change): every event exact, every
  deadline step right, every composed answer grammar-valid and graded correct by W1's own grader **given an
  oracle memory**. 31 corpus messages are word-for-word identical to a few-shot example (reported per corpus;
  it matters for a model, not for the parser).
- HFBackend plumbing on a tiny random model (no download), with `generate` mocked in one test (section 3).
- fetch_models / reassemble_model lock, refusal, repin, LFS cross-check, split and reassembly, with the two
  network functions replaced by fakes; `--help` and no network import at import time.

Not measured yet:

- Any real model: interpret accuracy, retry rate, latency, RAM, on CPU or GPU. The CUDA path has never run.
- fetch_models against the real Hub (only fakes here; no Hub access from this machine), the gated download,
  huggingface_hub's behaviour on a repeated `snapshot_download` into the same directory.
- Narrative quality (one dev sample is in the report, unscored).
- Robustness to language outside W1's templates (W1 has none; see W1_WORLD.md section 10).
- The individual's own memory: answers are scored only with an oracle memory.

## 8. Known weaknesses

- The DevelopmentParser's perfect score is a property of W1's closed template language; it generalises to
  nothing else. Never quote it as a language model's number.
- No prefix KV-cache reuse: every call re-encodes the ~2 000-token system + few-shot prefix, the main CPU cost.
  No grammar-constrained decoding either (a JSON-schema logits processor would make invalid output impossible
  instead of retried). Greedy retries can repeat the same mistake.
- The prefix/suffix split of the audit assumes the chat template renders the first k messages as a prefix of
  the full prompt; if a template does not, the full prompt text is stored instead (no information is lost).
- An amendment's `target` needs the organ's own record of the original request (`context["commitments"]`); a
  missed original gives `target: null`.
- `jsonschema` is a runtime dependency that `pyproject.toml` (frozen) does not list; install it explicitly.
- `continuous_brain_organs/__init__.py` is docstring-only here; the vision branch adds its own docstring-only
  `__init__.py`, so the merge has one trivial add/add conflict in that file (keep either docstring).
