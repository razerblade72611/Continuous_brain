"""Language organ: event schema, the DevelopmentParser against W1's templates and the live layer, deadline
arithmetic, answers through W1's own parser/grader, the HF backend's plumbing on a tiny offline model, and the
model fetcher's lock/split logic on fake local directories (no network anywhere)."""
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from continuous_brain_organs import language as L
from continuous_brain_organs import language_eval as E
from continuous_brain_organs import model_store as ms
from continuous_brain_world import daylight as dl
from continuous_brain_world import caretaker as C
from continuous_brain_world.obligations import grade_report
from continuous_brain_world.utterance import parse

REPO = Path(__file__).resolve().parents[1]
STRESS = {"arrival_rate0": 1 / 15, "arrival_rate_max": 1 / 10, "change_rate": 1 / 40, "difficulty0": 1.0}


def load(rel):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dev():
    return L.LanguageOrgan.from_config({"backend": "dev"})


# --- schema ----------------------------------------------------------------------------------------------
def test_schemas_are_well_formed_and_strict(dev):
    import jsonschema
    for s in (L.RAW_EVENT_SCHEMA, L.EVENT_SCHEMA, L.EVENTS_SCHEMA, L.NARRATE_INPUT_SCHEMA, L.MEMORY_SCHEMA):
        jsonschema.Draft202012Validator.check_schema(s)
    ctx = {"step": 120, "inventory": {"wood": 2}, "achievements": {"make_stone_pickaxe": 0}, "player_pos": [3, 4]}
    good = dev.interpret("[o1] Please gather 3 wood before nightfall.", ctx)[0]
    assert L.schema_errors([good], L.EVENTS_SCHEMA) == []
    bad = []
    for mutate in (lambda e: e.update(family="juggle"), lambda e: e.update(id="x1"), lambda e: e.update(extra=1),
                   lambda e: e.pop("deadline_step"), lambda e: e["spec"].update(tool="wood_pickaxe"),
                   lambda e: e["spec"].pop("target"), lambda e: e.update(value=4),
                   lambda e: e["provenance"].pop("backend")):
        e = copy.deepcopy(good)
        mutate(e)
        bad.append(L.schema_errors([e], L.EVENTS_SCHEMA))
    assert all(bad), bad
    am = dev.interpret("[o3] Change of plan: I need 4 logs after all (counting from when I first asked), by step 640.",
                       {"step": 300})[0]
    assert L.schema_errors([am], L.EVENTS_SCHEMA) == []
    e = copy.deepcopy(am)
    e["changes"].pop("deadline_step")                       # a phrase without its resolved step is invalid
    assert L.schema_errors([e], L.EVENTS_SCHEMA)
    e = copy.deepcopy(am)
    e["changes"] = {}
    assert L.schema_errors([e], L.EVENTS_SCHEMA)
    # completion refuses a raw event whose phrase is not verbatim or whose id is not the tag
    raw = {"type": "new_commitment", "id": "o1", "family": "collect", "value": 2, "deadline_phrase": "by noon",
           "spec": {"item": "wood", "quantity": 3}}
    with pytest.raises(L.OrganError, match="verbatim"):
        L.complete_event(raw, "[o1] Please gather 3 wood before nightfall.", {"step": 10})
    with pytest.raises(L.OrganError, match="not a recognised time phrase"):
        L.complete_event({**raw, "deadline_phrase": "before night"}, "[o1] Please gather 3 wood before nightfall.",
                         {"step": 10})
    with pytest.raises(L.OrganError, match="tag"):
        L.complete_event({**raw, "id": "o2", "deadline_phrase": "before nightfall"},
                         "[o1] Please gather 3 wood before nightfall.", {"step": 10})
    with pytest.raises(L.OrganError, match="not after"):
        L.complete_event({**raw, "deadline_phrase": "by step 5"}, "[o1] Please gather 3 wood by step 5.", {"step": 10})


def test_few_shot_examples_are_what_the_parser_reads():
    p = L.DevelopmentParser()
    for msg, raw in L.FEW_SHOT:
        assert L.schema_errors(raw, L.RAW_EVENT_SCHEMA) == [], msg
        assert p.parse_raw(msg) == [raw], msg
    fams = {r.get("family") for _, r in L.FEW_SHOT if r["type"] == "new_commitment"}
    assert fams == set(L.FAMILIES)
    assert {r["type"] for _, r in L.FEW_SHOT} == set(L.EVENT_TYPES) - {"unparsed"}


# --- the development parser on every caretaker template ---------------------------------------------------
TASK_PHRASES = ["by step 812", "within 380 steps", "before nightfall", "before it gets dark", "by tomorrow morning",
                "by morning", "by noon"]
UNTIL_PHRASES = ["until step 900", "for the next 450 steps", "until nightfall", "through the night until morning",
                 "until morning"]


def caretaker_message(oid, text, value):
    """Exactly what ObligationLayer.add_obligation makes of a caretaker text (value marker + lower-casing)."""
    marks = [m for m in C.VALUE_MARK[value]]
    mark = marks[(int(oid[1:]) * 7) % len(marks)]
    body = text
    if mark:
        first = body.split(" ", 1)[0]
        body = mark + (body if first in ("I", "I'd") else body[0].lower() + body[1:])
    return f"[{oid}] {body}"


def qty_phrases(item, n):
    out = []
    for sing, plur in C.ITEM_NOUNS[item]:
        if n == 1:
            arts = ["an" if sing[0] in "aeiou" else "a", "one"] if sing != plur else ["one", "1"]
            out += [f"{a} {sing}" for a in arts]
        else:
            for w in (C.NUMBER_WORDS[n], str(n)):
                out += [f"{w} {plur}", f"{w} more {plur}"]
    return out


def test_development_parser_reads_every_template_family():
    p = L.DevelopmentParser()
    n_checked = 0
    i = 0

    def check(text, raw_expect, value=2):
        nonlocal n_checked, i
        i += 1
        oid = raw_expect.get("id_prefix", "o") + str(i)
        msg = caretaker_message(oid, text, value) if raw_expect["type"] == "new_commitment" else f"[{oid}] {text}"
        expect = {k: v for k, v in raw_expect.items() if k != "id_prefix"}
        expect["id"] = oid
        if expect["type"] == "new_commitment":
            expect["value"] = value
        got = p.parse_raw(msg)
        assert got == [expect], (msg, got)
        n_checked += 1

    for t in C.T_COLLECT:
        for item in C.ITEM_NOUNS:
            for n in (1, 2, 9):
                for q in qty_phrases(item, n):
                    for k, ph in enumerate(TASK_PHRASES):
                        if (n + k + len(q)) % 3:           # a deterministic third of the full product
                            continue
                        check(t.format(q=q, dl=ph), {"type": "new_commitment", "family": "collect",
                                                     "deadline_phrase": ph, "spec": {"item": item, "quantity": n}},
                              value=1 + (n + k) % 3)
    for t in C.T_CRAFT:
        for tool, names in C.TOOL_NAMES.items():
            for name in names:
                for ph in TASK_PHRASES:
                    check(t.format(t=name, dl=ph), {"type": "new_commitment", "family": "craft",
                                                    "deadline_phrase": ph, "spec": {"tool": tool}}, value=3)
    for (what, rel, mat), ts in C.T_PLACE.items():
        for t in ts:
            for r in (0, 2, 4):
                for ph in TASK_PHRASES:
                    spec = {"what": what, rel: mat, **({"radius": r} if rel == "near" else {})}
                    check(t.format(r=r, dl=ph), {"type": "new_commitment", "family": "place", "deadline_phrase": ph,
                                                 "spec": spec}, value=1)
    for t in C.T_MAINTAIN:
        for stat, names in C.STAT_NAMES.items():
            for name in names:
                for thr in (2, 6):
                    for ph in UNTIL_PHRASES:
                        check(t.format(s=name, n=thr, u=ph), {"type": "new_commitment", "family": "maintain",
                                                              "deadline_phrase": ph,
                                                              "spec": {"stat": stat, "threshold": thr}})
    for src, ts in C.T_AVOID.items():
        for t in ts:
            for ph in UNTIL_PHRASES:
                check(t.format(u=ph), {"type": "new_commitment", "family": "avoid", "deadline_phrase": ph,
                                       "spec": {"source": src}}, value=3)
    for t in C.T_VISIT:
        for feat, names in C.FEATURE_NAMES.items():
            for name in names:
                for d in L.DIR8:
                    ph = TASK_PHRASES[len(name + d) % len(TASK_PHRASES)]
                    stated = 15 if "{n}" in t else None
                    check(t.format(f=name, d=d, n=15, dl=ph),
                          {"type": "new_commitment", "family": "visit", "deadline_phrase": ph,
                           "spec": {"feature": feat, "direction": d, "stated_distance": stated}})
    tail = " (Answer within 120 steps.)"
    q = {"id_prefix": "q", "deadline_phrase": "within 120 steps"}
    for t in C.T_LAST_SEEN:
        for kind, phrase in C.KIND_PHRASES.items():
            check(t.format(k=phrase, k2=phrase.split(" ", 1)[-1]) + tail,
                  {"type": "question", "question": "last_seen", "kind": kind, **q})
    for event, texts in C.COUNT_EVENTS.items():
        for t in texts:
            check(t + tail, {"type": "question", "question": "count_today", "event": event, **q})
    for which, texts in C.T_FIRST.items():
        for t in texts:
            check(t + tail, {"type": "question", "question": "first_request", "which": which, **q})
    for t in C.T_COUNT_REQ:
        check(t + tail, {"type": "question", "question": "count_requests", **q})
    for t in C.T_LAST_CANCEL:
        check(t + tail, {"type": "question", "question": "last_cancelled", **q})
    for t in C.T_CANCEL:
        for k in ("wood", "stone pickaxe", "trip", "zombies", "health", "table"):
            check(t.format(k=k), {"type": "cancellation", "reference": k if "{k}" in t else None})
    for k, ph in enumerate(TASK_PHRASES):
        n = 1 + k % 9
        check(L.CHANGE_TEMPLATES["collect_quantity"].format(q=qty_phrases("stone", n)[0], dl=ph),
              {"type": "amendment", "changes": {"quantity": n, "deadline_phrase": ph}})
        check(L.CHANGE_TEMPLATES["craft_tool"].format(t="wooden sword", dl=ph),
              {"type": "amendment", "changes": {"tool": "wood_sword", "deadline_phrase": ph}})
        check(L.CHANGE_TEMPLATES["craft_deadline"].format(t="iron pickaxe", dl=ph),
              {"type": "amendment", "changes": {"deadline_phrase": ph}})
        for noun in ("table", "sapling", "furnace", "stone", "trip"):
            check(L.CHANGE_TEMPLATES["deadline_now"].format(noun=noun, dl=ph),
                  {"type": "amendment", "changes": {"deadline_phrase": ph}})
            check(L.CHANGE_TEMPLATES["deadline_for"].format(noun=noun, dl=ph),
                  {"type": "amendment", "changes": {"deadline_phrase": ph}})
    for stat in L.STATS:
        check(L.CHANGE_TEMPLATES["maintain_threshold"].format(stat=stat, n=5),
              {"type": "amendment", "changes": {"threshold": 5}})
    for ph in UNTIL_PHRASES:
        check(L.CHANGE_TEMPLATES["maintain_deadline"].format(u=ph), {"type": "amendment",
                                                                     "changes": {"deadline_phrase": ph}})
        check(L.CHANGE_TEMPLATES["avoid_deadline"].format(u=ph), {"type": "amendment", "changes": {"deadline_phrase": ph}})
    fb = [("Thank you, that's done.", "fulfilled", None, None), ("Great, that's sorted.", "fulfilled", None, None),
          ("Well done, that's what I wanted.", "fulfilled", None, None),
          ("You broke that one at step 187.", "violated", 187, None),
          ("Too late: that was due by step 412.", "late", 412, None), ("Correct.", "correct", None, None),
          ("Yes, that's right.", "correct", None, None), ("Right, thank you.", "correct", None, None),
          ("Not quite.", "wrong", None, None), ("Not quite. The answer was: o3 (wood).", "wrong", None, "o3 (wood)"),
          ("You did not answer in time.", "no_answer", None, None),
          ("You did not answer in time. The answer was: 4 west (west near).", "no_answer", None, "4 west (west near)")]
    for text, outcome, st, ans in fb:
        check(text, {"type": "feedback", "outcome": outcome, "step_mentioned": st, "revealed_answer": ans})
    from continuous_brain_world.layer import GREETING
    assert p.parse_raw(GREETING) == [{"type": "info"}]
    with pytest.raises(L.OrganError):
        p.parse_raw("[o5] Please juggle three cows by noon.")
    assert n_checked > 1500


@pytest.fixture(scope="module")
def live_corpus():
    # seeds 4 and 8 of the stress configuration together exercise every request family, every question
    # type (both "first request" windows), all eight change frames, cancellations and every feedback outcome
    return E.build_corpus([4, 8], episode_length=900, config=STRESS)


def test_development_parser_matches_the_layers_ground_truth(dev, live_corpus):
    items = live_corpus["items"]
    truths = [it["truth"] for it in items]
    assert {t["family"] for t in truths if t["type"] == "new_commitment"} == set(L.FAMILIES)
    assert {t["spec"]["question"] for t in truths if t["type"] == "question"} == set(L.QUESTIONS)
    assert {t["spec"].get("which") for t in truths if t["type"] == "question"} >= {"this_morning", "today"}
    assert {t["outcome"] for t in truths if t["type"] == "feedback"} == set(L.OUTCOMES)
    assert any(t["type"] == "cancellation" for t in truths) and any(t["type"] == "info" for t in truths)
    rx = {k: L._template_regex(t, L._GROUPS) for k, t in L.CHANGE_TEMPLATES.items()}
    kinds = {k for it in items if it["truth"]["type"] == "amendment" for k, r in rx.items()
             if r.fullmatch(it["message"].split("] ", 1)[1])}
    assert kinds == set(L.CHANGE_TEMPLATES)
    r = E.evaluate(dev, live_corpus)
    assert r["exact_match"]["correct"] == r["exact_match"]["total"] == len(items), r["failures_shown"][:3]
    assert r["deadline_step_accuracy"]["accuracy"] == 1.0
    a = r["answers"]
    assert a["grammar_valid"]["correct"] == a["grammar_valid"]["total"] > 10
    assert a["correct_with_oracle_memory"]["correct"] == a["correct_with_oracle_memory"]["total"]
    # the scorer is not vacuous: a wrong deadline, quantity or id is caught
    it = next(i for i in items if i["truth"]["type"] == "new_commitment" and i["truth"]["family"] == "collect")
    ev = dev.interpret(it["message"], dict(it["context"]))
    for mutate in (lambda e: e.update(deadline_step=e["deadline_step"] + 1), lambda e: e["spec"].update(quantity=9),
                   lambda e: e.update(id="o999")):
        e = copy.deepcopy(ev)
        mutate(e[0])
        assert not E.score_item(it, e)["exact"]


# --- deadlines -------------------------------------------------------------------------------------------
def test_deadline_phrases_resolve_exactly_as_the_caretaker_chose_them():
    rng = np.random.default_rng(0)
    seen = set()
    for step in range(0, 3300, 3):
        for until in (False, True):
            for dur in (40, 120, 260, 400):
                deadline, phrase = C.choose_deadline(rng, step, float(dur), until=until)
                assert L.resolve_deadline(phrase, step) == deadline, (step, phrase)
                assert L.resolve_deadline(phrase.upper() + ".", step) == deadline
                seen.add(phrase.split(" ")[0] + (" N" if any(c.isdigit() for c in phrase) else " " + phrase))
    assert len(seen) >= 12
    # brute force on the daylight cycle itself
    for step in range(0, 1300):
        if not dl.is_dark(step):
            s = step + 1
            while not dl.is_dark(s):
                s += 1
            assert L.resolve_deadline("before nightfall", step) == s - 1
        s = step + 1
        while not (not dl.is_dark(s) and dl.is_dark(s - 1)):
            s += 1
        assert L.resolve_deadline("by tomorrow morning", step) == s == L.resolve_deadline("until morning", step)
        s = step + 1
        while s % 300 != dl.NOON_T:
            s += 1
        assert L.resolve_deadline("by noon", step) == s
    assert L.resolve_deadline("within 25 steps", 70) == 95 and L.resolve_deadline("until step 9", 0) == 9
    assert L.resolve_deadline("before lunch", 5) is None
    # question windows follow W1's days
    raw = {"question": "first_request", "which": "this_morning"}
    assert L.question_spec(raw, 400) == {"question": "first_request", "window": [273, 360], "which": "this_morning"}
    assert L.question_spec({"question": "count_requests"}, 400) == {"question": "count_requests", "window": [273, 400]}
    assert L.question_spec({"question": "count_today", "event": "eat_cow"}, 272)["day"] == 0
    assert L.question_spec({"question": "count_today", "event": "eat_cow"}, 273)["day"] == 1


# --- answers -----------------------------------------------------------------------------------------------
def _question(dev, text, step=500):
    return dev.interpret(text + " (Answer within 120 steps.)", {"step": step})[0]


def test_answers_round_trip_through_w1s_parser_and_grader(dev):
    q = _question(dev, "[q7] Where did you last see a cow?")
    mem = {"position": [10, 10], "last_seen": {"cow": {"step": 400, "positions": [[15, 8], [30, 30]]}}}
    a = dev.answer(q, mem)
    assert a == "answer q7: 5 east 2 north"
    c = parse(a)[0]
    truth = {"kind": "location", "positions": [[15, 8], [30, 30]]}
    assert grade_report(truth, c["payload"], (10, 10), 2)
    b = dev.answer(q, {**mem, "location_format": "bucket"})
    assert b == "answer q7: east mid" and grade_report(truth, parse(b)[0]["payload"], (10, 10), 2)
    assert dev.answer(q, {"position": [15, 9], "last_seen": mem["last_seen"]}) == "answer q7: 1 north"
    assert dev.answer(q, {"position": [0, 0], "last_seen": {}}) == "answer q7: never"
    assert grade_report({"kind": "never"}, parse("answer q7: never")[0]["payload"], (0, 0), 2)
    qc = _question(dev, "[q8] How many cows have you eaten today?")
    assert dev.answer(qc, {"counts_today": {"eat_cow": 3}}) == "answer q8: 3"
    with pytest.raises(L.OrganError, match="no count"):
        dev.answer(qc, {"counts_today": {}})                        # never guesses
    qf = _question(dev, "[q9] What was the first thing I asked you for today?", step=500)
    reqs = [{"id": "o2", "issue_step": 250}, {"id": "o5", "issue_step": 300}, {"id": "o4", "issue_step": 300}]
    assert dev.answer(qf, {"requests": reqs}) == "answer q9: o4"     # day 1 starts at 273; ties by id number
    assert dev.answer(qf, {"requests": reqs[:1]}) == "answer q9: nothing"
    qn = _question(dev, "[q10] How many requests have I made today?", step=500)
    assert dev.answer(qn, {"requests": reqs}) == "answer q10: 2"
    ql = _question(dev, "[q11] What did I last call off?")
    assert dev.answer(ql, {"cancellations": [{"id": "o3", "step": 20}, {"id": "o6", "step": 90}]}) == "answer q11: o6"
    assert dev.answer(ql, {"cancellations": []}) == "answer q11: nothing"
    for text in ("answer q11: o6", "answer q10: 2", "answer q7: east mid"):
        assert parse(text)[0]["kind"] == "answer"
    with pytest.raises(L.OrganError):
        dev.answer({**q, "type": "new_commitment"}, mem)


def test_narrate_with_the_development_backend(dev):
    evs = dev.interpret("[o1] Please gather 3 wood before nightfall.", {"step": 20}) + \
        dev.interpret("[o1] Thank you, that's done.", {"step": 90}) + \
        [{"type": "utterance", "step": 95, "text": "answer q2: 3"}, {"type": "note", "step": 99, "text": "I slept"}]
    text = dev.narrate(evs)
    assert text.startswith("At step 20 (morning of day 0) I was asked to gather 3 wood before nightfall [o1].")
    assert "I was told that [o1] was done" in text and "I said: \"answer q2: 3\"" in text
    assert dev.narrate(evs) == text
    with pytest.raises(L.OrganError):
        dev.narrate([{"type": "note"}])
    assert dev.records[-1]["method"] == "narrate"


# --- backend selection and provenance ----------------------------------------------------------------------
def test_backend_selection_and_provenance(dev):
    prov = dev.provenance()
    assert prov["backend"] == "development_parser" and "not a language model" in prov["label"]
    assert prov["weights"] is None and len(prov["code_sha256"]) == 64
    ev = dev.interpret("[o9] Never mind the coal.", {"step": 3})[0]
    assert ev["provenance"]["backend"] == "development_parser" and ev["provenance"]["attempts"] == 1
    un = dev.interpret("[o9] Sing me a song.", {"step": 3})[0]
    assert un["type"] == "unparsed" and "no W1 caretaker template" in un["error"]
    with pytest.raises(ValueError):
        L.build_backend({"backend": "gpt"})
    with pytest.raises(ValueError):
        L.build_backend({"backend": "dev", "model_id": "x"})
    with pytest.raises(L.OrganError):
        dev.interpret("[o1] x", {"inventory": {}})


def test_extract_json_is_robust():
    ev = {"type": "info"}
    for text in ('[{"type": "info"}]', 'Sure!\n```json\n[{"type": "info"}]\n```\nDone.', '{"type": "info"}',
                 '{"events": [{"type": "info"}]}', 'The answer is [ {"type":"info"} ] .',
                 'noise { not json } then [{"type": "info"}]'):
        assert L.extract_json(text) == [ev], text
    assert L.extract_json('[{"type": "x", "a": "[not] {a} bracket"}]') == [{"type": "x", "a": "[not] {a} bracket"}]
    for text in ("", "no json here", "[1, 2]", '{"type": "info"'):
        with pytest.raises(L.OrganError):
            L.extract_json(text)


# --- HF backend: a tiny randomly initialised causal LM, built offline -----------------------------------------
def build_tiny_model(d: Path) -> Path:
    """A real (random-weight) Qwen2 causal LM + a byte-level BPE tokenizer trained offline on the organ's
    own prompt text, with a ChatML chat template.  No download anywhere."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GenerationConfig, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = [L.SYSTEM_PROMPT, L.RETRY_PROMPT, L.NARRATE_SYSTEM] + [m for m, _ in L.FEW_SHOT] + \
        [L._canon([r]) for _, r in L.FEW_SHOT]
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=1200, special_tokens=specials, show_progress=False,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    ft = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    ft.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                        "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    ft.save_pretrained(d)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(ft), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=True, eos_token_id=ft.eos_token_id, pad_token_id=ft.pad_token_id)
    model = Qwen2ForCausalLM(cfg)
    model.generation_config = GenerationConfig(eos_token_id=ft.eos_token_id, pad_token_id=ft.pad_token_id)
    model.save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    return build_tiny_model(root / "test--tiny-qwen2-random")


def test_hf_backend_end_to_end_on_a_real_tiny_model(tiny_dir):
    """No mock: prompt construction with the chat template, tokenisation, greedy generate, decoding, failed
    extraction, the bounded retry with the error fed back, audit records, provenance and unparsed output."""
    b = L.HFBackend("test/tiny-qwen2-random", None, tiny_dir, "cpu", "float32", max_new_tokens=16, max_attempts=2)
    organ = L.LanguageOrgan(b)
    msg = "[o1] Please gather 3 wood before nightfall."
    ev = organ.interpret(msg, {"step": 10})
    assert len(ev) == 1 and ev[0]["type"] == "unparsed" and "after 2 attempts" in ev[0]["error"]
    assert ev[0]["provenance"]["attempts"] == 2 and ev[0]["provenance"]["weights_digest"] == b.weights_digest
    assert [a["attempt"] for a in b.audit] == [1, 2] and not any(a["ok"] for a in b.audit)
    prefix = b.prompt_prefixes[b.audit[0]["prompt_prefix_sha256"]]
    assert L.SYSTEM_PROMPT in prefix and L._canon([L.FEW_SHOT[0][1]]) in prefix and prefix.count("<|im_start|>user") \
        == len(L.FEW_SHOT)
    assert b.audit[0]["prompt_suffix"].startswith("<|im_start|>user\n" + msg)
    assert "Your reply was not valid" in b.audit[1]["prompt_suffix"] and b.audit[1]["prompt_tokens"] > \
        b.audit[0]["prompt_tokens"] > 300
    assert all(0 < a["new_tokens"] <= 16 and a["latency_s"] > 0 for a in b.audit)
    first = [a["response"] for a in b.audit]
    organ.interpret(msg, {"step": 10})
    assert [a["response"] for a in b.audit[2:]] == first                       # greedy: deterministic
    prov = organ.provenance()
    assert prov["weights"] == ms.hash_tree(tiny_dir) and prov["weights_hash_source"] == "computed(no lock)"
    assert prov["generation"]["do_sample"] is False and prov["generation"]["num_beams"] == 1
    assert organ.records[0]["model_id"] == "test/tiny-qwen2-random"


def _scripted_generate(backend, replies, seen):
    """Replace model.generate by one that returns scripted replies (encoded with the real tokenizer)."""
    import torch
    it = iter(replies)

    def generate(input_ids=None, attention_mask=None, generation_config=None, **kw):
        seen.append({"do_sample": generation_config.do_sample, "prompt_len": int(input_ids.shape[1]),
                     "max_new_tokens": generation_config.max_new_tokens})
        ids = backend.tokenizer(next(it), add_special_tokens=False)["input_ids"]
        return torch.cat([input_ids, torch.tensor([ids], dtype=input_ids.dtype)], dim=1)
    backend.model.generate = generate


def test_hf_backend_extraction_validation_and_retry_with_scripted_replies(tiny_dir, dev):
    """model.generate is MOCKED here (a random model cannot be made to emit valid JSON); everything around it
    is real: chat template, tokenizer encode/decode, extraction, completion, validation, retry and audit."""
    b = L.HFBackend("test/tiny-qwen2-random", None, tiny_dir, "cpu", "float32", max_new_tokens=64, max_attempts=3,
                    audit_path=tiny_dir.parent / "audit.jsonl")
    organ = L.LanguageOrgan(b)
    msg = "[o12] Top priority: please gather 4 more pieces of stone before it gets dark."
    good = {"type": "new_commitment", "id": "o12", "family": "collect", "value": 3,
            "deadline_phrase": "before it gets dark", "spec": {"item": "stone", "quantity": 4}}
    seen = []
    _scripted_generate(b, ['Sure: {"type": "new_commitment", "id": "o12"',
                           "```json\n" + json.dumps([{**good, "deadline_phrase": "before dark"}]) + "\n```",
                           "Here it is: " + json.dumps([good]) + " Hope that helps."], seen)
    ctx = {"step": 20, "inventory": {"stone": 1}}
    ev = organ.interpret(msg, ctx)
    assert ev[0]["type"] == "new_commitment" and ev[0]["provenance"]["attempts"] == 3
    ref = dev.interpret(msg, ctx)[0]
    strip = lambda e: {k: v for k, v in e.items() if k != "provenance"}  # noqa: E731
    assert strip(ev[0]) == strip(ref) and ev[0]["spec"]["target"] == 5 and ev[0]["deadline_step"] == 147
    assert [a["ok"] for a in b.audit] == [False, False, True]
    assert "no JSON array" in b.audit[0]["error"] and "verbatim" in b.audit[1]["error"]
    assert "verbatim" in b.audit[2]["prompt_suffix"]                          # the error was fed back
    assert all(s["do_sample"] is False and s["max_new_tokens"] == 64 for s in seen)
    lines = [json.loads(x) for x in (tiny_dir.parent / "audit.jsonl").read_text().splitlines()]
    assert [x["kind"] for x in lines] == ["prompt_prefix", "attempt", "attempt", "attempt"]
    # narrate through the model (scripted reply), and a reply that is not prose is retried
    _scripted_generate(b, ["{}", "I gathered stone before dark, as I had been asked."], seen)
    text = organ.narrate(ev)
    assert text == "I gathered stone before dark, as I had been asked." and organ.records[-1]["attempts"] == 2
    assert "gather 4 stone before it gets dark" in b.audit[-1]["prompt_suffix"]


def test_hf_backend_refuses_weights_that_disagree_with_the_lock(tiny_dir, tmp_path):
    import shutil
    models = tmp_path / "models"
    d = models / "test--tiny-qwen2-random"
    shutil.copytree(tiny_dir, d)
    lock = {"format": ms.LOCK_FORMAT, "models": {"test/tiny-qwen2-random": {
        "dir": d.name, "revision": "f" * 40, "files": ms.hash_tree(d)}}}
    ms.save_lock(models / ms.LOCK_NAME, lock)
    b = L.HFBackend("test/tiny-qwen2-random", None, d, "cpu", "float32")
    assert b.revision == "f" * 40 and b.hash_source == "computed+lock"
    assert L.HFBackend("test/tiny-qwen2-random", "f" * 40, d, hash_weights="lock").hash_source == "lock(size-checked)"
    with pytest.raises(L.OrganError, match="revision"):
        L.HFBackend("test/tiny-qwen2-random", "e" * 40, d)
    cfg = d / "config.json"
    cfg.write_text(cfg.read_text().replace('"hidden_size": 32', '"hidden_size": 32 '))
    with pytest.raises(L.OrganError, match="disagree"):
        L.HFBackend("test/tiny-qwen2-random", None, d)
    with pytest.raises(L.OrganError, match="no lock entry"):
        L.HFBackend("test/tiny-qwen2-random", None, tiny_dir, hash_weights="lock")
    with pytest.raises(ValueError, match="local_dir"):
        L.HFBackend("test/tiny-qwen2-random", None, None)


# --- fetch_models / reassemble_model: lock logic on fake directories (no network) -------------------------
@pytest.fixture
def fetch(monkeypatch):
    fm = load("scripts/explore/fetch_models.py")
    state = {"content": {"config.json": b'{"a": 1}', "model.safetensors": bytes(range(256)) * 10000,
                         "onnx/model.onnx": b"skipped"}, "sha": "a" * 40, "lfs": None, "calls": []}

    def resolve_revision(model_id, revision, token):
        state["calls"].append(("resolve", model_id, revision))
        lfs = state["lfs"] if state["lfs"] is not None else \
            {"model.safetensors": hashlib.sha256(state["content"]["model.safetensors"]).hexdigest()}
        return state["sha"], lfs

    def download(model_id, revision, dest, ignore, token):
        state["calls"].append(("download", model_id, revision))
        for rel, data in state["content"].items():
            p = Path(dest) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        (Path(dest) / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        (Path(dest) / ".cache" / "huggingface" / "x.metadata").write_text("bookkeeping")

    monkeypatch.setattr(fm, "resolve_revision", resolve_revision)
    monkeypatch.setattr(fm, "download", download)
    return fm, state


def test_fetch_models_pins_on_first_use_and_refuses_mismatches(fetch, tmp_path, capsys):
    fm, state = fetch
    md = tmp_path / "cb_models"
    args = ["--model", "Qwen/Qwen2.5-0.5B-Instruct", "--models-dir", str(md)]
    assert fm.main(args) == 0
    lock = json.loads((md / "models.lock.json").read_text())
    e = lock["models"]["Qwen/Qwen2.5-0.5B-Instruct"]
    assert e["revision"] == "a" * 40 and e["dir"] == "Qwen--Qwen2.5-0.5B-Instruct" and e["hub_lfs_sha256_checked"] == 1
    assert set(e["files"]) == {"config.json", "model.safetensors"}            # onnx/ ignored, .cache skipped
    assert e["files"]["model.safetensors"]["sha256"] == hashlib.sha256(state["content"]["model.safetensors"]).hexdigest()
    # second run: the lock's revision is used (no resolve), files verified
    state["calls"].clear()
    assert fm.main(args) == 0 and state["calls"] == [("download", "Qwen/Qwen2.5-0.5B-Instruct", "a" * 40)]
    assert fm.main(args + ["--verify-only"]) == 0
    # a changed file is refused and the lock stays as it was
    before = (md / "models.lock.json").read_text()
    state["content"]["config.json"] = b'{"a": 2}'
    assert fm.main(args) == 2
    assert "REFUSED" in capsys.readouterr().err and (md / "models.lock.json").read_text() == before
    assert fm.main(args + ["--verify-only"]) == 2
    state["content"]["config.json"] = b'{"a": 1}'
    assert fm.main(args) == 0
    (md / "Qwen--Qwen2.5-0.5B-Instruct" / "extra.bin").write_bytes(b"x")
    assert fm.main(args + ["--verify-only"]) == 2
    (md / "Qwen--Qwen2.5-0.5B-Instruct" / "extra.bin").unlink()
    # a conflicting revision is refused unless --repin
    assert fm.main(args + ["--revision", "Qwen/Qwen2.5-0.5B-Instruct=" + "b" * 40]) == 3
    state["sha"] = "b" * 40
    assert fm.main(args + ["--revision", "Qwen/Qwen2.5-0.5B-Instruct=" + "b" * 40, "--repin"]) == 0
    e2 = json.loads((md / "models.lock.json").read_text())["models"]["Qwen/Qwen2.5-0.5B-Instruct"]
    assert e2["revision"] == "b" * 40 and e2["repinned_from"] == "a" * 40
    # first download whose bytes disagree with the Hub's LFS sha256: refused, nothing pinned
    state["lfs"] = {"model.safetensors": "0" * 64}
    assert fm.main(["--model", "Qwen/Qwen2.5-1.5B-Instruct", "--models-dir", str(md)]) == 2
    assert "Qwen/Qwen2.5-1.5B-Instruct" not in json.loads((md / "models.lock.json").read_text())["models"]
    with pytest.raises(SystemExit):
        fm.main(["--model", "someone/unknown-model", "--models-dir", str(md)])


def test_split_and_reassemble_round_trip(fetch, tmp_path):
    fm, state = fetch
    state["content"]["model.safetensors"] = bytes(range(256)) * (5 * 4096 + 7)   # 5 MiB + 1792 bytes
    md = tmp_path / "cb_models"
    assert fm.main(["--model", "Qwen/Qwen2.5-0.5B-Instruct", "--models-dir", str(md), "--split-mb", "2"]) == 0
    split = md / "_split" / "Qwen--Qwen2.5-0.5B-Instruct"
    man = json.loads((split / "split_manifest.json").read_text())
    parts = man["files"]["model.safetensors"]["parts"]
    assert [p["size"] for p in parts] == [2 << 20, 2 << 20, (5 * 4096 + 7) * 256 - (4 << 20)]
    assert all(p.stat().st_size <= 2 << 20 for p in split.rglob("*.part*"))
    ra = load("scripts/explore/reassemble_model.py")
    other = tmp_path / "elsewhere"
    assert ra.main([str(split), "--models-dir", str(other)]) == 0
    src_lock = json.loads((md / "models.lock.json").read_text())["models"]["Qwen/Qwen2.5-0.5B-Instruct"]
    dst_lock = json.loads((other / "models.lock.json").read_text())["models"]["Qwen/Qwen2.5-0.5B-Instruct"]
    assert dst_lock["files"] == src_lock["files"] and dst_lock["revision"] == src_lock["revision"]
    assert ms.hash_tree(other / "Qwen--Qwen2.5-0.5B-Instruct") == src_lock["files"]
    assert ra.main([str(split), "--models-dir", str(other)]) == 0             # idempotent against its own lock
    # a corrupted chunk is detected and leaves nothing half-written
    third = tmp_path / "third"
    chunk = split / parts[1]["name"]
    data = bytearray(chunk.read_bytes())
    data[100] ^= 1
    chunk.write_bytes(bytes(data))
    assert ra.main([str(split), "--models-dir", str(third)]) == 2
    assert not (third / "Qwen--Qwen2.5-0.5B-Instruct" / "model.safetensors").exists()
    assert not list(third.rglob("*.reassembling")) and not (third / "models.lock.json").exists()


def test_scripts_have_help_and_no_network_at_import():
    code = ("import sys, importlib.util; sys.path.insert(0, %r)\n"
            "for rel in ('scripts/explore/fetch_models.py', 'scripts/explore/reassemble_model.py',"
            " 'scripts/explore/eval_language_organ.py'):\n"
            "    s = importlib.util.spec_from_file_location('m', %r + '/' + rel); m = importlib.util.module_from_spec(s);"
            " s.loader.exec_module(m)\n"
            "print('huggingface_hub' in sys.modules, 'transformers' in sys.modules)") % (str(REPO), str(REPO))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False"]
    for rel in ("scripts/explore/fetch_models.py", "scripts/explore/reassemble_model.py",
                "scripts/explore/eval_language_organ.py"):
        h = subprocess.run([sys.executable, str(REPO / rel), "--help"], capture_output=True, text=True, timeout=60)
        assert h.returncode == 0 and "usage:" in h.stdout, h.stderr


def test_model_cards_are_complete_and_unpinned():
    cards = json.loads((REPO / "continuous_brain_organs" / "model_cards.json").read_text())
    ids = {m["id"] for m in cards["models"]}
    assert ids == {"Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
                   "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
                   "HuggingFaceTB/SmolVLM-256M-Instruct", "Qwen/Qwen2-VL-2B-Instruct"}
    for m in cards["models"]:
        assert m["revision"] is None and m["files_sha256"] is None, m["id"]
        assert m["role"] in ("language", "vision-language") and m["parameters"] > 1e8 and m["expected_disk_bytes"] > 1e8
        assert set(m["licence"]) >= {"spdx_or_name", "gated", "non_apache", "commercial_use"}
    by = {m["id"]: m for m in cards["models"]}
    assert by["meta-llama/Llama-3.1-8B-Instruct"]["licence"]["gated"] is True
    assert by["meta-llama/Llama-3.1-8B-Instruct"]["licence"]["non_apache"] is True
    assert by["Qwen/Qwen2.5-3B-Instruct"]["licence"]["non_apache"] is True
    assert all(set(v) <= ids for v in cards["sets"].values())
