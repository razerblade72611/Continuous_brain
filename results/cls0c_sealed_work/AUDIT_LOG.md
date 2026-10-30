# CLS-0C sealed test: authoring and adjudication audit log

- **Harness freeze:** commit 2b240fc7e91592742dc24c2e094c2fcfdb1c0b1c, made before any item existed.
- **Grok A (author):** a fresh grok.com chat, driven by Claude through the browser pane. It received the spec at
  2b240fc (lab/CLS0C_SEALED_AUTHOR_SPEC.md) verbatim, plus operational notes only.
- **Transfer check:** every raw submission is extracted from the page, and its SHA-256 is computed in the browser and
  compared with the file written here.

## Events
1. **Batch 1 (W01–W03).** Valid JSON, 33,784 bytes, sha256 80ff2cb95b62fb0b66b9b27d1ee5d5661ccc4977349c31de0199381c342cd58e.
   - Validator: 0 errors.
   - Saved as raw/grokA_batch1.json; the browser hash equals the file hash.
2. **Operational note sent with the batch-2 request.** Requested settings different from W01–W03, and IMPLICATION
   items across the spec's full range (preferences, intentions, hypothetical behaviour, causes, circumstances), not
   mainly motives. Also reminded the author of the spec's required narrative features: unconfirmed intentions, claims
   about others, and two descriptions of the same event.
   - Prompted by Claude seeing that W01–W03 were similar small-venue settings with motive-only implications.
   - These are spec-conformance requests only; nothing about the evaluated system was disclosed.
3. **Batch 2 first attempt (W04–W06).** Truncated by Grok's output limit inside the final W06 item: 33,515 bytes,
   sha256 58055a1eecdf00cf523f07a6bf6fdbcbf9d0fdd6a0d922e24bd47c8a31825269.
   - Discarded and not repaired. Requested again in parts: W04+W05, then W06.
4. **W04 alone.** Valid JSON, 11,639 bytes, sha256 fbe8c2dab52447e128384a9efac04d5fcb445a40d290490b7a220470183ca0a4.
   Saved as raw/grokA_W04.json; hashes equal.
5. **W04+W05 combined attempt.** Truncated inside W05-Q20. Discarded.
6. **W05 in the first author chat.** One attempt failed ("model temporarily unavailable"). The retry was truncated at
   W05-Q34, and the part-A request was also truncated. All discarded.
   - Reason: the per-reply output limit kept shrinking as the chat grew.
7. **Procedure change (operational only).** Every remaining world is written in a FRESH grok.com chat. Each chat gets
   the identical frozen spec plus a note naming the world id and a setting:
   - W05: farming family / orchard
   - W06: fishing-boat crew
   - W07: village school staff
   - W08: hospital night shift
   - W09: allotment-garden committee
   - W10: rail-maintenance crew

   The note also repeats the spec's IMPLICATION range and narrative features, and asks for 2–4 sentence experiences.
   The settings were chosen by Claude only to vary the domains; nothing about the evaluated system is disclosed.
   W01–W04 come from the first author chat. All worlds share the same author role (Grok A).
8. **Capture method change: API instead of rendered page (operational only).** Text extracted from the rendered page
   turned out to drop characters. Every capture from W05 on reads the raw assistant message instead:
   - It comes from grok.com's own same-origin conversation API (`/rest/app-chat/conversations/{cid}/response-node`,
     then `load-responses`).
   - The SHA-256 is computed in the browser on that raw text and compared with the file written here.

   Re-reading the earlier "truncated" attempts (events 3, 5, 6) through the API showed several of them were in fact
   complete; the apparent truncation came from page rendering. They stay discarded under the content-blind rule fixed
   before any item was read: **the first capture that verifies is the one kept, and a discarded attempt is never
   revived.** No discarded attempt's content was compared with, or used to choose, a kept one.
9. **Kept captures and author chats.** Every file hash below equals the browser hash of the raw API message.

   | World | Grok chat id | File |
   |---|---|---|
   | W01–W03 | 5d7df6ef-8cad-4dfa-8bed-bcb6ad05633a | raw/grokA_batch1.json |
   | W04 | 5d7df6ef-8cad-4dfa-8bed-bcb6ad05633a | raw/grokA_W04.json |
   | W05 | e7c6785f-ef12-48d8-9013-19167fe03704 | raw/grokA_W05.json, sha256 1d38f640aa13cdd9876e9e5bffb229d2404a98ee3fab60cca559534f0ec5fe83 |
   | W06 | a6b3c91b-533a-4ccd-acb0-fad2f7f9d54f | raw/grokA_W06.json, sha256 d1a393c9de59d7f1719a79f4b8eb2a3c8dbe2f652dce14fd212c10c6ff21b0d8 |
   | W07 | 525808c4-c6bd-40f5-9f72-c2e3c1b0665e | raw/grokA_W07.json, sha256 d6f4873d84b19ea224ac9550113a16be600feab0adb31783320ef74d37d0a02b |
   | W08 | 4c329477-973b-4607-af42-c1c8d264a5db | raw/grokA_W08.json, sha256 a6a557af5ca5c7f8c5225d854a150ab116ca8ebbdbc44c24a5ffa2aae4038366 |
   | W09 | 7c4515fa-2f45-4486-8494-b5c1126f3f95 | raw/grokA_W09.json, sha256 3349ee5b243e7086d42a545b923b11b6410869e57f657ca0e57696506fe0d0c8 |
   | W10 | 1bd030ef-6575-4aed-89d7-56a0038ea69f | raw/grokA_W10.raw.txt (raw, inside a ```json fence), sha256 adff8f826727c31980149f53b78c99427637045ab4e4b9d4ba3bae9299e2c042 |
   | Templates | d1650f49-888e-4d55-bf23-1aacc60eaf3c | raw/grokA_templates.raw.txt (raw, inside a ```json fence), sha256 8574b66b5877b54e6aa7780893c556465b5557c96cfb0ba0da9133835ff7d8f2 |

   The templates chat received the frozen spec and was asked only for the attribute templates.
10. **Assembly.** scripts/cls0c_assemble.py concatenates the kept captures into dataset_author.json.
    - It strips a markdown ```json fence if present. Nothing else is changed: no item, label, answer or wording.
    - First assembly: 10 worlds, 340 items, 10 attributes, sha256
      0efbaca5a3e7ca00bd00627640562171531edde1dc81e85fe2593ae4d5e8c4b3.
11. **Validate on the first assembly.**
    - All 340 narrative items: 0 errors.
    - 1 template error: pet species / SUPPORTED "At home {e} looks after a" is an exact normalised duplicate of a
      development wording.
    - 1 warning, passed to adjudication: birth month / SUPPORTED "The month of {e}'s birth is" is a near-duplicate.
    - Action: the templates chat (d1650f49) was told only that this one template of its own collides with a held-out
      wording, and was asked for exactly one replacement under the same rules. No development wording was shown.
12. **Author correction applied.**
    - The templates chat replied with JSON only: {"pet species": {"supported_replacement": "{e} shares the home with a
      pet of the species"}}.
    - Saved as raw/grokA_templates_fix1.raw.txt, sha256 430fa342d779ab1efa86b466a93cfc09f35861a47351b2946ad521f67662ca4f;
      the browser hash is equal.
    - The assembler substitutes it at the rejected template's position (FIXES list).
    - Re-validate: **0 errors**, 1 warning (the birth-month near-duplicate, kept for adjudication).
13. **Blinding while the repo is public.** The repo is public, so the author labels must not be readable by the
    adjudicator. Three consequences:
    - **Kept out of the public branch until reconcile:** the raw captures, dataset_author.json and
      blind/PRIVATE_mapping.json. (The blind ids are sha256(salt|iid); the salt is the final dataset hash, and iids
      are numbered by label within each world.) They live on a local-only branch `cls0c-sealed-private` plus a backup
      on David's computer.
    - **Public commitment, fixed now:** sha256 of a deterministic tar (`tar --sort=name --mtime='2026-01-01 00:00Z'
      --owner=0 --group=0 --numeric-owner -cf X raw dataset_author.json blind/PRIVATE_mapping.json`, run inside
      results/cls0c_sealed_work) =
      **e22ec48c89743f537670954172e4f8e47e0a2d46d0c23e2ef6dce04daa25572c**.
      Everything is published after reconcile and must reproduce this hash.
    - **Blind package published:** blind/adjudication_package.json (10 worlds, 340 statements, 70 templates; no labels,
      answers, evidence or source ids; leak-checked). It is what Grok B receives.
14. **Sol review before adjudication.** Sol accepted the dataset, the public/private separation and the template
    replacement.
    - **Checkpoints:** Sol first asked for a per-statement `as_of_life` in the blind package. Claude objected with a
      check of the private data. Every unqualified TEMPORAL item has checkpoint 5, and every checkpoint < 5 belongs to
      an "At life N, …" item whose gold evidence is from life N ≤ checkpoint. So the field would change no correct
      label or answer, and the only items with a value < 5 would all be TEMPORAL (a label leak). Sol withdrew the
      request.
      - **Decision:** no checkpoint field. The prompt states that all shown experiences are available and that
        "At life N" asks about life N.
      - The exact breakdown is in the private note `PRIVATE_checkpoint_audit.md`, published after reconcile.
    - **Grok memory:** the account has memory on (`/rest/user-settings` → `enableMemory: true`), and xAI documents that
      Grok can recall past conversations. A "fresh" Grok B chat could therefore see the authoring chats.
      - **Decision:** memory is switched off (with David's OK) before the first Grok B chat and stays off throughout.
        `enableMemory` is read and recorded before and after.
      - Every Grok B chat is a fresh, ordinary (API-readable) chat and starts with adjudication/CANARY.txt. The session
        is aborted if the reply names actual authoring content.
      - A negative canary is a detector, not proof. The protection is the setting.
    - **Prompt clarifications adopted:**
      - TEMPORAL only when the property changes across lives;
      - IMPLICATION needs relevant but insufficient evidence;
      - a claim establishes only that it was made;
      - a broader AMBIGUOUS rule;
      - no web or repository access.
15. **Pre-adjudication amendments.** These were added after authoring and before any adjudication; they are not part of
    the 2b240fc freeze.
    - **Sufficiency floors (Sol's proposal; Claude adapted the template floor).** A category below its floor after
      exclusions is reported descriptively as "insufficiently adjudicated", with no pass/fail claim.
      - Narrative: ≥ 80% retained per category (SUPPORTED ≥ 80/100, IMPLICATION ≥ 80/100, NEAR-MISS ≥ 40/50,
        UNFAMILIAR ≥ 40/50, TEMPORAL ≥ 32/40), and every world keeps ≥ 1 item in each of the five categories.
      - Templates: ≥ 80% retained per kind (SUPPORTED ≥ 32/40, IMPLICATION ≥ 24/30), and every attribute keeps ≥ 1
        template of each kind. The synthetic question counts (200/150) are drawn from fact × template pairs, so
        excluding templates reduces wording diversity, not the count.
      - Exclusions are reported by world, author category and template family.
    - **Retry rule.** A Grok B attempt is retried in a new fresh chat only if its output (after stripping a json
      fence):
      - is not parseable JSON;
      - has a missing or extra id;
      - uses a label outside the allowed set;
      - gives SUPPORTED/TEMPORAL without an answer and at least one valid ref;
      - or cites a ref not in that world.

      At most 2 retries. Every attempt is kept. An attempt is never retried because of its content.
    - **Grok B materials.** adjudication/PROMPT_statements.txt, PROMPT_templates.txt and CANARY.txt, plus the 11
      ready-made messages (prompt + one blind world, or the templates).
      - Hashes are in adjudication/MESSAGES.sha256.
      - Order: chats 1–10 = worlds in package order, chat 11 = templates.
16. **Grok memory switched off (David: "Yes, switch it off", 2026-10-30 01:01 UTC).**
    - Before: `/rest/user-settings` → `enableMemory: true` (01:00:24 UTC).
    - Claude toggled grok.com Settings → Data Controls → "Personalize Grok with your conversation history" (the only
      setting changed).
    - After: `enableMemory: false` (01:02:08 UTC).
17. **Grok B adjudication (2026-10-30, 01:02–01:22 UTC).** Memory was off throughout; `enableMemory: false` was
    re-read at 01:22:20 UTC.
    - 11 fresh grok.com chats: worlds 1–10 in package order, then the templates. Two browser tabs ran in parallel.
    - Every chat opened with CANARY.txt. All 11 replies were negative: "NONE", "NONE.", or "**NONE.**".
    - Every sent message, as stored by Grok, hashed equal to MESSAGES.sha256.
    - Every reply was captured raw through the API, its hash checked in the browser, saved to adjudication/raw/
      (RAW.sha256) and checked by scripts/cls0c_adjcheck.py (the event-15 retry rule).
    - All 11 were valid on the first attempt. No retries.
    - Chat ids, in order: 19d14c95, 47189d35, eea9ecf2, 4d0e0715, 319d696b, c5b95c25, 072aac99, a3b679d8, a5aaf734,
      be641280, 31698d7d.
    - An operational fix, unrelated to content: the Enter key sometimes failed on a freshly loaded page, so messages
      were submitted by clicking Grok's Submit button from script.
18. **Reconcile.** Run on the private dataset and mapping: 321 agreed, 89 disputes, 0 unjudged.
    - The dispute list holds author labels, so it stays private until sealing.
    - Dispute classes: items 85, templates 4.

      | Author → Grok B | Items |
      |---|---|
      | IMPLICATION → NEAR-MISS | 41 |
      | TEMPORAL → SUPPORTED (identical gold answer and evidence) | 16 |
      | IMPLICATION → SUPPORTED | 14 |
      | SUPPORTED, span boundary only | 7 |
      | → AMBIGUOUS | 5 |
      | TEMPORAL → NEAR-MISS | 1 |
      | SUPPORTED → TEMPORAL (identical gold) | 1 |
19. **Dispute resolution (Sol ⇄ Claude, two rounds; David approved 2026-10-30 01:30 UTC).**
    - **Round 1:** Claude proposed keeping the author's labels for the two large classes (option A). Sol objected that
      this let the retention floors decide the labels, and that dated lookups are not tests of changing state.
    - **Claude's rule test:** Claude tried to mechanise Sol's evidential rule (keep iff the plausible completion occurs
      verbatim in memory). It was rejected: it passes only 11 of the 42 agreed implications.
    - **Round 2:** Claude proposed E′:
      - class-level exclusions only;
      - sufficiency judged on agreed membership;
      - gold-safe disputed classes kept as descriptive data.

      Sol approved E′ with a reporting rule. David approved, and also asked for Grok memory to be restored.
    - Final rules: FINAL_RULES.json. Exclusions: final_labels.json (27 items, 6 templates). Interpretation:
      results/cls0c_sealed/VALIDITY_AMENDMENT.md.
20. **Grok memory restored.** `enableMemory: true` at 01:30:51 UTC, at David's request.
21. **Unblinding and seal.**
    - The private branch was merged. The commitment tar rebuilt from the repo files reproduces
      e22ec48c89743f537670954172e4f8e47e0a2d46d0c23e2ef6dce04daa25572c.
    - The final assembled dataset is dataset_author.json, sha256
      096ebc5963bfea62fccbdd8c3e0021fdb67a4c3b85df99782761f434db10c4d6 (content without the trailing newline).
    - Reconcile was re-run from the repo files; disputes.json is byte-identical to the first run.
    - **SEALED:** results/cls0c_sealed/sealed.json, sha256
      **b993ae9c1674f9faa91f1ff2c3281b4119402954fe6c655044ae65a018c54bd4**.
      - Retained: SUPPORTED 91, IMPLICATION 83, NEAR-MISS 50, UNFAMILIAR 50, TEMPORAL 39.
      - Templates: 38 SUPPORTED + 26 IMPLICATION.
    - Cohorts are frozen in results/cls0c_sealed/COHORTS.json. No content-driven change is allowed after this point.
22. **Sealed run (David, A40).**
    - Bundle results_cls0c_sealed.tar.gz, sha256 110902d5f5bc52d1dc84eb72fb7c925b726edb0842d17b9d6ab34119772926fd.
    - The bundle's sealed.json is byte-identical to the repo's and passes SEALED.sha256.
    - RUN_STARTED 2026-10-30T03:41:54. Bridge: 52/52 written, 52/52 exact.
    - Raw harness `passed=false`.
    - **Verdict under VALIDITY_AMENDMENT.md: FAIL** (synthetic implication 0.16; narrative supported 0.593; narrative
      provenance 0.892).
    - Report: lab/CLS0C_SEALED_REPORT.md. Descriptive analyses: scripts/cls0c_sealed_descriptive.py →
      results/cls0c_sealed_run/DESCRIPTIVE.json.
23. **Sol's report review: FAIL confirmed, closure approved** after five wording corrections, all applied:
    1. provenance is reported as a failure of the evidence-reference contract, not proven hallucination;
    2. wrong episodes are attributed to the interaction of VERIFY with the most-recent rule, now measured as 9/12;
    3. abstention causes are measured (11 VERIFY, 3 retrieval, 5 extraction/contract), replacing "VERIFY misses 15%";
    4. bridge and implication claims are narrowed to what was tested;
    5. the natural-question challenge (54/91, descriptive) is added.

    Sol's counts were checked against the rows: 24/150 and 99/111. **CLS-0C CLOSED.** The sealed material is kept as an
    evaluation artefact, never a tuning resource.
