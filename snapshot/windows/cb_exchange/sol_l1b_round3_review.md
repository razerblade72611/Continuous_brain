# L1b independent handoff review — 11 October 2026

Decision: **changes requested before the benchmark launch**. The repaired experiment and its scientific decision rules are retained. No paid resource was created or altered by this review; canonical S2b and the retained L1 raw results were untouched.

Revision dfcaad74e30f58e937ec14b082927339bd92b282 was verified from the versioned bundle, SHA-256 b6e3b9d9ce530df5ed1a3c0484c66750b3f5fca3cffbb9a1b9875b053662c4bc. Revision 5c630fec2a2fd08f034e58422ec0d6594278aae4 adds only the WSL launcher and watcher; its bundle matches e996749723e9d3665fe8555714540addf4ea5602bbb88ee88806b65373988bff. The exchanged pod script matches d39bec13101c556ddd4852f6dd42f31f8ec9b12405c1423a7954e3c81a3edfd7.

Independent focused suites passed **39 checks, one diagnostic warning, in 42.75 seconds**. The warning concerns conversion of a gradient-bearing diagnostic tensor to a scalar. The reviewed source and exchanged LF shell scripts pass shell syntax checks. Windows Git converted the local checkout to CRLF; that local conversion is not a source defect.

The previous changed-revision/argument resume case is now refused before lock rewriting. Strict pairing refuses missing, duplicate, unexpected and non-finite rows. The cycle-0 instrument compares action/command traces and declared rows. Final partials carry run, revision, condition and content binding, and an interrupted unrecorded final batch explicitly refuses repetition. Origin evidence is portable. Rehearsals use separate smoke blocks and cannot make a capability claim; production practice RUN11 excludes the disclosed rehearsal exposure. Concurrent two-sleeper timing and worker scaling are now implemented. Actual GPU performance and a full-run cost remain unmeasured.

## Remaining result and transfer defects

| Reproduced case | Consequence | Required closure |
| --- | --- | --- |
| report() accepts an aggregate final.json from another run/revision when its worlds match the current plan, without loading verified partials | It can score foreign or corrupted results under the current run's lock | Share validated result loading between final() and report(); validate metadata and aggregate content against every bound, checksummed set |
| Fresh CPU restore extracts runs only, leaving the loop log at an unextracted workspace/l1b-run.log archive member; the final tar requires /workspace/l1b-run.log | Final packaging exits2 after restoring the run directory | Carry required logs inside the portable run output or explicitly restore them; verify the fresh-machine packaging path |

These were reproduced with synthetic records and tiny temporary archives: zero simulated lives, zero training updates and zero cloud use. The report-binding counterexample mocks identity_check only to isolate the report's missing result validation. It does not claim that actual individual identity checks failed. [Counterexample evidence](C:/Users/davei/Documents/Codex/2026-10-09/referenced-chatgpt-conversation-this-is-an/outputs/L1b_Round3_Review_Counterexamples.json).

## Launcher and billing protection

Fully mocked shell executions reproduced two additional failure paths. An upload error after successful creation exits before the watcher is started, with no cleanup attempt for the created pod. A deletion API error causes the watcher to exit0 without a live termination verification or retry. No real API or SSH calls were made. [Mock execution evidence](C:/Users/davei/Documents/Codex/2026-10-09/referenced-chatgpt-conversation-this-is-an/outputs/L1b_Launcher_Failure_Review.json).

The requested closure is to arm cleanup immediately after creation, parse and preserve the exact resource identity, bound SSH and transfer operations, verify termination with bounded retries and explicit failure reporting, and verify live ownership before reusing or deleting a pod. Slow transfers must not defeat the elapsed-time guard. Failed transfer/checksum verification should leave evidence recoverable while ending compute billing. A phase's STOP message ends its script; billing only ends when the resource lifecycle operation succeeds.

Installed CLI runpodctl2.14.0-dd55bcf was inspected read-only. The proposed create flags match its live help. Its create help does not expose the stop-after/terminate-after flags mentioned by the skill, so those flags must not be assumed available. The shared CLI was not updated.

The exact findings and evidence were sent directly to Claude as S-013/S-014 and recorded on [DAV-99](https://linear.app/david1/issue/DAV-99/cb2-l1b-repaired-powered-test-of-learning-from-its-own-life-draft-not), comment 3f09cf82-006e-49f5-8f2a-12817eaf91ff. This review completes the requested handoff evaluation with changes requested. The scheduled follow-up pauses at its defined review stop point. A new corrected handoff needs a further review; nothing is ready for an automatic paid launch.

Claude proposed a small, separate hardware benchmark before pricing the full loop. His message relays David's approval, but this review's current heartbeat explicitly forbids paid operations. That scope must be changed directly by David, or supported by trusted direct-user evidence, after the package passes review. The full loop remains a separate measured-cost decision. The limited positive CPU memory result remains valid; behavioral growth and learned primitive hands have not been demonstrated by this review.
