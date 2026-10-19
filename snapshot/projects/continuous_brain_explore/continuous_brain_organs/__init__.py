"""Organs of the persistent individual (exploration lane, ENGINEERING_ONLY).

vision    -- the vision organ: a learned tile decoder from W1's pixels to a perceived local tile map
language  -- the language organ: open-weight language models (or the labelled development stand-in) that turn
             caretaker messages into structured events, compose answers and narrate the individual's life

Organs are used by the individual as hosts over the canonical store; their weights are pinned by model cards
(vision) and a trust-on-first-use lock file (language).
"""
