#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import torch

from continuous_brain.additive_goal_options_v1 import (
    load_goal_options,
    sha256,
)
from continuous_brain.recurrent_goal_option_state_v1 import (
    load_controller,
)
from continuous_brain.multi_item_episodic_memory_v1 import (
    QUERY_FIRST,
    QUERY_SECOND,
    load_reader,
)

GOAL_DECODER_PATH = Path(
    "state/memory_guided_goal_option_v1/goal_decoder.pt"
)
GOAL_OPTIONS_PATH = Path(
    "state/additive_goal_options_v1/goal_options.pt"
)
TEMPORAL_CONTROLLER_PATH = Path(
    "state/recurrent_goal_option_state_v1/"
    "recurrent_goal_option_state.pt"
)
EPISODIC_READER_PATH = Path(
    "state/multi_item_episodic_memory_v1/"
    "episodic_reader.pt"
)
EPISODIC_TEST_RESULT = Path(
    "state/multi_item_episodic_memory_test_v1/result.json"
)

EXPECTED_GOAL_DECODER_SHA = (
    "d46dfb425b9c3a08f9779c8d1b8519382140b465d6da37977ef1442bf1d36066"
)
EXPECTED_GOAL_OPTIONS_SHA = (
    "bcb729e2fa00db7b63af64a975d6cd2399c5e4711385d888b8131331587e2717"
)
EXPECTED_TEMPORAL_CONTROLLER_SHA = (
    "2e006c1f19f42f180d4bad9eb23c2f4f5a1188515df1d2227072f5b74aef2813"
)
EXPECTED_EPISODIC_READER_SHA = (
    "db86e989d7706348deaeb231c176120b1cdbe7288967f11dc5b218c70a2db097"
)


def verify_integration_provenance():
    expected = {
        GOAL_DECODER_PATH: EXPECTED_GOAL_DECODER_SHA,
        GOAL_OPTIONS_PATH: EXPECTED_GOAL_OPTIONS_SHA,
        TEMPORAL_CONTROLLER_PATH: EXPECTED_TEMPORAL_CONTROLLER_SHA,
        EPISODIC_READER_PATH: EXPECTED_EPISODIC_READER_SHA,
    }
    for path, wanted in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        got = sha256(path)
        if got != wanted:
            raise RuntimeError(
                f"provenance mismatch for {path}: {got} != {wanted}"
            )
    return {
        str(path): wanted
        for path, wanted in expected.items()
    }


@torch.no_grad()
def select_canonical_recall(
    reader,
    slot0: torch.Tensor,
    slot1: torch.Tensor,
    query: int,
):
    """
    Use frozen v0.23 query-conditioned attention to select from the ORIGINAL
    canonical recall128 vectors, not the reader's private encoded features.
    """
    if query not in (QUERY_FIRST, QUERY_SECOND):
        raise ValueError(
            "embodied selection currently supports FIRST/SECOND only"
        )

    slots = torch.cat(
        (slot0.unsqueeze(1), slot1.unsqueeze(1)),
        dim=1,
    )
    valid = torch.ones(
        slots.shape[0], 2, dtype=torch.bool, device=slots.device
    )
    q = torch.full(
        (slots.shape[0],),
        int(query),
        dtype=torch.long,
        device=slots.device,
    )

    out = reader(slots, valid, q)
    weights = out["attention"]
    selected = (
        weights.unsqueeze(-1) * slots
    ).sum(dim=1)

    return {
        "selected_recall": selected,
        "attention": weights,
        "reader_answer_logits": out["logits"],
    }


@torch.no_grad()
def decode_goal(decoder, canonical_recall):
    logits = decoder(canonical_recall)
    if float(logits.abs().max().item()) <= 1e-10:
        return None, logits
    return int(torch.argmax(logits, dim=-1)[0].item()), logits
