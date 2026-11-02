"""Claude's SUPPORT-1 analyzer: comparison helpers. Regression for the bug Sol's independent analyzer exposed on the tiny
end-to-end run (D4 disagreed): `(P or 9) - 0.10` coerced a legitimate P value of 0.0 to 9, an impossible bar."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import support1_analyze as A


def test_zero_is_a_value_not_missing():
    assert A.ge(0.0, A.minus(0.0, 0.10)) and A.ge(0.05, A.minus(0.10, 0.10))
    assert A.le(0.0, 0.10) and A.ge(0.8, 0.80) and A.ge(0.7 - 0.4, 0.3)


def test_missing_on_either_side_fails():
    assert not A.ge(None, 0.5) and not A.ge(0.9, None) and not A.ge(0.9, A.minus(None, 0.1))
    assert not A.le(None, 0.1) and not A.le(0.0, None)


def test_pooled_missing_seed_is_missing():
    assert A.pooled([0.5, None]) is None and A.pooled([0.0, 0.0]) == 0.0 and A.pooled([]) is None


def test_frozen_code_hash_unchanged():
    import support1
    assert support1.code_hash() == A.FROZEN_CODE_HASH
