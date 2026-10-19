"""Build the fixed competence sentence sets from public-domain Gutenberg texts (NLTK sample corpus).

Source: https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip
Usage: python scripts/build_competence_sets.py <dir with gutenberg/*.txt>
Dev set (used only to accept/reject sleep): chesterton-thursday.
Eval set (used only for the gate): chesterton-brown + bryant-stories.
"""
import random
import re
import sys
from pathlib import Path


def sentences(path: Path):
    text = path.read_text(encoding="latin-1")
    text = re.sub(r"\s+", " ", text)
    for s in re.split(r"(?<=[.!?])\s+", text):
        s = s.strip().strip('"').strip()
        words = s.split()
        if 10 <= len(words) <= 35 and s.isascii() and s[:1].isupper() and not s.isupper() and "[" not in s:
            yield s


def pick(paths, n, seed):
    pool = []
    for p in paths:
        pool.extend(sentences(p))
    pool = sorted(set(pool))
    random.Random(seed).shuffle(pool)
    return pool[:n]


def main():
    src = Path(sys.argv[1])
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    dev = pick([src / "chesterton-thursday.txt"], 100, 1)
    ev = pick([src / "chesterton-brown.txt", src / "bryant-stories.txt"], 200, 2)
    (out / "competence_dev.txt").write_text("\n".join(dev) + "\n")
    (out / "competence_eval.txt").write_text("\n".join(ev) + "\n")
    print(len(dev), len(ev))


if __name__ == "__main__":
    main()
