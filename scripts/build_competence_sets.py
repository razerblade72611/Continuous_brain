"""Build the fixed competence sentence sets from public-domain Gutenberg texts (NLTK sample corpus).

Source: https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip
Usage: python scripts/build_competence_sets.py <dir with gutenberg/*.txt>
Dev set (used only to accept/reject sleep): chesterton-thursday.
Eval set (used only for the gate): chesterton-brown + bryant-stories.
Modern eval set (gate, out-of-domain for rehearsal): Brown corpus, 200 sentences.
Sealed-only confirmation sets: chesterton-ball + edgeworth-parents (Gutenberg); Brown items 200-399.
Background set (optional sleep rehearsal, never scored): austen-emma + carroll-alice.
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


def brown_sentences(brown_dir: Path):
    """Brown corpus (1961 American English, all genres): tagged lines -> plain sentences."""
    for f in sorted(brown_dir.glob("c[a-r][0-9][0-9]")):
        for line in f.read_text(encoding="latin-1").splitlines():
            toks = [t.rsplit("/", 1)[0] for t in line.split() if "/" in t]
            if not toks:
                continue
            s = " ".join(toks).replace("``", '"').replace("''", '"')
            s = re.sub(r" ([.,;:!?')])", r"\1", s).replace("( ", "(").replace(" n't", "n't").replace(" 's", "'s")
            words = s.split()
            if 10 <= len(words) <= 35 and s.isascii() and s[:1].isupper() and not s.isupper():
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
    bg = pick([src / "austen-emma.txt", src / "carroll-alice.txt"], 300, 3)
    modern = sorted(set(brown_sentences(src.parent / "brown")))
    random.Random(4).shuffle(modern)
    (out / "competence_modern.txt").write_text("\n".join(modern[:200]) + "\n")
    # Sealed-only confirmation sets (Sol, DAV-104): disjoint from every set above, built here and never
    # printed or measured in development. Only the sealed run reads them.
    used = set(dev) | set(ev) | set(bg)
    gut = [x for x in pick([src / "chesterton-ball.txt", src / "edgeworth-parents.txt"], 10_000, 5) if x not in used]
    (out / "confirm_gutenberg.txt").write_text("\n".join(gut[:200]) + "\n")
    (out / "confirm_modern.txt").write_text("\n".join(modern[200:400]) + "\n")
    (out / "background.txt").write_text("\n".join(bg) + "\n")
    (out / "competence_dev.txt").write_text("\n".join(dev) + "\n")
    (out / "competence_eval.txt").write_text("\n".join(ev) + "\n")
    print(len(dev), len(ev))


if __name__ == "__main__":
    main()
