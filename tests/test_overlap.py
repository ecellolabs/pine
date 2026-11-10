"""MMLongBench overlap check: a shared document is caught, unrelated ones are not."""

from pine.training.overlap import find_overlap, shingles


def _text(seed: int, n: int = 3000) -> str:
    # deterministic pseudo-random words, so different seeds share no 8-grams
    x, words = seed, []
    for _ in range(n):
        x = (x * 1103515245 + 12345) % 2**31
        words.append(f"w{x % 5000}")
    return " ".join(words)


def test_find_overlap() -> None:
    brochure = _text(1, 1500)
    vidore = {
        "report_with_brochure": _text(2) + "\n" + brochure.upper(),  # case is ignored
        "unrelated": _text(3),
    }
    other = {"brochure.pdf": brochure, "other.pdf": _text(4)}
    report = find_overlap(vidore, other, threshold=0.3)
    assert report["exclude"] == ["report_with_brochure"]
    top = report["matches"][0]
    assert top["other"] == "brochure.pdf" and top["containment"] > 0.9
    assert report["matches"][1].get("containment", 0) < 0.05
    assert shingles("too short") == set()
