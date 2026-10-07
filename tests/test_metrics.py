import pytest

from pine.metrics import (
    compute_anls,
    compute_exact_match,
    compute_f1,
    levenshtein_distance,
    normalize_answer,
)


def test_levenshtein_distance() -> None:
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("abc", "abc") == 0
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("flaw", "lawn") == 2
    assert levenshtein_distance("", "test") == 4
    assert levenshtein_distance("test", "") == 4


def test_normalize_answer() -> None:
    assert normalize_answer("The Apple, Inc.") == "apple inc"
    assert normalize_answer("  An  orange.  ") == "orange"
    assert normalize_answer("A bird!") == "bird"
    assert normalize_answer("10.5%") == "105"


def test_compute_anls_identical() -> None:
    assert compute_anls("Revenue was $10M", "Revenue was $10M") == 1.0
    # Case insensitivity
    assert compute_anls("apple", "Apple") == 1.0


def test_compute_anls_small_typo() -> None:
    # "apple" vs "aple": dist=1, max_len=5, nls=0.2 < 0.5 => 1 - 0.2 = 0.8
    score = compute_anls("aple", "apple")
    assert round(score, 2) == 0.8


def test_compute_anls_threshold_failure() -> None:
    # "apple" vs "banana": dist=5, max_len=6, nls=5/6 > 0.5 => 0.0
    assert compute_anls("apple", "banana") == 0.0


def test_compute_anls_multiple_targets() -> None:
    # Prediction matches the second target
    targets = ["wrong answer", "correct answer"]
    score = compute_anls("correct answer", targets)
    assert score == 1.0


def test_compute_f1() -> None:
    assert compute_f1("apple banana", "apple banana") == 1.0
    assert compute_f1("apple banana cherry", "apple banana") == pytest.approx(4 / 5)
    assert compute_f1("dog", "cat") == 0.0
    assert compute_f1("the apple", "an apple") == 1.0  # normalized articles removed


def test_compute_exact_match() -> None:
    assert compute_exact_match("The Apple", "apple") == 1.0
    assert compute_exact_match("apple pie", "apple") == 0.0
    assert compute_exact_match("apple", ["orange", "the apple"]) == 1.0
