"""Evaluation metrics for document question answering (ANLS, F1, Exact Match)."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes the Levenshtein edit distance between two strings using DP."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def normalize_answer(text: str) -> str:
    """Standard SQuAD/DocVQA text normalization: lowercase, strip punctuation and articles."""
    text = text.lower().translate(_PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", text).split())


def _single_anls(prediction: str, target: str, threshold: float = 0.5) -> float:
    """Computes Normalized Levenshtein Similarity for a single target.
    Case-insensitive, space-sensitive."""
    p = prediction.strip().lower()
    t = target.strip().lower()

    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0

    max_len = max(len(p), len(t))
    dist = levenshtein_distance(p, t)
    norm_dist = dist / max_len

    return 1.0 - norm_dist if norm_dist < threshold else 0.0


def compute_anls(
    prediction: str,
    targets: str | Sequence[str],
    threshold: float = 0.5,
) -> float:
    """Computes ANLS score between prediction and ground-truth targets.
    Takes the maximum score across all valid target answers."""
    if isinstance(targets, str):
        target_list = [targets]
    else:
        target_list = [t for t in targets if t is not None]

    if not target_list:
        return 0.0

    return max(_single_anls(prediction, t, threshold=threshold) for t in target_list)


def _single_f1(prediction: str, target: str) -> float:
    """Computes token-level F1 overlap score between prediction and a single target."""
    pred_tokens = normalize_answer(prediction).split()
    target_tokens = normalize_answer(target).split()

    if not pred_tokens or not target_tokens:
        return float(pred_tokens == target_tokens)

    common = sum((Counter(pred_tokens) & Counter(target_tokens)).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(target_tokens)
    return 2.0 * precision * recall / (precision + recall)


def compute_f1(
    prediction: str,
    targets: str | Sequence[str],
) -> float:
    """Computes token-level F1 score, taking the maximum over all reference targets."""
    if isinstance(targets, str):
        target_list = [targets]
    else:
        target_list = [t for t in targets if t is not None]

    if not target_list:
        return 0.0

    return max(_single_f1(prediction, t) for t in target_list)


def compute_exact_match(
    prediction: str,
    targets: str | Sequence[str],
) -> float:
    """Computes binary exact match after text normalization, taking the maximum over targets."""
    if isinstance(targets, str):
        target_list = [targets]
    else:
        target_list = [t for t in targets if t is not None]

    if not target_list:
        return 0.0

    norm_pred = normalize_answer(prediction)
    return float(any(norm_pred == normalize_answer(t) for t in target_list))
