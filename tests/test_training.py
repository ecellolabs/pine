"""STaR data from ViDoRe: candidate pages and the short-answer rewrite parser."""

from pine.training.vidore import candidate_pages, keep_type, parse_short


def test_candidate_pages_and_short_answers() -> None:
    pages = {p: f"filler page {p}" for p in range(1, 31)}
    pages[7] = pages[21] = "temporary contracts of mobile workers"
    gold = [25, 3]
    q = "temporary contracts mobile workers"
    with_gold = candidate_pages(q, pages, gold, 10, answerable=True)
    assert len(with_gold) == 10 and {3, 7, 21, 25} <= set(with_gold)
    assert with_gold == sorted(with_gold)
    without = candidate_pages(q, pages, gold, 10, answerable=False)
    assert len(without) == 10 and {7, 21} <= set(without) and not {3, 25} & set(without)

    assert keep_type(["extractive", "numerical"]) and not keep_type(
        ["open-ended", "numerical"]
    )
    assert parse_short(
        '```json\n{"answer": "115 days", "answer_format": "Str"}\n```'
    ) == (
        "115 days",
        "Str",
    )
    assert parse_short('{"answer": ["DE", "ES"], "answer_format": "List"}') == (
        '["DE", "ES"]',
        "List",
    )
    assert parse_short('{"answer": null, "answer_format": null}') == (None, None)
    assert parse_short("no json here") == (None, None)
