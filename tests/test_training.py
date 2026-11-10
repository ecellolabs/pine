"""STaR data from ViDoRe: candidate pages and the short-answer rewrite parser."""

import json

import pymupdf

from pine.training.vidore import candidate_pages, keep_type, parse_short, scanned_pages


def test_candidate_pages_and_short_answers() -> None:
    pages = {p: f"filler page {p}" for p in range(1, 31)}
    pages[7] = pages[21] = "temporary contracts of mobile workers"
    gold = [25, 3]
    q = "temporary contracts mobile workers"
    with_gold = candidate_pages(q, pages, gold, 10, answerable=True)
    assert len(with_gold) == 10 and {3, 7, 21, 25} <= set(with_gold)
    assert with_gold == sorted(with_gold)
    without = candidate_pages(q, pages, gold, 10, answerable=False, exclude={21})
    assert len(without) == 10 and 7 in without
    assert not {2, 3, 4, 24, 25, 26, 21} & set(without)  # gold, its neighbours, a scan

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
    paragraph = " ".join(["word"] * 11)
    assert parse_short(f'{{"answer": "{paragraph}", "answer_format": "Str"}}') == (
        None,
        None,
    )
    long_item = json.dumps({"answer": ["short", paragraph], "answer_format": "List"})
    assert parse_short(long_item) == (None, None)

    pdf = pymupdf.open()
    pdf.new_page().insert_text(
        (72, 72), "A born-digital page with a real text layer. " * 3
    )
    scan = pdf.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 80), False)
    pix.clear_with(200)
    scan.insert_image(scan.rect, pixmap=pix)
    pdf.new_page()  # blank: no text, no image, so not a scan
    assert scanned_pages(pdf) == {2}
