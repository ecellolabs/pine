from pine.agent.scoring import anls, eval_score, extract_answer


def test_str_uses_anls_with_threshold() -> None:
    gold = "National Atmospheric Research Laboratory"
    assert (
        eval_score(gold, "National Atmospheric Research Laboratory (NARL)", "Str")
        == 1.0
    )
    assert eval_score(gold, "national atmospheric research lab", "Str") > 0.5
    assert eval_score(gold, "Berlin", "Str") == 0.0
    assert anls("abc", "abc") == 1.0
    assert anls("abc", "xyz") == 0.0


def test_int_and_float_rules() -> None:
    assert eval_score("3", "3", "Int") == 1.0
    assert eval_score("3", "three", "Int") == 0.0
    assert eval_score("51%", "51%", "Float") == 1.0
    assert eval_score("51%", "0.51", "Float") == 1.0  # percentage tolerance
    assert eval_score("51%", "50%", "Float") == 0.0  # 1% relative tolerance
    assert eval_score("17.5", "17.5", "Float") == 1.0


def test_not_answerable_and_list_rules() -> None:
    assert eval_score("Not answerable", "Not answerable", "None") == 1.0
    assert eval_score("Not answerable", "green", "None") == 0.0
    assert eval_score("green", "Not answerable", "Str") == 0.0
    assert eval_score("['green', 'yellow']", "green and yellow", "List") == 1.0
    assert eval_score("['green', 'yellow']", "['yellow', 'green']", "List") == 1.0
    assert eval_score("['green', 'yellow']", "green", "List") == 0.0
    assert eval_score("[16, 19, 25]", "[25, 16, 19]", "List") == 1.0


def test_extract_answer_normalisation() -> None:
    assert extract_answer("The answer is: 51%.") == "51%"
    assert (
        extract_answer("The document does not contain this information.")
        == "Not answerable"
    )
    assert extract_answer("Not answerable") == "Not answerable"
