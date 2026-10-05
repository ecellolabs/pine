import json
from pathlib import Path

import pytest

from pipeline_v2.sampling import (
    SAMPLE_SETS_DIR,
    SampleSet,
    dataset_load_kwargs,
    describe,
    resolve_max_samples,
)


def test_integer_max_samples_unchanged() -> None:
    assert resolve_max_samples(None) is None
    assert resolve_max_samples(3) == 3
    assert resolve_max_samples("5") == 5
    assert dataset_load_kwargs(4, "slidevqa") == {"max_samples": 4}
    assert dataset_load_kwargs(None, "slidevqa") == {}
    with pytest.raises(ValueError):
        resolve_max_samples("0")


def test_s0_sample_set_ships_with_repo() -> None:
    s0 = resolve_max_samples("s0")
    assert isinstance(s0, SampleSet)
    assert s0.name == "s0"
    assert s0.dataset == "mmlongbench_doc"
    assert len(s0.samples) == 3
    assert len(s0.doc_ids) == 3
    assert all(s.question for s in s0.samples)
    assert s0.source == (SAMPLE_SETS_DIR / "s0.json").resolve()
    assert dataset_load_kwargs(s0, "mmlongbench_doc") == {"doc_ids": s0.doc_ids}
    assert "s0" in describe(s0)


def test_sample_set_from_path_and_doc_filtering(tmp_path: Path) -> None:
    data = {
        "name": "mini",
        "dataset": "mmlongbench_doc",
        "samples": [
            {"run_id": "a", "doc_id": "x.pdf", "question": "Q1?"},
            {"run_id": "b", "doc_id": "x.pdf", "question": "Q2?"},
            {"run_id": "c", "doc_id": "y.pdf"},
        ],
    }
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(data))
    resolved = resolve_max_samples(str(path))
    assert isinstance(resolved, SampleSet)
    assert resolved.doc_ids == ["x.pdf", "y.pdf"]
    assert [s.run_id for s in resolved.questions_for("x.pdf")] == ["a", "b"]
    assert resolved.questions_for("y.pdf")[0].question is None
    # same set resolvable by name when the search dir is given
    by_name = SampleSet.load("mini", search_dir=tmp_path)
    assert by_name.name == "mini"


def test_sample_set_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_max_samples("does-not-exist")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "bad", "samples": []}))
    with pytest.raises(ValueError):
        resolve_max_samples(str(bad))
    other = SampleSet.from_dict(
        {"name": "o", "dataset": "slidevqa", "samples": [{"doc_id": "d"}]}
    )
    with pytest.raises(ValueError):
        dataset_load_kwargs(other, "mmlongbench_doc")
    with pytest.raises(ValueError):
        dataset_load_kwargs(
            SampleSet.from_dict({"name": "p", "samples": [{"doc_id": "d"}]}), "slidevqa"
        )
