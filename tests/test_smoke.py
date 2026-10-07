from atria_core.datasets import datasets

from pine.datasets.mmlongbench_doc import MMLongBenchDocConfig
from pine.datasets.mpdocvqa import MPDocVQAConfig
from pine.datasets.slidevqa import SlideVQAConfig


def test_registered_datasets() -> None:
    registered = datasets.list()
    assert "slidevqa" in registered
    assert "mpdocvqa" in registered
    assert "mmlongbench_doc" in registered


def test_dataset_configs() -> None:
    slide_cfg = SlideVQAConfig(max_samples=5)
    assert slide_cfg.max_samples == 5

    mm_cfg = MMLongBenchDocConfig(max_samples=10)
    assert mm_cfg.max_samples == 10

    mp_cfg = MPDocVQAConfig(max_samples=2)
    assert mp_cfg.max_samples == 2
