"""Caption extraction, checked on excerpts of the RAPTOR paper (2401.18059v1)
pages that previously produced false 'Figure 1' tags."""

from pathlib import Path

from pine.agent.captions import (
    caption_tokens,
    captions_from_markdown,
    extract_page_visuals,
    mentions_from_markdown,
    verify_hints,
)

PAGE1 = (
    "## 1 INTRODUCTION\n\n"
    "To address this, we design an indexing and retrieval system that uses a tree "
    "structure to capture both high-level and low-level details about a text. As shown "
    "in Figure 1, RAPTOR clusters chunks of text, generates text summaries of those "
    "clusters, and then repeats, generating a tree from the bottom up.\n"
)
PAGE2 = (
    "Published as a conference paper at ICLR 2024\n\n"
    "Figure 1: Tree construction process: RAPTOR recursively clusters chunks of text "
    "based on their vector embeddings and generates text summaries of those clusters, "
    "constructing a tree from the bottom up.\n\n<!-- image -->\n\nOur main contribution ...\n"
)
PAGE7 = (
    "Figure 4: Querying Process: Illustration of how RAPTOR retrieves information for "
    "two questions about the Cinderella story.\n\n<!-- image -->\n\n"
    "Table 1: NarrativeQA Performance With + Without RAPTOR: Performance comparison of "
    "various retrieval methods.\n\n| Model | ROUGE | BLEU-1 |\n|---|---|---|\n| SBERT | 22.5 | 16.1 |\n"
)
PAGE18 = (
    "## F QUALITATIVE ANALYSIS\n\n"
    "To qualitatively examine RAPTOR's retrieval process, we test it on thematic, multi-hop "
    "questions about a 1500-word version of the fairytale Cinderella, shown in Figure 1 and "
    "discussed with Table 12.\n"
)


def test_captions_only_from_caption_lines() -> None:
    assert captions_from_markdown(PAGE1)[:2] == ([], [])
    figs, tabs, lines = captions_from_markdown(PAGE2)
    assert figs == [
        "Figure 1: Tree construction process: RAPTOR recursively clusters chunks of text based on their vector embeddings and generates text summaries of those clusters, constructing a tree from the bottom up."[
            :200
        ]
    ]
    assert tabs == [] and lines == {2}
    figs, tabs, _ = captions_from_markdown(PAGE7)
    assert [f[:9] for f in figs] == ["Figure 4:"]
    assert [t[:8] for t in tabs] == ["Table 1:"]
    assert captions_from_markdown(PAGE18)[:2] == ([], [])


def test_mentions_are_separate_from_captions() -> None:
    assert mentions_from_markdown(PAGE1) == ([1], [])
    _, _, lines = captions_from_markdown(PAGE2)
    assert mentions_from_markdown(PAGE2, lines) == ([], [])
    assert mentions_from_markdown(PAGE18) == ([1], [12])


def test_model_hints_must_be_verbatim_on_the_page() -> None:
    invented = (
        "Figure 1: Not explicitly mentioned in the text but implied by the context"
    )
    real = (
        "Figure 4: Querying Process: Illustration of how RAPTOR retrieves information"
    )
    assert verify_hints([invented, real], PAGE7) == [real]
    assert verify_hints(["Figure 1: Tree construction process"], PAGE2) == [
        "Figure 1: Tree construction process"
    ]
    assert verify_hints(["Fig 1"], PAGE1) == []  # too short to be a caption


def test_extract_page_visuals_markdown_fallback(tmp_path: Path) -> None:
    pages = {1: PAGE1, 2: PAGE2, 7: PAGE7, 18: PAGE18}
    visuals = extract_page_visuals(tmp_path, pages, {2: 1, 7: 1})
    assert [p for p, v in visuals.items() if v.figures] == [2, 7]
    assert visuals[1].figures == [] and visuals[1].mentions_figures == [1]
    assert visuals[18].figures == [] and visuals[18].mentions_figures == [1]
    assert visuals[7].tables and visuals[7].tables[0].startswith("Table 1:")
    assert visuals[2].has_visual_elements and not visuals[1].has_visual_elements
    assert visuals[2].captions_source == "markdown"
    assert caption_tokens(visuals[2].figures + visuals[7].tables) == [
        "figure1",
        "table1",
    ]
