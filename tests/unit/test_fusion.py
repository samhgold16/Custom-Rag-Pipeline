import pytest

from localrag.retrieval.fusion import reciprocal_rank_fusion


def test_hand_computed_scores_and_order():
    # a: 1/61 + 1/62 ; b: 1/62 + 1/61 ; c: 1/63 ; d: 1/63
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], c=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)


def test_agreement_beats_a_single_first_place():
    fused = reciprocal_rank_fusion([["x", "shared"], ["y", "shared"]])
    assert fused[0][0] == "shared"


def test_weights_shift_the_winner():
    lists = [["lexical"], ["dense"]]
    assert reciprocal_rank_fusion(lists, weights=[0.8, 0.2])[0][0] == "lexical"
    assert reciprocal_rank_fusion(lists, weights=[0.2, 0.8])[0][0] == "dense"


def test_ties_keep_first_seen_order():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert [k for k, _ in fused] == ["a", "b"]


def test_duplicates_within_a_list_count_once_at_best_rank():
    fused = dict(reciprocal_rank_fusion([["a", "a", "b"]], c=0))
    assert fused == {"a": 1.0, "b": pytest.approx(1 / 3)}


def test_argument_validation():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 1.0])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], c=-1)
    assert reciprocal_rank_fusion([]) == []
