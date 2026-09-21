import pytest

from typed_decisions.open_jev import decide_request


class FakeModel:
    config = {
        "base_model": "measured/model",
        "pool": "span",
        "temperature": 1.05,
        "train": {"questions": 42},
        "metrics": {"ood": {"acc": 0.69}},
    }

    def decide(self, state, questions):
        assert state == "ontology state"
        assert questions[0]["type"] == "choice"
        return [{"choice": "a", "probabilities": {"a": 0.9, "b": 0.1}, "confidence": 0.9}]


def test_json_surface_is_typed_and_provenance_bearing():
    result = decide_request(
        FakeModel(),
        {"state": "ontology state", "questions": [{"type": "choice", "instructions": "pick", "options": ["a", "b"]}]},
    )
    assert result["kind"] == "typed-decisions/open-jev-v1"
    assert result["generated_text"] is False
    assert result["model"]["base_model"] == "measured/model"
    assert result["decisions"][0]["choice"] == "a"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"state": "", "questions": [{}]},
        {"state": "x", "questions": []},
        {"state": "x", "questions": [{}], "prompt": "forbidden"},
    ],
)
def test_json_surface_rejects_open_or_empty_requests(payload):
    with pytest.raises(ValueError):
        decide_request(FakeModel(), payload)
