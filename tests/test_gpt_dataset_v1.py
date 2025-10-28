from build_llm_from_scratch.gpt_dataset_v1 import create_dataloader_v1

from inline_snapshot import snapshot
import pytest


@pytest.fixture
def raw_text():
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    yield raw_text


def test_create_dataloader_v1(raw_text: str):
    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    assert [el.tolist() for el in first_batch] == snapshot(
        [[[40, 367, 2885, 1464]], [[367, 2885, 1464, 1807]]]
    )

    second_batch = next(data_iter)
    assert [el.tolist() for el in second_batch] == snapshot(
        [[[367, 2885, 1464, 1807]], [[2885, 1464, 1807, 3619]]]
    )


def test_create_dataloader_v1_bigger_batch_size(raw_text: str):
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    assert inputs.tolist() == snapshot(
        [
            [40, 367, 2885, 1464],
            [1807, 3619, 402, 271],
            [10899, 2138, 257, 7026],
            [15632, 438, 2016, 257],
            [922, 5891, 1576, 438],
            [568, 340, 373, 645],
            [1049, 5975, 284, 502],
            [284, 3285, 326, 11],
        ]
    )
    assert targets.tolist() == snapshot(
        [
            [367, 2885, 1464, 1807],
            [3619, 402, 271, 10899],
            [2138, 257, 7026, 15632],
            [438, 2016, 257, 922],
            [5891, 1576, 438, 568],
            [340, 373, 645, 1049],
            [5975, 284, 502, 284],
            [3285, 326, 11, 287],
        ]
    )
