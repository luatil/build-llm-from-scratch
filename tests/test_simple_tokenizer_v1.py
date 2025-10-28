from build_llm_from_scratch.simple_tokenizer_v1 import SimpleTokenizerV1, build_vocab

from inline_snapshot import snapshot
import pytest


def test_simple_tokenizer_v1():
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    vocab = build_vocab(raw_text)
    tokenizer = SimpleTokenizerV1(vocab=vocab)
    text = """It's the last he painted, you know,"
            Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    assert ids == snapshot(
        [
            56,
            2,
            850,
            988,
            602,
            533,
            746,
            5,
            1126,
            596,
            5,
            1,
            67,
            7,
            38,
            851,
            1108,
            754,
            793,
            7,
        ]
    )
    assert tokenizer.decode(ids) == snapshot(
        "It' s the last he painted, you know,\" Mrs. Gisburn said with pardonable pride."
    )
    with pytest.raises(KeyError):
        text = "Hello, do you like tea?"
        tokenizer.encode(text)
