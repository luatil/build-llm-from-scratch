from build_llm_from_scratch.simple_tokenizer_v2 import SimpleTokenizerV2, build_vocab

from inline_snapshot import snapshot


def test_simple_tokenizer_v2():
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    vocab = build_vocab(raw_text)
    tokenizer = SimpleTokenizerV2(vocab=vocab)
    text = """It's the last he painted, you know,"
            Mrs. Gisburn said with pardonable pride."""
    # ids = tokenizer.encode(text)
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    assert tokenizer.encode(text) == snapshot(
        [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]
    )
    assert tokenizer.decode(tokenizer.encode(text)) == snapshot(
        "<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>."
    )
