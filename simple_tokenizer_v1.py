import re


def build_vocab(text: str) -> dict[str, int]:
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_words = sorted(set(preprocessed))
    vocab = {token: integer for integer, token in enumerate(all_words)}
    return vocab


class SimpleTokenizerV1:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.str2int = vocab
        self.int2str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str2int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int2str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


if __name__ == "__main__":
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    vocab = build_vocab(raw_text)
    tokenizer = SimpleTokenizerV1(vocab=vocab)
    text = """It's the last he painted, you know,"
            Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(f"{ids=}")
    print(f"{tokenizer.decode(ids)=}")
    try:
        text = "Hello, do you like tea?"
        print(f"{tokenizer.encode(text)=}")
    except KeyError as e:
        print(f"KeyError: {e}")
