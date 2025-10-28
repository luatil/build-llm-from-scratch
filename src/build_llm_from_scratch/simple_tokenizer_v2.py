import re


def build_vocab(text: str) -> dict[str, int]:
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    return vocab


class SimpleTokenizerV2:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.str2int = vocab
        self.int2str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str2int else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str2int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int2str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text
