import tiktoken

with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(raw_text)

print(len(enc_text))
enc_sample = enc_text[50:]
# print(enc_sample)

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1 : context_size + 1]
print(f"{x=}")
print(f"{y=}")

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    # print(context, "---->", desired)
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
