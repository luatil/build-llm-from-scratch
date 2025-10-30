import torch


def main():
    _ = torch.manual_seed(123)  # pyright: ignore[reportUnknownMemberType]

    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],
        ]  # step     (x^6)
    )

    d_in = inputs.shape[1]
    d_out = 2

    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    query = inputs @ W_query
    keys = inputs @ W_key
    values = inputs @ W_value

    attn_scores = query @ keys.T
    d_k = keys.shape[-1]
    attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)  # pyright: ignore[reportAny]
    context_vec = attn_weights @ values

    return context_vec


if __name__ == "__main__":
    _ = main()
