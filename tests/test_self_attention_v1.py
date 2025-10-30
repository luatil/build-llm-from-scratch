from build_llm_from_scratch.self_attention_v1 import SelfAttentionV1

import torch
from inline_snapshot import snapshot

torch.manual_seed(123)  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]


def test_self_attention_v1():
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
    model = SelfAttentionV1(d_in=3, d_out=2)
    y: torch.Tensor = model(inputs)  # pyright: ignore[reportAny]
    assert y.tolist() == snapshot(  # pyright: ignore[reportUnknownMemberType]
        [
            [-0.6421990990638733, 1.3542132377624512],
            [-0.6208996176719666, 1.3454437255859375],
            [-0.627853512763977, 1.3491147756576538],
            [-0.6875931620597839, 1.3777376413345337],
            [-0.8373287916183472, 1.4805678129196167],
            [-0.6094250679016113, 1.3358074426651],
        ]
    )
