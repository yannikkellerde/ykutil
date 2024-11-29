import torch
from torch.nn.functional import log_softmax
from transformer_heads.output import HeadedModelOutput
from transformers import PreTrainedModel


@torch.inference_mode()
def compute_seq_log_probability(
    model: PreTrainedModel,
    pre_seq_tokens: list[int],
    post_seq_tokens: list[int],
) -> float:
    inputs = torch.tensor(
        pre_seq_tokens + post_seq_tokens, device=model.device
    ).unsqueeze(0)

    output = model(inputs)
    if isinstance(output, HeadedModelOutput):
        logits = output.preds_by_head["lm_head"][
            0, -len(post_seq_tokens) - 1 : -1
        ].cpu()
    else:
        logits = output.logits[0, -len(post_seq_tokens) - 1 : -1].cpu()

    logprobs = log_softmax(logits, dim=-1)

    return float(
        torch.exp(
            torch.sum(
                logprobs.gather(1, torch.tensor(post_seq_tokens).unsqueeze(1)).squeeze()
            )
        )
    )
