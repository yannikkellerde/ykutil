import numpy as np
import torch
from transformer_heads.output import HeadedModelOutput
from transformers import TrainerCallback
from transformers.modeling_outputs import CausalLMOutputWithPast

from ykutil.constants import IGNORE_INDEX


def compute_policy_metrics(
    outputs: CausalLMOutputWithPast,
    labels: torch.Tensor,
    values=None,  # pylint: disable=W0613
):
    metrics = {}
    distrib = torch.softmax(outputs.logits.detach(), dim=-1)
    correctness = (distrib[:, :-1].argmax(-1) == labels[:, 1:]).float()
    correctness = correctness[labels[:, 1:] != IGNORE_INDEX]
    metrics["policy_accuracy"] = correctness.mean().item()
    metrics["policy_entropy"] = -torch.mean(
        torch.sum(distrib * torch.log(distrib), dim=-1)
    ).item()
    return metrics


def compute_head_metrics(
    outputs: HeadedModelOutput, labels: torch.Tensor, values: torch.Tensor
):
    metrics = {}

    for key, value in outputs.loss_by_head.items():
        metrics[f"loss_{key}"] = float(value)

    for key, value in outputs.adapted_loss_by_head.items():
        metrics[f"adapted_loss_{key}"] = float(value)

    if "lm_head" in outputs.preds_by_head:
        preds = outputs.preds_by_head["lm_head"].detach().argmax(-1)
        correctness = (preds[:, :-1] == labels[:, 1:]).float()
        correctness = correctness[labels[:, 1:] != IGNORE_INDEX]
        metrics["policy_accuracy"] = correctness.mean().item()

    if "value_head" in outputs.preds_by_head:
        preds = torch.round(outputs.preds_by_head["value_head"][:, :, 0].detach())
        correctness = (preds == values).float()
        mask = torch.logical_or(values == 0.0, values == 1.0)
        correctness = correctness[mask]
        metrics["value_accuracy"] = correctness.mean().item()
        metrics["value_bias"] = outputs.preds_by_head["value_head"].mean().item()
        metrics["value_std"] = outputs.preds_by_head["value_head"].std().item()

    return metrics


def compute_metric(pred: float, targ: float, metric_type: str):
    pred = float(pred)
    targ = float(targ)
    match metric_type:
        case "mae":
            return abs(pred - targ)
        case "mse":
            return (pred - targ) ** 2
        case "acc":
            return int(round(pred) == targ)
        case "bce":
            return -targ * np.log(pred) - (1 - targ) * np.log(1 - pred)
        case "_":
            raise ValueError(f"Unknown metric type {metric_type}")


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0 and args.logging_first_step:
            control.should_evaluate = True


compute_metrics_functions = dict(
    compute_head_metrics=compute_head_metrics,
    compute_policy_metrics=compute_policy_metrics,
)
