import numpy as np
import torch
from torch.nn.functional import sigmoid
from transformer_heads.model.model import HeadedModel
from transformer_heads.output import HeadedModelOutput
from transformers import TrainerCallback
from transformers.modeling_outputs import CausalLMOutputWithPast

from ykutil.constants import IGNORE_INDEX


@torch.inference_mode()
def compute_policy_metrics(
    outputs: CausalLMOutputWithPast,
    input_dic: dict[str, torch.Tensor],
    model: torch.nn.Module,
):
    labels = input_dic["labels"]
    metrics = {}
    distrib = torch.softmax(outputs.logits.detach(), dim=-1)
    correctness = (distrib[:, :-1].argmax(-1) == labels[:, 1:]).float()
    correctness = correctness[labels[:, 1:] != IGNORE_INDEX]
    metrics["policy_accuracy"] = correctness.mean().item()
    metrics["policy_entropy"] = -torch.mean(
        torch.sum(distrib * torch.log(distrib), dim=-1)
    ).item()
    return metrics


@torch.inference_mode()
def compute_classification_head_metrics(
    outputs: HeadedModelOutput,
    input_dic: dict[str, torch.Tensor],
    model: HeadedModel,
):
    metrics = {}

    for key, value in outputs.loss_by_head.items():
        metrics[f"loss_{key}"] = float(value)
        if (
            outputs.adapted_loss_by_head is not None
            and key in outputs.adapted_loss_by_head
        ):
            adapted = outputs.adapted_loss_by_head[key]
            if adapted != value:
                metrics[f"adapted_loss_{key}"] = float(adapted)

    if "lm_head" in outputs.preds_by_head:
        preds = outputs.preds_by_head["lm_head"].detach().argmax(-1)
        labels = input_dic["labels"]
        correctness = (preds[:, :-1] == labels[:, 1:]).float()
        correctness = correctness[labels[:, 1:] != IGNORE_INDEX]
        metrics["policy_accuracy"] = correctness.mean().item()

    for head_name, preds in outputs.preds_by_head.items():
        if head_name == "lm_head":
            continue
        labs = input_dic[model.head_configs[head_name].target]
        preds = sigmoid(preds[labs != IGNORE_INDEX]).detach()
        labs = labs[labs != IGNORE_INDEX]
        preds_round = torch.round(preds)
        correctness = (preds_round == labs).float()
        precision = (preds_round * labs).sum() / preds_round.sum()
        recall = (preds_round * labs).sum() / labs.sum()
        f1 = 2 * precision * recall / (precision + recall)
        metrics[f"{head_name}_accuracy"] = correctness.mean().item()
        metrics[f"{head_name}_bias"] = preds.mean().item()
        metrics[f"{head_name}_std"] = preds.std().item()
        metrics[f"{head_name}_precision"] = precision.item()
        metrics[f"{head_name}_recall"] = recall.item()
        metrics[f"{head_name}_f1"] = f1.item()

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
    compute_classification_head_metrics=compute_classification_head_metrics,
    compute_policy_metrics=compute_policy_metrics,
)
