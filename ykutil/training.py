import os

from transformers import Trainer


def train_eval_and_get_metrics(trainer: Trainer, checkpoint=None):
    all_metrics = {
        "run_name": os.environ["WANDB_NAME"] if "WANDB_NAME" in os.environ else "tmp"
    }

    print("*** Train ***")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    all_metrics.update(metrics)

    print("*** Evaluate ***")
    metrics = trainer.evaluate(metric_key_prefix="eval")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    all_metrics.update(metrics)

    return all_metrics
