"""
Smoke tests for the training loop
"""

import json
import math
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.model import DEFAULT_MODEL, HallucinationClassifier
from src.train import (
    TrainConfig,
    _loader,
    check_paired_init,
    evaluate,
    run_training,
)

SEED = 42
CONDITION = "B"
TRAIN_N = 26
VAL_N = 12
PHYSICAL_BATCH = 2
EFFECTIVE_BATCH = 4
MAX_EPOCHS = 10
LEARNING_RATE = 5e-5
MAX_LENGTH = 256

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


SUMMARY_SOURCE = (
    "The cafe on the corner reopened in January after a long refit. It seats forty people "
    "and serves breakfast until noon."
)
QA_SOURCE = {
    "question": "when did the cafe reopen",
    "passages": "passage 1: The cafe on the corner reopened in January after a refit.",
}


def make_record(i, fold):
    # Synthetic rather than a real subset: short texts keep the smoke run to a couple of
    # minutes, and the signal is obvious enough that a working loop should memorize it.
    label = i % 2
    if i % 4 < 2:
        response = ("The cafe reopened in January and seats forty people."
                    if label == 0 else
                    "The cafe reopened in March and seats two hundred people.")
        return {"id": f"{fold}-{i}", "fold": fold, "label": label,
                "task_type": "Summary", "response": response,
                "source_info": SUMMARY_SOURCE}
    response = ("It reopened in January." if label == 0
                else "It reopened in 1998, shortly after a kitchen fire.")
    return {"id": f"{fold}-{i}", "fold": fold, "label": label,
            "task_type": "QA", "response": response, "source_info": QA_SOURCE}


def make_records(n, fold):
    return [make_record(i, fold) for i in range(n)]


class RecordingClassifier(HallucinationClassifier):
    """Same model, but it notes the mode of every forward and keeps its initial weights."""

    built = []

    def __init__(self, model_name=DEFAULT_MODEL):
        super().__init__(model_name)
        self.forward_modes = []
        watched = [n for n, p in self.named_parameters()
                   if n.startswith("encoder.") and p.requires_grad and p.dim() >= 2]
        self.watched = watched[:2]
        self.initial = {n: p.detach().clone() for n, p in self.named_parameters()
                        if n in self.watched}
        RecordingClassifier.built.append(self)

    def forward(self, input_ids, attention_mask):
        self.forward_modes.append((self.training, torch.is_grad_enabled()))
        return super().forward(input_ids, attention_mask)


def main():
    device = pick_device()
    print(f"device {device}, model {DEFAULT_MODEL}")
    print("the first run downloads the encoder, later runs use the cache\n")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    train_records = make_records(TRAIN_N, "train")
    val_records = make_records(VAL_N, "val")

    print("paired init check (two models at one seed)")
    try:
        check_paired_init(DEFAULT_MODEL, SEED)
        check("paired init gives identical starting weights", True)
    except AssertionError as exc:
        check("paired init gives identical starting weights", False, str(exc))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config = TrainConfig(
            condition=CONDITION,
            seed=SEED,
            learning_rate=LEARNING_RATE,
            checkpoint_path=tmp / "best.pt",
            val_logits_path=tmp / "val_logits.jsonl",
            test_logits_path=tmp / "test_logits.jsonl",
            effective_batch=EFFECTIVE_BATCH,
            physical_batch=PHYSICAL_BATCH,
            max_length=MAX_LENGTH,
            max_epochs=MAX_EPOCHS,
            # No early stopping, the point here is to let it overfit.
            patience=MAX_EPOCHS,
            device=device,
            precision="fp32",
        )

        print("\ntest leak guard")
        try:
            run_training(config, train_records, val_records, tokenizer,
                         test_records=val_records)
            check("dev run refuses test records", False, "no error was raised")
        except ValueError as exc:
            check("dev run refuses test records", True, f"raised: {exc}")

        print(f"\ntraining on {TRAIN_N} examples for {MAX_EPOCHS} epochs")
        import src.train as train_module

        original = train_module.HallucinationClassifier
        train_module.HallucinationClassifier = RecordingClassifier
        try:
            result = run_training(config, train_records, val_records, tokenizer)
        finally:
            train_module.HallucinationClassifier = original

        model = RecordingClassifier.built[0]

        print()
        first_loss = result.history[0]["train_loss"]
        last_loss = result.history[-1]["train_loss"]
        check(
            "1. train loss falls on a tiny subset",
            last_loss < 0.5 * first_loss,
            f"{first_loss:.4f} -> {last_loss:.4f}",
        )

        moved = []
        for name, param in model.named_parameters():
            if name in model.initial:
                same = torch.allclose(param.detach().cpu(),
                                      model.initial[name].cpu(), atol=1e-7)
                moved.append((name, not same))
        check(
            "2. encoder weights actually move",
            moved and all(changed for _, changed in moved),
            ", ".join(f"{n.split('.')[-3] if n.count('.') > 2 else n}"
                      f"{' changed' if c else ' UNCHANGED'}" for n, c in moved),
        )

        train_modes = [m for m in model.forward_modes if m[0]]
        eval_modes = [m for m in model.forward_modes if not m[0]]
        check(
            "3. validation runs in eval mode with grad off",
            train_modes and eval_modes
            and all(grad for _, grad in train_modes)
            and not any(grad for _, grad in eval_modes),
            f"{len(train_modes)} train forwards, {len(eval_modes)} eval forwards",
        )

        saved = {}
        with open(result.val_logits_path) as fh:
            for line in fh:
                row = json.loads(line)
                saved[row["id"]] = (row["logit_clean"], row["logit_hallucinated"])

        fresh = HallucinationClassifier(DEFAULT_MODEL).to(torch.device(device))
        fresh.load_state_dict(torch.load(result.checkpoint_path,
                                         map_location=torch.device(device),
                                         weights_only=True))
        _, val_loader = _loader(val_records, tokenizer, config, shuffle=False)
        logits, _, ids, _ = evaluate(fresh, val_loader, torch.device(device), config)

        worst = 0.0
        for row_id, row in zip(ids, logits.tolist()):
            want = saved[row_id]
            worst = max(worst, abs(row[0] - want[0]), abs(row[1] - want[1]))
        check(
            "4. reloaded checkpoint reproduces the saved logits",
            len(saved) == len(ids) and worst < 1e-4,
            f"largest logit difference {worst:.2e} over {len(ids)} examples",
        )

        accumulation = EFFECTIVE_BATCH // PHYSICAL_BATCH
        batches = math.ceil(TRAIN_N / PHYSICAL_BATCH)
        per_epoch = math.ceil(batches / accumulation)
        expected = per_epoch * result.epochs_run
        check(
            "5. optimizer step count matches the hand computation",
            result.optimizer_steps == expected,
            f"{result.optimizer_steps} steps, expected {per_epoch} per epoch times "
            f"{result.epochs_run} epochs = {expected}",
        )

        check(
            "6. no test logits on a dev run",
            result.test_logits_path is None
            and result.test_scores is None
            and not (tmp / "test_logits.jsonl").exists(),
            "test logits file was not written",
        )

        print(f"\nbest epoch {result.best_epoch}, best val macro f1 "
              f"{result.best_val_macro_f1:.4f}")

    failed = [name for name, ok in RESULTS if not ok]
    print()
    if failed:
        print(f"{len(failed)} of {len(RESULTS)} checks FAILED:")
        for name in failed:
            print(f"  {name}")
        return 1
    print(f"all {len(RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
