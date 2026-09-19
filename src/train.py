"""Training loop for the hallucination detector, one condition and one seed per call.

Importable with no side effects. Predictions and F1 come from src.metrics, never from a
second definition in here.
"""

import argparse
import contextlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import MAX_LENGTH
from src.data.dataset import RagTruthDataset, make_collate_fn
from src.metrics import predict_from_logits, scores
from src.model import DEFAULT_MODEL, HallucinationClassifier


@dataclass
class TrainConfig:
    condition: str
    seed: int
    learning_rate: float
    checkpoint_path: Path
    val_logits_path: Path
    test_logits_path: Path | None = None
    model_name: str = DEFAULT_MODEL
    effective_batch: int = 16
    physical_batch: int = 2
    max_length: int = MAX_LENGTH
    max_epochs: int = 6
    patience: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    device: str = "cpu"
    precision: str = "fp32"
    is_final_run: bool = False
    num_workers: int = 0


@dataclass
class TrainResult:
    condition: str
    seed: int
    max_length: int
    best_epoch: int
    best_val_macro_f1: float
    best_val_scores: dict
    epochs_run: int
    optimizer_steps: int
    planned_optimizer_steps: int
    checkpoint_path: str
    val_logits_path: str
    test_logits_path: str | None = None
    test_scores: dict | None = None
    history: list = field(default_factory=list)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_paired_init(model_name=DEFAULT_MODEL, seed=42):
    """Build two models at one seed and assert they start identical. Run once by hand."""
    seed_everything(seed)
    first = HallucinationClassifier(model_name).state_dict()
    seed_everything(seed)
    second = HallucinationClassifier(model_name).state_dict()
    assert first.keys() == second.keys(), "the two models have different parameter names"
    # Only holds on the same hardware and the same precision. An MPS fp32 init and a
    # cluster bf16 init will not match bit for bit, so pair A and B within a machine.
    differing = [k for k in first if not torch.equal(first[k], second[k])]
    assert not differing, f"same seed gave different weights for {differing[:5]}"


def build_param_groups(model, weight_decay):
    # Read the real parameter names instead of assuming the string "LayerNorm".
    # ModernBERT names its norms things like "norm.weight" and "final_norm.weight", so a
    # hardcoded name would quietly apply weight decay to every one of them.
    decay, no_decay, no_decay_names = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        if lowered.endswith(".bias") or "norm" in lowered:
            no_decay.append(param)
            no_decay_names.append(name)
        else:
            decay.append(param)

    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert len(decay) + len(no_decay) == trainable, (
        f"param groups hold {len(decay)} + {len(no_decay)} tensors but the model has "
        f"{trainable} trainable ones, so something was dropped or counted twice"
    )
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return groups, no_decay_names


def build_scheduler(optimizer, total_steps, warmup_ratio):
    # Warmup is measured in optimizer steps, not batches. With accumulation those differ
    # by the accumulation factor, and using batches would warm up far too slowly.
    warmup = max(1, int(round(warmup_ratio * total_steps)))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        remaining = total_steps - warmup
        if remaining <= 0:
            return 0.0
        return max(0.0, (total_steps - step) / remaining)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _autocast(device, precision):
    # Parameters stay fp32. autocast casts only the ops it runs, so AdamW still updates
    # fp32 master weights. Casting the whole model to bf16 would throw that away.
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _free_memory(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def accumulation_windows(loader, accumulation_steps):
    window = []
    for batch in loader:
        window.append(batch)
        if len(window) == accumulation_steps:
            yield window
            window = []
    if window:
        yield window


def train_one_epoch(model, loader, optimizer, scheduler, device, config,
                    accumulation_steps):
    model.train()
    total_loss, seen, steps = 0.0, 0, 0
    for window in accumulation_windows(loader, accumulation_steps):
        window_n = sum(int(b["labels"].shape[0]) for b in window)
        optimizer.zero_grad(set_to_none=True)
        for batch in window:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with _autocast(device, config.precision):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss_sum = F.cross_entropy(logits, labels, reduction="sum")
            # Backprop the sum divided by the window's real example count. Dividing every
            # microbatch by a fixed accumulation count, or leaving cross_entropy on its
            # default mean, would misweight the short final window of an epoch.
            (loss_sum / window_n).backward()
            total_loss += float(loss_sum.detach())
            seen += int(labels.shape[0])
            del logits, loss_sum, input_ids, attention_mask, labels
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        steps += 1
    return total_loss / max(seen, 1), steps


@torch.no_grad()
def evaluate(model, loader, device, config):
    model.eval()
    chunks, label_chunks, ids = [], [], []
    total_loss, seen = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with _autocast(device, config.precision):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss_sum = F.cross_entropy(logits, labels, reduction="sum")
        total_loss += float(loss_sum.detach())
        seen += int(labels.shape[0])
        # Straight to cpu fp32 so no device tensors pile up across the eval set.
        chunks.append(logits.detach().float().cpu())
        label_chunks.append(labels.detach().cpu())
        ids.extend(batch["ids"])
        del logits, loss_sum, input_ids, attention_mask, labels
    return torch.cat(chunks), torch.cat(label_chunks), ids, total_loss / max(seen, 1)


def save_logits(path, ids, logits, labels):
    """One json object per line: id, the two raw logits, and the true label."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = logits.tolist()
    truth = labels.tolist()
    with path.open("w") as fh:
        for rid, (clean, hallucinated), label in zip(ids, rows, truth):
            fh.write(json.dumps({
                "id": rid,
                "logit_clean": clean,
                "logit_hallucinated": hallucinated,
                "label": int(label),
            }) + "\n")


def _loader(records, tokenizer, config, shuffle):
    dataset = RagTruthDataset(records, tokenizer, config.condition, config.max_length)
    return dataset, DataLoader(
        dataset,
        batch_size=config.physical_batch,
        shuffle=shuffle,
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
        num_workers=config.num_workers,
    )


def run_training(config, train_records, val_records, tokenizer, test_records=None):
    # The test guard is structural, not a convention. A dev grid run passes no test
    # records at all, so there is no path by which test can reach model selection.
    if test_records is not None and not config.is_final_run:
        raise ValueError(
            "test_records was passed with is_final_run False. Test is only read on a "
            "final run, so this is refused rather than quietly evaluated."
        )
    if config.is_final_run and test_records is None:
        raise ValueError("is_final_run is True but no test_records were given")
    if config.effective_batch % config.physical_batch:
        raise ValueError(
            f"effective_batch {config.effective_batch} is not a multiple of "
            f"physical_batch {config.physical_batch}"
        )

    seed_everything(config.seed)
    device = torch.device(config.device)
    model = HallucinationClassifier(config.model_name).to(device)

    train_set, train_loader = _loader(train_records, tokenizer, config, shuffle=True)
    _, val_loader = _loader(val_records, tokenizer, config, shuffle=False)

    accumulation_steps = config.effective_batch // config.physical_batch
    batches_per_epoch = math.ceil(len(train_set) / config.physical_batch)
    steps_per_epoch = math.ceil(batches_per_epoch / accumulation_steps)
    planned_steps = steps_per_epoch * config.max_epochs

    groups, _ = build_param_groups(model, config.weight_decay)
    optimizer = torch.optim.AdamW(groups, lr=config.learning_rate)
    scheduler = build_scheduler(optimizer, planned_steps, config.warmup_ratio)

    best_macro = -1.0
    best_epoch = 0
    best_scores = None
    since_improved = 0
    optimizer_steps = 0
    history = []
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.max_epochs + 1):
        train_loss, steps = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, config, accumulation_steps
        )
        optimizer_steps += steps

        logits, labels, _, val_loss = evaluate(model, val_loader, device, config)
        val_scores = scores(predict_from_logits(logits), labels)
        del logits, labels

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_scores,
        })
        print(f"epoch {epoch} train_loss {train_loss:.4f} val_loss {val_loss:.4f} "
              f"val_macro_f1 {val_scores['macro_f1']:.4f} "
              f"val_positive_f1 {val_scores['positive_f1']:.4f}")

        if val_scores["macro_f1"] > best_macro:
            best_macro = val_scores["macro_f1"]
            best_epoch = epoch
            best_scores = val_scores
            torch.save(model.state_dict(), checkpoint_path)
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= config.patience:
                print(f"no macro f1 improvement for {config.patience} epochs, stopping")
                break
        _free_memory(device)

    epochs_run = history[-1]["epoch"]
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    _free_memory(device)

    logits, labels, ids, _ = evaluate(model, val_loader, device, config)
    save_logits(config.val_logits_path, ids, logits, labels)
    del logits, labels
    _free_memory(device)

    test_scores = None
    test_path = None
    if config.is_final_run:
        _, test_loader = _loader(test_records, tokenizer, config, shuffle=False)
        logits, labels, ids, _ = evaluate(model, test_loader, device, config)
        test_scores = scores(predict_from_logits(logits), labels)
        test_path = str(config.test_logits_path)
        save_logits(config.test_logits_path, ids, logits, labels)
        del logits, labels
        _free_memory(device)

    return TrainResult(
        condition=config.condition,
        seed=config.seed,
        max_length=config.max_length,
        best_epoch=best_epoch,
        best_val_macro_f1=best_macro,
        best_val_scores=best_scores,
        epochs_run=epochs_run,
        optimizer_steps=optimizer_steps,
        planned_optimizer_steps=planned_steps,
        checkpoint_path=str(checkpoint_path),
        val_logits_path=str(config.val_logits_path),
        test_logits_path=test_path,
        test_scores=test_scores,
        history=history,
    )


def _resolve_device(name, precision):
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if precision == "auto":
        precision = "bf16" if name == "cuda" else "fp32"
    return name, precision


def _stratified_head(records, limit):
    # Round robin over (task family, label) so a capped smoke run still has both classes
    # and all three families in it.
    if limit is None or limit >= len(records):
        return records
    buckets = {}
    for r in records:
        buckets.setdefault((r["task_type"], r["label"]), []).append(r)
    keys = sorted(buckets)
    picked = []
    i = 0
    while len(picked) < limit and any(buckets[k] for k in keys):
        bucket = buckets[keys[i % len(keys)]]
        if bucket:
            picked.append(bucket.pop(0))
        i += 1
    return picked


def main():
    parser = argparse.ArgumentParser(description="one run: one condition, one seed")
    parser.add_argument("--model-dir", required=True,
                        help="local ModernBERT-base folder, not a hub name")
    parser.add_argument("--data-dir", required=True,
                        help="folder holding response.jsonl and source_info.jsonl")
    parser.add_argument("--condition", required=True, choices=["A", "B"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap records per fold, smoke tests only")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--precision", default="auto",
                        choices=["auto", "fp32", "bf16"])
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from src.data.labels import label_records
    from src.data.load import load_joined
    from src.data.splits import assert_disjoint, assign_split, load_val_split

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"no config.json under {model_dir}, expected a local model folder")

    split_file = Path(__file__).resolve().parents[1] / "splits" / "val_source_ids.json"
    if not split_file.exists():
        raise SystemExit(
            f"frozen split missing at {split_file}. Refusing to draw a new one, since a "
            f"redraw would train on a different split than the one already validated."
        )

    device, precision = _resolve_device(args.device, args.precision)
    seed_everything(args.seed)

    records = label_records(load_joined(args.data_dir).to_dict("records"))
    val_ids = load_val_split(split_file)
    assigned = assign_split(records, val_ids)
    assert_disjoint(records, val_ids)

    folds = {"train": [], "val": [], "test": []}
    for r in assigned:
        folds[r["fold"]].append(r)
    for fold, fold_records in folds.items():
        folds[fold] = _stratified_head(fold_records, args.limit)

    out = Path(args.out).expanduser().resolve()
    stem = f"{args.condition}_seed{args.seed}"
    config = TrainConfig(
        condition=args.condition,
        seed=args.seed,
        learning_rate=args.lr,
        checkpoint_path=out / f"{stem}_best.pt",
        val_logits_path=out / f"{stem}_val_logits.jsonl",
        test_logits_path=out / f"{stem}_test_logits.jsonl" if args.final else None,
        model_name=str(model_dir),
        device=device,
        precision=precision,
        is_final_run=args.final,
    )

    print(f"condition {args.condition} seed {args.seed} lr {args.lr} device {device} "
          f"precision {precision} max_length {config.max_length}")
    print(f"train {len(folds['train']):,} val {len(folds['val']):,} "
          f"test {len(folds['test']):,}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    result = run_training(
        config,
        folds["train"],
        folds["val"],
        tokenizer,
        test_records=folds["test"] if args.final else None,
    )

    print(f"done condition {result.condition} seed {result.seed} "
          f"best epoch {result.best_epoch} val macro f1 {result.best_val_macro_f1:.4f} "
          f"steps {result.optimizer_steps} checkpoint {result.checkpoint_path} "
          f"val logits {result.val_logits_path}")

    # Last write of the run, so its presence means everything above finished.
    with (out / f"{stem}_result.json").open("w") as fh:
        json.dump({
            "condition": result.condition,
            "seed": result.seed,
            "learning_rate": config.learning_rate,
            "max_length": config.max_length,
            "best_epoch": result.best_epoch,
            "best_val_macro_f1": result.best_val_macro_f1,
            "best_val_scores": result.best_val_scores,
            "epochs_run": result.epochs_run,
            "optimizer_steps": result.optimizer_steps,
            "checkpoint_path": result.checkpoint_path,
            "val_logits_path": result.val_logits_path,
        }, fh, indent=2)


if __name__ == "__main__":
    main()
