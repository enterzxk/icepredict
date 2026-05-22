"""Train private-first binary ice/snow classifiers with fixed batch mixing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path, PureWindowsPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BINARY_LABEL = "\u51b0\u96ea\u5f02\u5e38"


def filename_from_any_path(image_path: str) -> str:
    return PureWindowsPath(str(image_path)).name or Path(str(image_path)).name


def build_filename_index(roots: Iterable[Path]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                index.setdefault(path.name, str(path))
    return index


def resolve_image_path(image_path: str, data_dir: Path, filename_index: Dict[str, str]) -> Optional[str]:
    candidates = [Path(image_path)]
    if not Path(image_path).is_absolute():
        candidates.append(data_dir / image_path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return filename_index.get(filename_from_any_path(image_path))


def load_items(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"label file must be a list json: {path}")
    return data


def count_items(items: Sequence[Dict]) -> Dict[str, int]:
    positive = sum(int(bool(item.get("label_vector", [0])[0])) for item in items)
    return {
        "count": len(items),
        "positive": positive,
        "negative": len(items) - positive,
    }


def parse_folds(value: str, cv_dir: Path) -> List[int]:
    if value.lower() == "all":
        folds = []
        for path in sorted(cv_dir.glob("fold_*")):
            if path.is_dir():
                try:
                    folds.append(int(path.name.split("_", 1)[1]))
                except (IndexError, ValueError):
                    continue
        if not folds:
            raise ValueError(f"no fold_* directories found in {cv_dir}")
        return folds
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def make_search_roots(data_dir: Path) -> List[Path]:
    return [
        data_dir,
        data_dir / "data" / "imagine",
        data_dir / "data" / "roboflow_train",
        Path("data/imagine"),
        Path("data/roboflow_train"),
    ]


def load_fold_groups(cv_dir: Path, fold: int, hard_negative_repeat: int = 5) -> Dict[str, List[Dict]]:
    fold_dir = cv_dir / f"fold_{fold}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"fold directory not found: {fold_dir}")

    private_train = load_items(fold_dir / "train_private_labels.json")
    val_items = load_items(fold_dir / "val_labels.json")
    roboflow_pos = load_items(fold_dir / "train_roboflow_pos_labels.json")
    roboflow_neg = load_items(fold_dir / "train_roboflow_neg_labels.json")
    hard_negative = load_items(fold_dir / "train_hard_negative_labels.json")

    private_pos = [item for item in private_train if int(bool(item.get("label_vector", [0])[0])) == 1]
    private_neg = [item for item in private_train if int(bool(item.get("label_vector", [0])[0])) == 0]
    aux_neg = list(roboflow_neg) + list(hard_negative) * max(0, hard_negative_repeat)

    return {
        "private_pos": private_pos,
        "roboflow_pos": roboflow_pos,
        "private_neg": private_neg,
        "aux_neg": aux_neg,
        "val": val_items,
        "hard_negative": hard_negative,
    }


def make_batch_plan(args, group_lengths: Dict[str, int]) -> Dict[str, int]:
    plan = {
        "private_pos": args.private_pos_per_batch,
        "roboflow_pos": args.roboflow_pos_per_batch,
        "private_neg": args.private_neg_per_batch,
        "aux_neg": args.aux_neg_per_batch,
    }
    if sum(plan.values()) != args.batch_size:
        raise ValueError(f"batch mix sums to {sum(plan.values())}, expected batch_size={args.batch_size}")

    if group_lengths.get("roboflow_pos", 0) == 0:
        plan["private_pos"] += plan["roboflow_pos"]
        plan["roboflow_pos"] = 0
    if group_lengths.get("aux_neg", 0) == 0:
        plan["private_neg"] += plan["aux_neg"]
        plan["aux_neg"] = 0
    if group_lengths.get("private_pos", 0) == 0 and plan["private_pos"] > 0:
        raise ValueError("private positive group is empty")
    if group_lengths.get("private_neg", 0) == 0 and plan["private_neg"] > 0:
        if group_lengths.get("aux_neg", 0) == 0:
            raise ValueError("negative groups are empty")
        plan["aux_neg"] += plan["private_neg"]
        plan["private_neg"] = 0
    return plan


def estimate_steps_per_epoch(args, group_lengths: Dict[str, int], batch_plan: Dict[str, int]) -> int:
    candidates = [args.min_steps_per_epoch]
    if batch_plan.get("private_pos", 0) > 0:
        candidates.append(math.ceil(
            group_lengths["private_pos"] * args.private_pos_repeat / batch_plan["private_pos"]
        ))
    if batch_plan.get("private_neg", 0) > 0:
        candidates.append(math.ceil(group_lengths["private_neg"] / batch_plan["private_neg"]))
    return max(1, max(candidates))


def build_transforms(image_size: int):
    from torchvision import transforms as T

    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    private_pos_train = T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        T.RandomRotation(10),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        normalize,
    ])
    standard_train = T.Compose([
        T.Resize((image_size + 16, image_size + 16)),
        T.RandomCrop(image_size),
        T.RandomRotation(5),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08, hue=0.01),
        T.ToTensor(),
        normalize,
    ])
    val = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        normalize,
    ])
    return private_pos_train, standard_train, val


class JsonImageDataset:
    def __init__(
        self,
        items: Sequence[Dict],
        data_dir: Path,
        filename_index: Dict[str, str],
        transform,
        name: str,
    ):
        self.transform = transform
        self.name = name
        self.data: List[Dict] = []
        missing: List[str] = []
        for item in items:
            resolved = resolve_image_path(item.get("image_path", ""), data_dir, filename_index)
            if not resolved:
                if len(missing) < 5:
                    missing.append(item.get("image_path", ""))
                continue
            copied = dict(item)
            copied["resolved_path"] = resolved
            self.data.append(copied)
        if missing:
            print(f"[{name}] skipped missing images: {len(missing)} examples={missing}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        import torch
        from PIL import Image

        item = self.data[idx]
        image = Image.open(item["resolved_path"]).convert("RGB")
        image = self.transform(image)
        target = float(int(bool(item.get("label_vector", [0])[0])))
        label = torch.tensor([target], dtype=torch.float32)
        return image, label, item


def sample_from_dataset(dataset: JsonImageDataset, count: int, rng: random.Random):
    if count <= 0:
        return [], [], []
    if len(dataset) == 0:
        raise ValueError(f"dataset {dataset.name} is empty but requested {count} samples")
    images = []
    labels = []
    metas = []
    for _ in range(count):
        image, label, meta = dataset[rng.randrange(len(dataset))]
        images.append(image)
        labels.append(label)
        metas.append(meta)
    return images, labels, metas


def sample_mixed_batch(datasets: Dict[str, JsonImageDataset], batch_plan: Dict[str, int], rng: random.Random):
    import torch

    images = []
    labels = []
    for name, count in batch_plan.items():
        sampled_images, sampled_labels, _ = sample_from_dataset(datasets[name], count, rng)
        images.extend(sampled_images)
        labels.extend(sampled_labels)

    order = list(range(len(images)))
    rng.shuffle(order)
    batch_images = torch.stack([images[idx] for idx in order])
    batch_labels = torch.stack([labels[idx] for idx in order])
    return batch_images, batch_labels


def load_model_from_checkpoint(args, device: str):
    import torch
    from src.ice_classifier import create_model

    model = create_model(
        model_name=args.model_name,
        num_classes=1,
        pretrained=False,
        dropout=args.dropout,
        freeze_backbone=True,
    )
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=False)
    model.to(device)
    return model


def set_trainable(model, train_layer4: bool = False) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    if train_layer4 and hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True


def train_one_epoch(
    model,
    datasets: Dict[str, JsonImageDataset],
    batch_plan: Dict[str, int],
    optimizer,
    criterion,
    device: str,
    steps_per_epoch: int,
    rng: random.Random,
    freeze_backbone: bool,
) -> float:
    model.train()
    if freeze_backbone and hasattr(model, "backbone"):
        model.backbone.eval()

    total_loss = 0.0
    for _ in range(steps_per_epoch):
        images, labels = sample_mixed_batch(datasets, batch_plan, rng)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item())

    return total_loss / max(1, steps_per_epoch)


def predict_dataset(model, dataset: JsonImageDataset, batch_size: int, device: str, fold: int) -> List[Dict]:
    import torch

    model.eval()
    rows: List[Dict] = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = [dataset[idx] for idx in range(start, min(start + batch_size, len(dataset)))]
            if not batch:
                continue
            images = torch.stack([item[0] for item in batch]).to(device)
            probabilities = torch.sigmoid(model(images)).detach().cpu().numpy().reshape(-1)
            for (_, label, meta), probability in zip(batch, probabilities):
                rows.append({
                    "fold": fold,
                    "image_path": meta.get("image_path", ""),
                    "resolved_path": meta.get("resolved_path", ""),
                    "true_label": int(float(label.item())),
                    "pred_prob": f"{float(probability):.8f}",
                    "source": meta.get("source", ""),
                })
    return rows


def rows_to_targets(rows: Sequence[Dict]) -> Tuple[List[int], List[float]]:
    targets = [int(row["true_label"]) for row in rows]
    probabilities = [float(row["pred_prob"]) for row in rows]
    return targets, probabilities


def save_predictions(path: Path, rows: Sequence[Dict], threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fold",
        "image_path",
        "resolved_path",
        "true_label",
        "pred_prob",
        "threshold",
        "pred_label",
        "error_type",
        "source",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            target = int(row["true_label"])
            probability = float(row["pred_prob"])
            pred = int(probability >= threshold)
            if target == 1 and pred == 1:
                error_type = "TP"
            elif target == 0 and pred == 1:
                error_type = "FP"
            elif target == 1 and pred == 0:
                error_type = "FN"
            else:
                error_type = "TN"
            output = dict(row)
            output["threshold"] = f"{threshold:.6f}"
            output["pred_label"] = pred
            output["error_type"] = error_type
            writer.writerow(output)


def save_checkpoint(path: Path, model, optimizer, metrics: Dict, args, fold: int) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics,
        "metadata": {
            "task": "binary",
            "label_names": [BINARY_LABEL],
            "num_classes": 1,
            "fold": fold,
            "config": vars(args),
        },
    }, path)


def train_fold(args, fold: int) -> Dict:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from tools.summarize_private_cv import make_thresholds, select_threshold

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    cv_dir = Path(args.cv_dir)
    output_dir = Path(args.output_dir) / f"fold_{fold}"
    weights_dir = Path(args.weights_dir) / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    groups = load_fold_groups(cv_dir, fold, args.hard_negative_repeat)
    raw_counts = {name: count_items(items) for name, items in groups.items()}

    filename_index = build_filename_index(make_search_roots(data_dir))
    private_pos_transform, standard_transform, val_transform = build_transforms(args.image_size)
    datasets = {
        "private_pos": JsonImageDataset(groups["private_pos"], data_dir, filename_index, private_pos_transform, "private_pos"),
        "roboflow_pos": JsonImageDataset(groups["roboflow_pos"], data_dir, filename_index, standard_transform, "roboflow_pos"),
        "private_neg": JsonImageDataset(groups["private_neg"], data_dir, filename_index, standard_transform, "private_neg"),
        "aux_neg": JsonImageDataset(groups["aux_neg"], data_dir, filename_index, standard_transform, "aux_neg"),
    }
    val_dataset = JsonImageDataset(groups["val"], data_dir, filename_index, val_transform, "val_private")

    group_lengths = {name: len(dataset) for name, dataset in datasets.items()}
    if len(val_dataset) == 0:
        raise ValueError(f"fold_{fold} validation dataset is empty")
    batch_plan = make_batch_plan(args, group_lengths)
    steps_per_epoch = estimate_steps_per_epoch(args, group_lengths, batch_plan)

    print(f"\n=== fold_{fold} ===")
    print(f"device: {device}")
    print(f"raw_counts: {raw_counts}")
    print(f"resolved_train_counts: {group_lengths}, val={len(val_dataset)}")
    print(f"batch_plan: {batch_plan}, steps_per_epoch={steps_per_epoch}")
    print("pos_weight=False, weighted_sampling=False, mixup=False, cutmix=False, vertical_flip=False")

    model = load_model_from_checkpoint(args, device)
    criterion = nn.BCEWithLogitsLoss()
    rng = random.Random(args.seed + fold)
    thresholds = make_thresholds(args.threshold_start, args.threshold_end, args.threshold_step)
    best_score: Optional[Tuple] = None
    best_metrics: Optional[Dict] = None

    def run_stage(stage_name: str, epochs: int, lr: float, train_layer4: bool) -> None:
        nonlocal best_score, best_metrics
        if epochs <= 0:
            return
        set_trainable(model, train_layer4=train_layer4)
        optimizer = optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr,
            weight_decay=args.weight_decay,
        )
        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(
                model=model,
                datasets=datasets,
                batch_plan=batch_plan,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                steps_per_epoch=steps_per_epoch,
                rng=rng,
                freeze_backbone=not train_layer4,
            )
            rows = predict_dataset(model, val_dataset, args.eval_batch_size, device, fold)
            targets, probabilities = rows_to_targets(rows)
            metrics = select_threshold(targets, probabilities, args.max_fpr, thresholds)
            metrics.update({
                "stage": stage_name,
                "epoch": epoch,
                "train_loss": loss,
            })
            score = (
                1 if metrics["meets_fpr_constraint"] else 0,
                metrics["recall"],
                metrics["f0_5"],
                -metrics["fpr"],
                metrics["precision"],
            )
            print(
                f"{stage_name} epoch {epoch}/{epochs}: "
                f"loss={loss:.4f}, threshold={metrics['threshold']:.3f}, "
                f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                f"FPR={metrics['fpr']:.3f}, TP={metrics['tp']}, FP={metrics['fp']}, "
                f"FN={metrics['fn']}, TN={metrics['tn']}"
            )
            if best_score is None or score > best_score:
                best_score = score
                best_metrics = dict(metrics)
                save_checkpoint(weights_dir / "best.pth", model, optimizer, best_metrics, args, fold)

    run_stage("stage1_classifier", args.epochs_stage1, args.lr_stage1, train_layer4=False)
    run_stage("stage2_layer4", args.epochs_stage2, args.lr_stage2, train_layer4=True)

    if best_metrics is None:
        raise RuntimeError(f"fold_{fold} did not produce a checkpoint")

    checkpoint = torch.load(weights_dir / "best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    rows = predict_dataset(model, val_dataset, args.eval_batch_size, device, fold)
    save_predictions(output_dir / "predictions.csv", rows, best_metrics["threshold"])

    fold_summary = {
        "fold": fold,
        "best_metrics": best_metrics,
        "batch_plan": batch_plan,
        "steps_per_epoch": steps_per_epoch,
        "raw_counts": raw_counts,
        "resolved_train_counts": group_lengths,
        "resolved_val_count": len(val_dataset),
        "checkpoint": str(weights_dir / "best.pth"),
        "predictions": str(output_dir / "predictions.csv"),
    }
    with open(output_dir / "fold_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, ensure_ascii=False, indent=2)
    return fold_summary


def dry_run(args) -> None:
    cv_dir = Path(args.cv_dir)
    folds = parse_folds(args.folds, cv_dir)
    print("=== private CV train dry-run ===")
    for fold in folds:
        groups = load_fold_groups(cv_dir, fold, args.hard_negative_repeat)
        counts = {name: count_items(items) for name, items in groups.items()}
        lengths = {name: len(items) for name, items in groups.items()}
        batch_plan = make_batch_plan(args, {
            "private_pos": lengths["private_pos"],
            "roboflow_pos": lengths["roboflow_pos"],
            "private_neg": lengths["private_neg"],
            "aux_neg": lengths["aux_neg"],
        })
        steps = estimate_steps_per_epoch(args, {
            "private_pos": lengths["private_pos"],
            "private_neg": lengths["private_neg"],
        }, batch_plan)
        print(f"fold_{fold}: counts={counts}, batch_plan={batch_plan}, estimated_steps={steps}")


def train_private_cv(args) -> Dict:
    folds = parse_folds(args.folds, Path(args.cv_dir))
    if args.dry_run:
        dry_run(args)
        return {"dry_run": True, "folds": folds}

    fold_summaries = [train_fold(args, fold) for fold in folds]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "folds": fold_summaries,
        "config": vars(args),
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train private-first binary CV models")
    parser.add_argument("--cv-dir", default="data/private_cv")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--init-checkpoint", default="weights/ice_binary_classifier/best_stage2.pth")
    parser.add_argument("--weights-dir", default="weights/ice_binary_private_cv")
    parser.add_argument("--output-dir", default="experiments/private_cv")
    parser.add_argument("--folds", default="all", help="'all' or comma-separated fold ids, e.g. 0,1")
    parser.add_argument("--model-name", default="resnet50")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--private-pos-per-batch", type=int, default=4)
    parser.add_argument("--roboflow-pos-per-batch", type=int, default=4)
    parser.add_argument("--private-neg-per-batch", type=int, default=16)
    parser.add_argument("--aux-neg-per-batch", type=int, default=8)
    parser.add_argument("--private-pos-repeat", type=int, default=25)
    parser.add_argument("--hard-negative-repeat", type=int, default=5)
    parser.add_argument("--min-steps-per-epoch", type=int, default=40)
    parser.add_argument("--epochs-stage1", type=int, default=6)
    parser.add_argument("--epochs-stage2", type=int, default=3)
    parser.add_argument("--lr-stage1", type=float, default=3e-5)
    parser.add_argument("--lr-stage2", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument("--threshold-start", type=float, default=0.01)
    parser.add_argument("--threshold-end", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    train_private_cv(args)


if __name__ == "__main__":
    main()
