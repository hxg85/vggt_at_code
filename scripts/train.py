#!/usr/bin/env python3
"""Train VGGT-AL with the multi-task objective from the paper.

Example
-------
python scripts/train.py --config configs/default.json \
    --manifest manifests/dtu_test_6v.json --out checkpoints/dtu_al \
    --max-steps 20            # smoke run
Full paper schedule: 200 epochs, AdamW lr 1e-4, wd 1e-2, cosine decay,
input 512x384, batch 32 on 8 A100 GPUs (config defaults use batch 8 so the
code also runs on a single consumer GPU).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vggt_al.data import SparseViewSceneDataset  # noqa: E402
from vggt_al.losses import LossWeights, VGGTALLoss  # noqa: E402
from vggt_al.model import VGGTAL, VGGTALConfig  # noqa: E402
from vggt_al.utils import (  # noqa: E402
    AverageMeter,
    collate_sparse_view,
    load_checkpoint,
    load_config,
    save_checkpoint,
    set_seed,
)


def build_scheduler(optimizer, epochs: int, warmup: int):
    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if k in ("name", "sequence"):
            continue
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-from-vggt", default=None,
                        help="optional .pt checkpoint with standard VGGT attention keys")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))
    train_cfg = cfg["train"]
    data_cfg = dict(cfg["data"])
    if args.manifest:
        data_cfg["manifest"] = args.manifest
    if not data_cfg.get("manifest"):
        raise SystemExit("provide --manifest or set data.manifest in the config")
    out_dir = Path(args.out or train_cfg.get("output_dir", "checkpoints/vggt_al"))
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = args.device or train_cfg.get("device", "auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    model_cfg = VGGTALConfig(**cfg["model"])
    model = VGGTAL(model_cfg).to(device)
    print(f"[model] {model.num_parameters / 1e6:.2f}M parameters, "
          f"fusion mode={model_cfg.fusion_mode}, device={device}")

    if args.init_from_vggt or train_cfg.get("init_from_vggt"):
        path = args.init_from_vggt or train_cfg["init_from_vggt"]
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("model", state) if isinstance(state, dict) else state
        model.init_from_vggt(state)

    dataset = SparseViewSceneDataset(
        manifest=data_cfg["manifest"],
        img_size=tuple(model_cfg.img_size),
        num_views=data_cfg.get("num_views", model_cfg.num_views),
        augment=data_cfg.get("augment", True),
        load_depth=data_cfg.get("load_depth", True),
        load_pointcloud=data_cfg.get("load_pointcloud", False),
        view_sampling=data_cfg.get("view_sampling", "random"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 2)),
        collate_fn=collate_sparse_view,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    loss_cfg = dict(cfg["loss"])
    criterion = VGGTALLoss(
        LossWeights(**{k: v for k, v in loss_cfg.items() if k != "use_lpips"}),
        use_lpips=bool(loss_cfg.get("use_lpips", False)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )
    epochs = int(train_cfg.get("epochs", 200))
    scheduler = build_scheduler(optimizer, epochs, int(train_cfg.get("warmup_epochs", 5)))
    amp_enabled = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # older PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 0
    resume = args.resume or train_cfg.get("resume")
    if resume:
        payload = load_checkpoint(resume, model, map_location=str(device))
        start_epoch = int(payload.get("epoch", 0)) + 1
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        print(f"[resume] from {resume}, starting at epoch {start_epoch}")

    step = 0
    max_steps = int(train_cfg.get("max_steps", -1))
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        meters = AverageMeter()
        for batch in loader:
            batch = to_device(batch, device)
            images = batch["images"]
            intrinsics = batch["intrinsics"]
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if scaler.is_enabled()
                else nullcontext()
            )
            with amp_ctx:
                pred = model(images, intrinsics, return_gates=True)
                losses = criterion(pred, batch)
            loss = losses["total"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}, step {step}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if float(train_cfg.get("grad_clip", 0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_cfg["grad_clip"])
                )
            scaler.step(optimizer)
            scaler.update()

            meters.update(loss.item(), images.shape[0])
            step += 1
            if step % int(train_cfg.get("log_every", 20)) == 0:
                detail = " ".join(
                    f"{k}={losses[k].item():.4f}"
                    for k in ("depth_l1", "photometric", "camera", "fusion_smooth")
                    if k in losses
                )
                print(f"epoch {epoch:03d} step {step:05d} loss={meters.avg:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} {detail}")
            if max_steps > 0 and step >= max_steps:
                break
        scheduler.step()
        if (epoch + 1) % int(train_cfg.get("save_every", 10)) == 0 or epoch == epochs - 1:
            ckpt = out_dir / f"epoch_{epoch + 1:03d}.pt"
            save_checkpoint(ckpt, model, optimizer, epoch)
            print(f"[save] {ckpt}")
        if max_steps > 0 and step >= max_steps:
            break
    print(f"[done] {step} steps in {time.time() - t0:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
