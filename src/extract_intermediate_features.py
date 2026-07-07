"""
Extract INTERMEDIATE-LAYER embeddings from a frozen ViT backbone, resumably.

Motivation (instructor, 2026-07-07 + Odonnat et al., CAO@ICLR-2026, arXiv
2603.05280): intermediate BACKBONE layers are more robust than the final layer
under distribution shift, and the FFN GeLU activation ('Act') is the best
module tap under strong shift. Layer taps are candidate MDCP score-space
dimensions (views) for the false-corr screen in src/mdcp_dim_screen.py.

What is saved per image (all L2-normalized, mirroring extract_features.py):
  layer<KK>  CLS token of transformer block KK's output (KK 1-based),
             the paper's RC2 convention, e.g. layer03/layer06/layer09/layer12
  act<KK>    (optional, --act_layers) CLS token of the GeLU activation inside
             block KK's FFN (4*d-dimensional; the paper's strong-shift winner)
  final      the model's standard output embedding (post-norm pooled token,
             identical convention to the existing embeddings_*.pt files)

CRASH SAFETY / STEPWISE EXECUTION: images are processed in fixed-size chunks
in the deterministic ImageFolder order. Each completed chunk is written
atomically to <output_dir>/tmp_<output_name>/chunk_XXXXX.pt; on restart,
completed chunks are skipped, so a mid-run crash loses at most one chunk.
A meta.json guards against resuming with a changed configuration. When all
chunks exist they are consolidated atomically into <output_dir>/<output_name>
and the chunk dir is removed (keep with --keep_chunks).

Run (cluster, per dataset -- see cluster/run_intermediate_extraction.sh):
python src/extract_intermediate_features.py \
    --data_dir data/cifar100 --output_name embeddings_cifar100_layers.pt \
    --model dinov2-base --input_size 518 --batch_size 64 --layers 3 6 9 12
"""
import argparse
import json
import os
import shutil
import time

import torch
import timm
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

from extract_features import MODEL_PRESETS, get_transform


def get_args():
    p = argparse.ArgumentParser(description="Resumable intermediate-layer ViT feature extraction")
    p.add_argument("--data_dir", required=True, help="ImageFolder root")
    p.add_argument("--output_name", required=True, help="output .pt filename")
    p.add_argument("--output_dir", default="output")
    p.add_argument("--model", default="dinov2-base",
                   help=f"preset ({', '.join(MODEL_PRESETS)}) or timm name")
    p.add_argument("--input_size", type=int, default=None,
                   help="override preset input size (multiple of patch size)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--layers", type=int, nargs="+", default=[3, 6, 9, 12],
                   help="1-based transformer block indices to tap (block output CLS)")
    p.add_argument("--act_layers", type=int, nargs="*", default=[],
                   help="1-based block indices for FFN-GeLU ('Act') taps (4*d dims each)")
    p.add_argument("--chunk_size", type=int, default=2048,
                   help="images per checkpoint chunk (crash loses at most one chunk)")
    p.add_argument("--num_per_class", type=int, default=None)
    p.add_argument("--keep_chunks", action="store_true",
                   help="keep the chunk dir after consolidation")
    return p.parse_args()


def build_dataset(args, input_size):
    dataset = datasets.ImageFolder(root=args.data_dir, transform=get_transform(input_size))
    if args.num_per_class is not None:
        counts = {i: 0 for i in range(len(dataset.classes))}
        keep = []
        for idx, (_p, lab) in enumerate(dataset.samples):
            if counts[lab] < args.num_per_class:
                keep.append(idx)
                counts[lab] += 1
        dataset = Subset(dataset, keep)
    return dataset


def make_hooks(model, layers, act_layers):
    """Register forward hooks; returns (captures dict, handles list).
    Block tap = block output CLS token; act tap = FFN GeLU output CLS token."""
    captures, handles = {}, []

    def block_hook(key):
        def fn(_m, _inp, out):
            captures[key] = out[:, 0].detach()
        return fn

    n_blocks = len(model.blocks)
    for k in layers:
        if not 1 <= k <= n_blocks:
            raise ValueError(f"--layers {k} out of range 1..{n_blocks}")
        handles.append(model.blocks[k - 1].register_forward_hook(block_hook(f"layer{k:02d}")))
    for k in act_layers:
        if not 1 <= k <= n_blocks:
            raise ValueError(f"--act_layers {k} out of range 1..{n_blocks}")
        handles.append(model.blocks[k - 1].mlp.act.register_forward_hook(block_hook(f"act{k:02d}")))
    return captures, handles


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model in MODEL_PRESETS:
        model_name, input_size = MODEL_PRESETS[args.model]
    else:
        model_name, input_size = args.model, 224
    if args.input_size is not None:
        input_size = args.input_size

    dataset = build_dataset(args, input_size)
    n = len(dataset)
    n_chunks = (n + args.chunk_size - 1) // args.chunk_size

    work_dir = os.path.join(args.output_dir, f"tmp_{args.output_name}".replace(".pt", ""))
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- config guard: never mix chunks from different configurations -------
    meta = dict(model=model_name, input_size=input_size, layers=sorted(args.layers),
                act_layers=sorted(args.act_layers), chunk_size=args.chunk_size,
                n_images=n, data_dir=os.path.abspath(args.data_dir),
                num_per_class=args.num_per_class)
    meta_path = os.path.join(work_dir, "meta.json")
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        if old != meta:
            raise RuntimeError(
                f"Resume config mismatch in {work_dir}:\n  saved: {old}\n  now:   {meta}\n"
                f"Delete the chunk dir to restart from scratch.")
        print(f"Resuming into {work_dir}")
    else:
        json.dump(meta, open(meta_path, "w"), indent=2)

    done = {f for f in os.listdir(work_dir) if f.startswith("chunk_") and f.endswith(".pt")}
    todo = [c for c in range(n_chunks) if f"chunk_{c:05d}.pt" not in done]
    print(f"{n} images, {n_chunks} chunks of {args.chunk_size}; "
          f"{n_chunks - len(todo)} done, {len(todo)} to go")

    # --- extraction ---------------------------------------------------------
    if todo:
        print(f"Loading {model_name} (input {input_size}) on {device}...")
        create_kwargs = {"pretrained": True, "num_classes": 0}
        if args.input_size is not None:
            create_kwargs["img_size"] = input_size
        model = timm.create_model(model_name, **create_kwargs).to(device).eval()
        captures, handles = make_hooks(model, args.layers, args.act_layers)

        for c in todo:
            t0 = time.time()
            lo, hi = c * args.chunk_size, min((c + 1) * args.chunk_size, n)
            loader = DataLoader(Subset(dataset, range(lo, hi)),
                                batch_size=args.batch_size, shuffle=False, num_workers=2)
            taps = {f"layer{k:02d}": [] for k in args.layers}
            taps.update({f"act{k:02d}": [] for k in args.act_layers})
            taps["final"] = []
            labels = []
            with torch.no_grad():
                for images, labs in loader:
                    out = model(images.to(device))            # standard final embedding
                    captures["final"] = out.detach()
                    for key, buf in taps.items():
                        feat = torch.nn.functional.normalize(captures[key], p=2, dim=1)
                        buf.append(feat.cpu())
                    labels.append(labs)
            chunk = {"labels": torch.cat(labels),
                     "taps": {k: torch.cat(v) for k, v in taps.items()}}
            atomic_save(chunk, os.path.join(work_dir, f"chunk_{c:05d}.pt"))
            print(f"  chunk {c + 1}/{n_chunks} [{lo}:{hi}] saved ({time.time() - t0:.0f}s)")

        for h in handles:
            h.remove()

    # --- consolidation ------------------------------------------------------
    print("Consolidating...")
    tap_keys = [f"layer{k:02d}" for k in sorted(args.layers)] + \
               [f"act{k:02d}" for k in sorted(args.act_layers)] + ["final"]
    parts = {k: [] for k in tap_keys}
    labels = []
    for c in range(n_chunks):
        chunk = torch.load(os.path.join(work_dir, f"chunk_{c:05d}.pt"),
                           map_location="cpu", weights_only=False)
        labels.append(chunk["labels"])
        for k in tap_keys:
            parts[k].append(chunk["taps"][k])
    out = {"labels": torch.cat(labels), "meta": meta}
    for k in tap_keys:
        out[k] = torch.cat(parts[k])
        assert len(out[k]) == n, f"{k}: {len(out[k])} rows != {n} images"
    assert len(out["labels"]) == n

    final_path = os.path.join(args.output_dir, args.output_name)
    atomic_save(out, final_path)
    shapes = {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)}
    print(f"Saved {final_path}\n  shapes: {shapes}")

    if not args.keep_chunks:
        shutil.rmtree(work_dir)
        print(f"Removed {work_dir}")


if __name__ == "__main__":
    main()
