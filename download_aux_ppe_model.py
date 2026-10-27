"""
Fetch the auxiliary PPE model (ear protection / goggles / gloves / boots) and rebuild it safely.

Source: https://huggingface.co/killuminati1/construction-ppe-yolov8 (Apache-2.0, YOLOv8, 19 classes:
Boots, Ear-Protection, Glass, Glove, Hard_hat, Mask, No-* variants, Worker, Vest, site equipment).

A downloaded .pt is a pickle and ordinary loading can execute code from the file. This script never
does that: it loads with torch's weights_only unpickler (only an allow-list of torch / ultralytics /
collections classes, checked against the file's own imports first), keeps just the weight tensors,
rebuilds the network from the locally installed ultralytics package, and saves that as
model/ppe_aux.pt. The app runs without this file (helmet/vest only) if it is missing.

Usage:
    python download_aux_ppe_model.py               # download from Hugging Face
    python download_aux_ppe_model.py --from best.pt  # use an already-downloaded file
"""
import argparse
import importlib
import os
import pickletools
import tempfile
import urllib.request
import zipfile

import torch

import config

SOURCE_URL = "https://huggingface.co/killuminati1/construction-ppe-yolov8/resolve/main/best.pt"
ALLOWED_TOP_LEVEL = ("torch", "ultralytics", "collections", "__builtin__", "builtins")


def pickle_globals(path):
    """Every (module, name) the checkpoint's pickle would import, read without unpickling."""
    with zipfile.ZipFile(path) as zf:
        data = zf.read(next(n for n in zf.namelist() if n.endswith(".pkl")))
    found, strings = set(), []
    for op, arg, _ in pickletools.genops(data):
        if "UNICODE" in op.name or "STRING" in op.name:
            strings.append(arg)
        elif op.name == "STACK_GLOBAL":
            found.add((strings[-2], strings[-1]))
        elif op.name == "GLOBAL":
            found.add(tuple(arg.split(" ")))
    return found


def safe_load(path):
    allowed = []
    for module, name in sorted(pickle_globals(path)):
        if module.split(".")[0] not in ALLOWED_TOP_LEVEL:
            raise RuntimeError(f"Refusing to load: checkpoint imports {module}.{name}")
        mod = importlib.import_module("builtins" if module == "__builtin__" else module)
        allowed.append(getattr(mod, name))
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def rebuild(ckpt, out_path):
    from ultralytics.nn.tasks import DetectionModel

    src = (ckpt.get("ema") or ckpt["model"]).float()
    model = DetectionModel(cfg=src.yaml, nc=len(src.names), verbose=False)
    model.load_state_dict(src.state_dict(), strict=True)
    model.names = dict(src.names)
    model.eval()
    torch.save({"model": model, "train_args": {}, "source": SOURCE_URL}, out_path)
    return model.names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="src", help="path to an already-downloaded best.pt")
    args = parser.parse_args()

    src = args.src
    if not src:
        src = os.path.join(tempfile.mkdtemp(), "best.pt")
        print(f"Downloading {SOURCE_URL} ...")
        urllib.request.urlretrieve(SOURCE_URL, src)

    names = rebuild(safe_load(src), config.AUX_PPE_MODEL_PATH)
    print(f"Saved {config.AUX_PPE_MODEL_PATH}")
    print(f"Classes: {names}")


if __name__ == "__main__":
    main()
