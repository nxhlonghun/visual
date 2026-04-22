"""
One-shot dataset builder from official COCO2017 + VisDrone2019-DET (local folders).

Expected layout (under DATASET_ROOT, default D:/task/school/dataset):
  coco2017/
    train2017/  val2017/
    annotations/instances_train2017.json  instances_val2017.json
  VisDrone2019/
    VisDrone2019-DET-train/{images,annotations}/
    VisDrone2019-DET-val/{images,annotations}/

Steps:
  1) Wipe merged train/val images+labels (does not touch coco2017/ or VisDrone2019/).
  2) COCO: person-only -> YOLO (class 0), hardlink-or-copy into images/{train,val}.
  3) VisDrone: pedestrian(1)+people(2) -> YOLO class 0, copy as vd_*.jpg into train/val.
  4) Test split: stratified sample from val (COCO + vd_ pos move to test); VisDrone val
     negatives (no person in raw ann) copied to test as vd_neg_* with empty labels.
  5) Write data.yaml (with test: if step 4 ran).

Usage:
  python build_dataset_from_sources.py
  python build_dataset_from_sources.py --skip-test
  python build_dataset_from_sources.py --test-fraction 0.15 --seed 0

Requires: Pillow (PIL) for VisDrone image sizes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Defaults (override with --dataset)
# ---------------------------------------------------------------------------
# Default to a path relative to this script so it works across machines.
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "dataset"
# Source folder names under dataset root.
# If you rename source folders, only modify these two constants.
DEFAULT_COCO_DIR = "coco2017"
DEFAULT_VISDRONE_DIR = "VisDrone2019"
# COCO internal names.
COCO_TRAIN_IMAGES_DIR = "train2017"
COCO_VAL_IMAGES_DIR = "val2017"
COCO_ANN_DIR = "annotations"
COCO_TRAIN_ANN = "instances_train2017.json"
COCO_VAL_ANN = "instances_val2017.json"
# VisDrone internal names.
VIS_TRAIN_DIR = "VisDrone2019-DET-train"
VIS_VAL_DIR = "VisDrone2019-DET-val"
VIS_IMAGES_DIR = "images"
VIS_ANN_DIR = "annotations"

PERSON_CATS = {1, 2}  # VisDrone: pedestrian, people


def _clear_merged_split_dirs(
    root: Path, splits: tuple[str, ...] = ("train", "val", "test")
) -> None:
    for sp in splits:
        for sub in ("images", "labels"):
            d = root / sub / sp
            if not d.is_dir():
                continue
            for p in d.iterdir():
                if p.is_file():
                    p.unlink()


def _remove_label_caches(root: Path) -> None:
    for name in ("train.cache", "val.cache", "test.cache"):
        c = root / "labels" / name
        if c.is_file():
            c.unlink()
            print(f"Removed {c}")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        # Prefer modern pathlib API when available (Python 3.10+).
        dst.hardlink_to(src)
        return
    except (AttributeError, OSError):
        pass
    try:
        # Python 3.9-compatible hardlink fallback.
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# --- COCO ------------------------------------------------------------------


def convert_coco_split(
    coco_root: Path,
    dst_root: Path,
    images_dir_name: str,
    ann_file_name: str,
    out_split: str,
) -> int:
    ann_path = coco_root / COCO_ANN_DIR / ann_file_name
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {im["id"]: im for im in data["images"]}
    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        if ann.get("category_id") == 1:
            anns_by_image[ann["image_id"]].append(ann)

    src_images_dir = coco_root / images_dir_name
    dst_images_dir = dst_root / "images" / out_split
    dst_labels_dir = dst_root / "labels" / out_split
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for image_id, image_info in images.items():
        file_name = image_info["file_name"]
        img_w = float(image_info["width"])
        img_h = float(image_info["height"])
        src_img = src_images_dir / file_name
        dst_img = dst_images_dir / file_name
        link_or_copy(src_img, dst_img)
        yolo_lines: list[str] = []
        for ann in anns_by_image.get(image_id, []):
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            xc = (x + w / 2.0) / img_w
            yc = (y + h / 2.0) / img_h
            wn = w / img_w
            hn = h / img_h
            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        lbl = dst_labels_dir / f"{Path(file_name).stem}.txt"
        lbl.write_text("\n".join(yolo_lines), encoding="utf-8")
        n += 1
    print(f"COCO {out_split}: wrote {n} images/labels")
    return n


# --- VisDrone --------------------------------------------------------------


def parse_visdrone_person_line(line: str) -> tuple[float, float, float, float] | None:
    parts = line.strip().split(",")
    if len(parts) < 6:
        return None
    left, top, w, h = map(float, parts[0:4])
    category = int(parts[5])
    if category not in PERSON_CATS or w <= 1 or h <= 1:
        return None
    return left, top, w, h


def visdrone_ann_to_yolo_lines(ann_path: Path, img_w: int, img_h: int) -> list[str]:
    if not ann_path.is_file():
        return []
    out: list[str] = []
    for raw in ann_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parsed = parse_visdrone_person_line(raw)
        if parsed is None:
            continue
        left, top, w, h = parsed
        xc = (left + w / 2.0) / img_w
        yc = (top + h / 2.0) / img_h
        wn = w / img_w
        hn = h / img_h
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        wn = max(0.0, min(1.0, wn))
        hn = max(0.0, min(1.0, hn))
        out.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    return out


def merge_visdrone_split(
    split_root: Path,
    dst_images: Path,
    dst_labels: Path,
    label: str,
) -> tuple[int, int]:
    img_dir = split_root / VIS_IMAGES_DIR
    ann_dir = split_root / VIS_ANN_DIR
    if not img_dir.is_dir() or not ann_dir.is_dir():
        print(f"Skip VisDrone {label}: missing images/ or annotations/")
        return 0, 0
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    n_ok, n_skip = 0, 0
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        ann_path = ann_dir / f"{stem}.txt"
        with Image.open(img_path) as im:
            iw, ih = im.size
        lines = visdrone_ann_to_yolo_lines(ann_path, iw, ih)
        if not lines:
            n_skip += 1
            continue
        out_img = dst_images / f"vd_{stem}.jpg"
        out_lbl = dst_labels / f"vd_{stem}.txt"
        shutil.copy2(img_path, out_img)
        out_lbl.write_text("\n".join(lines), encoding="utf-8")
        n_ok += 1
    print(f"VisDrone {label}: copied {n_ok} with person; skipped {n_skip} (no pedestrian/people).")
    return n_ok, n_skip


def visdrone_ann_has_person(ann_path: Path) -> bool:
    if not ann_path.is_file():
        return False
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        if int(parts[5]) in PERSON_CATS:
            return True
    return False


# --- Test split ------------------------------------------------------------


def yolo_label_nonempty(lbl_path: Path) -> bool:
    return lbl_path.is_file() and bool(lbl_path.read_text(encoding="utf-8").strip())


def move_val_to_test(img_path: Path, lbl_path: Path, img_test: Path, lbl_test: Path) -> None:
    img_test.mkdir(parents=True, exist_ok=True)
    lbl_test.mkdir(parents=True, exist_ok=True)
    dst_img = img_test / img_path.name
    dst_lbl = lbl_test / f"{img_path.stem}.txt"
    shutil.move(str(img_path), str(dst_img))
    if lbl_path.is_file():
        shutil.move(str(lbl_path), str(dst_lbl))
    else:
        dst_lbl.write_text("", encoding="utf-8")


def copy_visdrone_neg_to_test(
    img_path: Path, img_test: Path, lbl_test: Path
) -> None:
    stem = img_path.stem
    out_name = f"vd_neg_{stem}.jpg"
    img_test.mkdir(parents=True, exist_ok=True)
    lbl_test.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, img_test / out_name)
    (lbl_test / f"vd_neg_{stem}.txt").write_text("", encoding="utf-8")


def build_test_split(
    dst_root: Path,
    vis_root: Path,
    fraction: float,
    seed: int,
) -> None:
    random.seed(seed)
    img_val = dst_root / "images" / "val"
    lbl_val = dst_root / "labels" / "val"
    img_test = dst_root / "images" / "test"
    lbl_test = dst_root / "labels" / "test"

    _clear_merged_split_dirs(dst_root, ("test",))
    img_test.mkdir(parents=True, exist_ok=True)
    lbl_test.mkdir(parents=True, exist_ok=True)

    coco_imgs: list[Path] = []
    vd_pos: list[Path] = []
    for p in sorted(img_val.glob("*.jpg")):
        (vd_pos if p.name.startswith("vd_") else coco_imgs).append(p)

    coco_w = [p for p in coco_imgs if yolo_label_nonempty(lbl_val / f"{p.stem}.txt")]
    coco_wo = [p for p in coco_imgs if not yolo_label_nonempty(lbl_val / f"{p.stem}.txt")]

    def sample(lst: list[Path]) -> list[Path]:
        if not lst:
            return []
        k = max(1, int(len(lst) * fraction))
        k = min(k, len(lst))
        return random.sample(lst, k)

    take_cw, take_co, take_vp = sample(coco_w), sample(coco_wo), sample(vd_pos)
    print(
        f"Test split: COCO val {len(coco_imgs)} ({len(coco_w)} w/ person, {len(coco_wo)} w/o) "
        f"-> test {len(take_cw)} / {len(take_co)}; vd val pos {len(vd_pos)} -> test {len(take_vp)}"
    )
    for img in take_cw + take_co + take_vp:
        move_val_to_test(img, lbl_val / f"{img.stem}.txt", img_test, lbl_test)

    raw_img = vis_root / VIS_VAL_DIR / VIS_IMAGES_DIR
    raw_ann = vis_root / VIS_VAL_DIR / VIS_ANN_DIR
    vd_neg: list[Path] = []
    if raw_img.is_dir() and raw_ann.is_dir():
        for img in sorted(raw_img.glob("*.jpg")):
            if not visdrone_ann_has_person(raw_ann / f"{img.stem}.txt"):
                vd_neg.append(img)
    take_vn = sample(vd_neg)
    print(f"Test split: VisDrone raw val no-person {len(vd_neg)} candidates -> test {len(take_vn)}")
    for img in take_vn:
        copy_visdrone_neg_to_test(img, img_test, lbl_test)


def write_data_yaml(dst_root: Path, include_test: bool) -> None:
    lines = [
        f"path: {dst_root.as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if include_test:
        lines.append("test: images/test")
    lines.extend(["", "names:", "  0: person", ""])
    (dst_root / "data.yaml").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {dst_root / 'data.yaml'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build YOLO person dataset from COCO2017 + VisDrone2019.")
    ap.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Dataset root (default: {DEFAULT_DATASET_ROOT})",
    )
    ap.add_argument("--skip-test", action="store_true", help="Only build train/val; no test split.")
    ap.add_argument("--test-fraction", type=float, default=0.12, help="Fraction per stratum for test.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for test sampling.")
    args = ap.parse_args()

    root = args.dataset.resolve()
    coco_root = root / DEFAULT_COCO_DIR
    vis_root = root / DEFAULT_VISDRONE_DIR

    if not (coco_root / COCO_ANN_DIR / COCO_TRAIN_ANN).is_file():
        raise SystemExit(f"Missing COCO annotations under {coco_root}")
    if not (vis_root / VIS_TRAIN_DIR / VIS_IMAGES_DIR).is_dir():
        print(f"Warning: VisDrone train not found under {vis_root}; skip VisDrone merge.")

    print("Clearing merged train/val (official source folders are not deleted)...")
    _clear_merged_split_dirs(root, ("train", "val"))
    _remove_label_caches(root)

    convert_coco_split(coco_root, root, COCO_TRAIN_IMAGES_DIR, COCO_TRAIN_ANN, "train")
    convert_coco_split(coco_root, root, COCO_VAL_IMAGES_DIR, COCO_VAL_ANN, "val")

    vt = vis_root / VIS_TRAIN_DIR
    vv = vis_root / VIS_VAL_DIR
    if vt.is_dir():
        merge_visdrone_split(vt, root / "images" / "train", root / "labels" / "train", "train")
    if vv.is_dir():
        merge_visdrone_split(vv, root / "images" / "val", root / "labels" / "val", "val")

    include_test = not args.skip_test
    if include_test:
        build_test_split(root, vis_root, args.test_fraction, args.seed)
    else:
        _clear_merged_split_dirs(root, ("test",))

    _remove_label_caches(root)
    write_data_yaml(root, include_test=include_test)

    print("Done.")
    if include_test:
        print(
            "Evaluate on test: yolo detect val model=YOUR.pt "
            f"data={root.as_posix()}/data.yaml split=test device=0"
        )


if __name__ == "__main__":
    main()
