#!/usr/bin/env python3
"""Day 10, Pha 1: kiem 9 loi luat, Python 3.10+, chi thu vien chuan.

Chay tu thu muc goc repo:
  python validator/validate.py runs/r45007/delivery --out runs/r45007/errors.json
  python validator/validate.py runs/r45007/delivery --out runs/r45007/rules.json \
      --details runs/r45007/rules_details.json

Chi doc du lieu ben trong delivery, khong sua v1, khong doc dap an/khong mang.
--candidates duoc ho tro ve giao dien nhung tra danh sach rong: Pha 2 chua lam.
Exit 0: da kiem xong (co the co loi dataset); exit 2: khong kiem duoc day du.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zlib


# YOLO cua lab lam tron 6 chu so. Sai so mep toi da ~0.75e-6.
# Khong dung 0.01: nguong do se bo sot box chi tran mep mot chut.
ROUNDING_EPS = 1e-6
DUPLICATE_SHIFT_PX = 2.0  # Theo GUIDE: box cung kich thuoc, lech 1-2 pixel.
RULE_TYPES = (
    "missing_image", "wrong_hash", "invalid_class", "bbox_out_of_bounds",
    "duplicate_box", "scene_leak", "guideline_mismatch",
    "invalid_category_id", "thing_overlap",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def inside(root: Path, relative: str) -> Path:
    """Khong cho manifest/path/symlink dan ra ngoai delivery."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Duong dan nam ngoai delivery: {relative}")
    return path


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_rgb_png(path: Path) -> tuple[int, int, bytes]:
    """Doc mask PNG RGB 8-bit, ca 5 bo loc PNG; khong can Pillow.

    ID panoptic = R + G*256 + B*65536, KHONG phai gia tri RGB big-endian.
    """
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Khong phai PNG: {path}")
    pos, width, height = 8, 0, 0
    compressed = bytearray()
    ended = False
    while pos + 12 <= len(data):
        length = struct.unpack_from(">I", data, pos)[0]
        kind = data[pos + 4:pos + 8]
        end = pos + 8 + length
        if end + 4 > len(data):
            raise ValueError(f"PNG bi cat ngan: {path}")
        payload = data[pos + 8:end]
        crc = zlib.crc32(payload, zlib.crc32(kind)) & 0xFFFFFFFF
        if crc != struct.unpack_from(">I", data, end)[0]:
            raise ValueError(f"PNG CRC sai: {path}")
        if kind == b"IHDR":
            width, height, depth, color, comp, filt, inter = struct.unpack(">IIBBBBB", payload)
            if (depth, color, comp, filt, inter) != (8, 2, 0, 0, 0):
                raise ValueError(f"Can PNG RGB 8-bit, non-interlaced: {path}")
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            ended = True
            break
        pos = end + 4
    if not ended or width <= 0 or height <= 0:
        raise ValueError(f"PNG thieu IHDR/IEND hop le: {path}")
    raw = zlib.decompress(compressed)
    stride = width * 3
    if len(raw) != height * (stride + 1):
        raise ValueError(f"PNG sai kich thuoc pixel: {path}")
    previous = bytearray(stride)
    pixels = bytearray()
    for y in range(height):
        offset = y * (stride + 1)
        filt = raw[offset]
        row = bytearray(raw[offset + 1:offset + stride + 1])
        if filt not in range(5):
            raise ValueError(f"PNG filter khong hop le: {filt}")
        if filt:
            for i in range(stride):
                left = row[i - 3] if i >= 3 else 0
                up = previous[i]
                upper_left = previous[i - 3] if i >= 3 else 0
                if filt == 1:
                    predictor = left
                elif filt == 2:
                    predictor = up
                elif filt == 3:
                    predictor = (left + up) // 2
                else:
                    estimate = left + up - upper_left
                    a, b, c = abs(estimate - left), abs(estimate - up), abs(estimate - upper_left)
                    predictor = left if a <= b and a <= c else up if b <= c else upper_left
                row[i] = (row[i] + predictor) & 255
        pixels.extend(row)
        previous = row
    return width, height, bytes(pixels)


def read_labels(path: Path) -> list[tuple[int, float, float, float, float, float]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}, dong {number}: can dung 5 cot YOLO")
        try:
            rows.append((number, *map(float, fields)))
        except ValueError as exc:
            raise ValueError(f"{path}, dong {number}: gia tri YOLO khong phai so") from exc
    return rows


def duplicate_pairs(rows, width: int, height: int):
    """So kich thuoc + lech tam; IoU cao mot minh KHONG du de ket luan."""
    valid = [r for r in rows if all(math.isfinite(v) for v in r[1:])
             and r[1].is_integer() and 0 <= r[1] < 80 and r[4] > 0 and r[5] > 0]
    for i, a in enumerate(valid):
        for b in valid[i + 1:]:
            if a[1] != b[1]:
                continue
            if abs(a[4] - b[4]) > ROUNDING_EPS or abs(a[5] - b[5]) > ROUNDING_EPS:
                continue
            dx, dy = abs(a[2] - b[2]) * width, abs(a[3] - b[3]) * height
            if dx <= DUPLICATE_SHIFT_PX + ROUNDING_EPS * width \
                    and dy <= DUPLICATE_SHIFT_PX + ROUNDING_EPS * height:
                yield {"lines": [a[0], b[0]], "center_delta_px": [round(dx, 6), round(dy, 6)]}


def validate(delivery: Path) -> tuple[list[dict], list[dict], int]:
    root = delivery.resolve(strict=True)
    manifest = read_json(inside(root, "manifest.json"))["items"]
    metadata_rows = read_json(inside(root, "metadata.json"))["items"]
    metadata = {row["image"]: row for row in metadata_rows}
    if len(metadata) != len(metadata_rows) or set(metadata) != set(manifest):
        raise ValueError("metadata va manifest phai co cung tap anh, khong trung image")
    with inside(root, "jobs.csv").open(encoding="utf-8-sig", newline="") as handle:
        job_rows = list(csv.DictReader(handle))
    jobs = {row["job_id"]: row for row in job_rows}
    if len(jobs) != len(job_rows):
        raise ValueError("jobs.csv bi trung job_id")
    class_ids = {int(line.split(maxsplit=1)[0])
                 for line in inside(root, "classes.txt").read_text(encoding="utf-8-sig").splitlines()
                 if line.strip()}
    if class_ids != set(range(80)):
        raise ValueError("classes.txt cua lab phai khai bao 80 class_id tu 0 den 79")
    panoptic = read_json(inside(root, "panoptic.json"))
    categories = {row["id"]: row for row in panoptic["categories"]}
    images_by_id = {str(row["id"]): row for row in panoptic["images"]}
    annotations = {}
    for ann in panoptic["annotations"]:
        image = images_by_id[str(ann["image_id"])]
        name = Path(image["file_name"]).stem
        if name in annotations:
            raise ValueError(f"Panoptic trung annotation cho anh: {name}")
        annotations[name] = (image, ann)

    evidence = defaultdict(list)

    def add(kind, target, detail):
        evidence[(kind, target)].append(detail)

    scenes = defaultdict(set)
    for name in sorted(manifest):
        item, meta = manifest[name], metadata[name]
        scenes[meta["scene_id"]].add(meta["split"])
        job = jobs[meta["job_id"]]
        if meta["guideline_version"] != job["guideline"]:
            add("guideline_mismatch", name, {
                "job_id": meta["job_id"], "image_guideline": meta["guideline_version"],
                "job_guideline": job["guideline"],
            })

        # Kiem ca 3 hash. Thieu image KHONG bao them wrong_hash cho file khong ton tai.
        paths = {role: inside(root, item[role]) for role in ("image", "label", "mask")}
        for role, path in paths.items():
            if not path.is_file():
                if role == "image":
                    add("missing_image", name, {"file": item[role]})
                    continue
                raise ValueError(f"Thieu {role}: {path}; khong the kiem du 9 luat")
            actual = sha256(path)
            expected = item[f"{role}_sha256"].strip().lower()
            if actual != expected:
                add("wrong_hash", name, {"file": item[role], "expected": expected, "actual": actual})

        image, ann = annotations[name]
        width, height, pixels = read_rgb_png(paths["mask"])
        if (width, height) != (image["width"], image["height"]):
            raise ValueError(f"Kich thuoc mask va panoptic.images khong khop: {name}")
        rows = read_labels(paths["label"])
        for number, cls, x, y, w, h in rows:
            if not math.isfinite(cls) or not cls.is_integer() or int(cls) not in class_ids:
                add("invalid_class", name, {"line": number, "class_id": str(cls)})
            finite = all(math.isfinite(v) for v in (x, y, w, h))
            if not finite or w <= 0 or h <= 0 or \
                    min(x - w / 2, y - h / 2) < -ROUNDING_EPS or \
                    max(x + w / 2, y + h / 2) > 1 + ROUNDING_EPS:
                add("bbox_out_of_bounds", name, {
                    "line": number, "xywh_normalized": [str(v) for v in (x, y, w, h)],
                })
        for pair in duplicate_pairs(rows, width, height):
            add("duplicate_box", name, pair)

        # Dem pixel theo segment ID, bo nen id=0; khong so bbox cac segment.
        counts = Counter(r + (g << 8) + (b << 16)
                         for r, g, b in zip(pixels[0::3], pixels[1::3], pixels[2::3]))
        for segment in ann["segments_info"]:
            category = categories.get(segment["category_id"])
            if category is None:
                add("invalid_category_id", name, {
                    "segment_id": segment["id"], "category_id": segment["category_id"],
                })
                continue
            if segment["id"] == 0 or not category["isthing"]:
                continue
            declared, actual = segment["area"], counts[segment["id"]]
            if declared > actual:
                add("thing_overlap", name, {
                    "segment_id": segment["id"], "declared_area": declared, "mask_pixels": actual,
                })

    for scene, splits in sorted(scenes.items()):
        if len(splits) > 1:
            add("scene_leak", scene, {"splits": sorted(splits)})

    # Cham theo cap (type, target): nhieu box/segment cung loi chi xuat mot lan.
    errors = [{"type": kind, "target": target} for kind, target in sorted(evidence)]
    details = [{**error, "evidence": evidence[(error["type"], error["target"])]}
               for error in errors]
    return errors, details, len(manifest)


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Day 10: 9 loi luat, chi doc delivery (Pha 1).")
    parser.add_argument("delivery", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="errors.json hoac rules.json")
    parser.add_argument("--details", type=Path, help="JSON bang chung tung loi, khong dung de cham")
    parser.add_argument("--candidates", type=Path, help="Pha 1 chi xuat candidates rong; chua doi chieu nhan")
    args = parser.parse_args(argv)
    try:
        root = args.delivery.resolve(strict=True)
        outputs = [p.resolve() for p in (args.out, args.details, args.candidates) if p is not None]
        if any(p.is_relative_to(root) for p in outputs):
            raise ValueError("Output phai nam NGOAI delivery de giu nguyen v1")
        if len(set(outputs)) != len(outputs):
            raise ValueError("--out, --details, --candidates phai la cac file khac nhau")
        errors, details, total = validate(root)
        write_json(args.out, {"errors": errors})
        if args.details:
            write_json(args.details, {"findings": details})
        if args.candidates:
            write_json(args.candidates, {"candidates": []})
            print("LUU Y: Pha 1 chua sinh ung vien loi gan nhan (Pha 2).", file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError, struct.error, zlib.error) as exc:
        print(f"KHONG KIEM XONG: {exc}", file=sys.stderr)
        return 2
    counts = Counter(error["type"] for error in errors)
    print(f"Da kiem {total} anh; {len(errors)} cap (type, target).")
    for kind in RULE_TYPES:
        print(f"  {kind}: {counts[kind]}")
    print(f"Da ghi: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
