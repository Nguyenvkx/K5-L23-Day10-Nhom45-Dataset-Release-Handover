#!/usr/bin/env python3
"""Day 10, Pha 1 + 2: loi luat va ung vien det-panoptic, Python 3.10+.

Chay tu thu muc goc repo:
  python validator/validate.py runs/r45007/delivery --out runs/r45007/errors.json
  python validator/validate.py runs/r45007/delivery --out runs/r45007/rules.json \
      --candidates runs/r45007/candidates.json --review-html runs/r45007/review.html

Chi doc du lieu ben trong delivery, khong sua v1, khong doc dap an/khong mang.
--out CHI chua loi luat. Ung vien KHONG tu dong tro thanh loi da xac nhan.
Exit 0: da kiem xong (co the co loi dataset); exit 2: khong kiem duoc day du.
"""

from __future__ import annotations

import argparse
import base64
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

# Nguong hinh hoc cho hang doi REVIEW, khong phai nguong chung minh loi.
# Khong dung diem cham/ten anh/seed de chon nguong.
CANDIDATE_THRESHOLDS = {
    "min_segment_pixels": 64,
    "missing_min_pixels": 128,
    "missing_min_image_fraction": 0.001,
    "class_min_iou": 0.60,
    "class_min_mask_coverage": 0.85,
    "drift_min_iou": 0.25,
    "drift_max_mask_coverage": 0.90,
    "missing_max_mask_coverage": 0.10,
    "merge_min_part_coverage": 0.18,
    "merge_min_union_coverage": 0.85,
    "merge_max_box_iou": 0.40,
}


def rect_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def rect_intersection(a, b):
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def box_iou(a, b):
    intersection = rect_area(rect_intersection(a, b))
    union = rect_area(a) + rect_area(b) - intersection
    return intersection / union if union else 0.0


def mask_spans(width, height, pixels, wanted):
    """RLE tung hang: (y, x_start, x_end_exclusive), dung pixel that cua mask."""
    spans = {sid: [] for sid in wanted}
    for y in range(height):
        start, previous = 0, None
        offset = y * width * 3
        for x in range(width):
            i = offset + x * 3
            sid = pixels[i] + (pixels[i + 1] << 8) + (pixels[i + 2] << 16)
            if sid != previous:
                if previous in spans:
                    spans[previous].append((y, start, x))
                start, previous = x, sid
        if previous in spans:
            spans[previous].append((y, start, width))
    return spans


def pixels_in_box(spans, box):
    # Pixel thuoc box neu TAM pixel nam trong hinh chu nhat [left, right).
    left, top, right, bottom = [math.ceil(v - 0.5) for v in box]
    return sum(max(0, min(end, right) - max(start, left))
               for y, start, end in spans if top <= y < bottom)


def semantic_candidates(name, rows, ann, categories, class_names, width, height, pixels, thresholds=None):
    """Chi sinh gia thuyet can nguoi xem anh; khong quyet dinh ben nao gan sai."""
    t = dict(CANDIDATE_THRESHOLDS if thresholds is None else thresholds)
    detections = []
    for line, cls, x, y, w, h in rows:
        if not all(math.isfinite(v) for v in (cls, x, y, w, h)) or not cls.is_integer():
            continue
        if int(cls) not in class_names or w <= 0 or h <= 0:
            continue
        detections.append({"line": line, "class": class_names[int(cls)],
                           "bbox": [(x - w/2)*width, (y - h/2)*height,
                                    (x + w/2)*width, (y + h/2)*height]})
    valid = [s for s in ann["segments_info"] if s["id"] != 0
             and s["category_id"] in categories
             and categories[s["category_id"]]["isthing"] and not s.get("iscrowd", 0)]
    spans_by_id = mask_spans(width, height, pixels, {s["id"] for s in valid})
    segments = []
    for s in valid:
        spans = spans_by_id[s["id"]]
        area = sum(end - start for _, start, end in spans)
        if area < t["min_segment_pixels"]:
            continue
        # Lay bbox tu pixel that; bbox ghi san trong JSON chi la metadata.
        bbox = [min(start for _, start, _ in spans), spans[0][0],
                max(end for _, _, end in spans), spans[-1][0] + 1]
        segments.append({"id": s["id"], "class": categories[s["category_id"]]["name"],
                         "bbox": bbox, "area": area, "spans": spans})

    pairs = {}
    for si, s in enumerate(segments):
        for di, d in enumerate(detections):
            iou = box_iou(s["bbox"], d["bbox"])
            count = pixels_in_box(s["spans"], d["bbox"]) if iou > 0 else 0
            coverage = count / s["area"]
            pairs[si, di] = {"iou": iou, "coverage": coverage,
                             "score": 0.65 * iou + 0.35 * coverage}
    found = defaultdict(list)

    def add(kind, sis, dis, reason, metrics):
        found[kind].append({"reason": reason,
                           "detection": [detections[i] for i in dis],
                           "panoptic": [{k: v for k, v in segments[i].items() if k != "spans"} for i in sis],
                           "metrics": {k: round(v, 4) for k, v in metrics.items()}})

    # Mot segment phu hai box rieng biet: tinh PHAN PIXEL RIENG, khong cong lap.
    merged = set()
    for si, s in enumerate(segments):
        eligible = [di for di, d in enumerate(detections) if d["class"] == s["class"]
                    and t["merge_min_part_coverage"] <= pairs[si, di]["coverage"] <= 0.85]
        best = None
        for k, di in enumerate(eligible):
            for dj in eligible[k + 1:]:
                a, b = detections[di], detections[dj]
                if box_iou(a["bbox"], b["bbox"]) > t["merge_max_box_iou"]:
                    continue
                shared = pixels_in_box(s["spans"], rect_intersection(a["bbox"], b["bbox"])) / s["area"]
                ca, cb = pairs[si, di]["coverage"], pairs[si, dj]["coverage"]
                union = ca + cb - shared
                if min(ca - shared, cb - shared) < t["merge_min_part_coverage"]:
                    continue
                if union < t["merge_min_union_coverage"]:
                    continue
                # Khong ghep nham mot box da co segment khac khop tot hon.
                if any(other != si and ss["class"] == s["class"]
                       and pairs[other, dd]["iou"] > pairs[si, dd]["iou"] + 0.15
                       and pairs[other, dd]["coverage"] >= 0.85
                       for dd in (di, dj) for other, ss in enumerate(segments)):
                    continue
                if best is None or union > best[0]:
                    best = (union, di, dj, ca, cb, shared)
        if best:
            union, di, dj, ca, cb, shared = best
            merged.add(si)
            add("instance_merge", [si], [di, dj],
                "Mot segment co pixel rieng trong hai box cung lop; can xem co hai vat hay khong.",
                {"coverage_box_1": ca, "coverage_box_2": cb, "shared_coverage": shared, "union_coverage": union})

    # Ghép 1-1 theo hinh hoc, khong khoa lop: neu khoa lop se bo sot doi lop.
    edges = sorted(((p["score"], si, di) for (si, di), p in pairs.items()
                    if p["iou"] >= 0.10 and p["coverage"] >= 0.20), reverse=True)
    matched_s, matched_d, matched = set(), set(), {}
    for _, si, di in edges:
        if si not in matched_s and di not in matched_d:
            matched_s.add(si)
            matched_d.add(di)
            matched[si] = di

    for si, s in enumerate(segments):
        if si in merged:
            continue
        di = matched.get(si)
        if di is not None:
            d, p = detections[di], pairs[si, di]
            # Cap khop phai la lua chon tot nhat cua ca hai phia.
            mutual = (all(p["score"] >= pairs[si, j]["score"] for j in range(len(detections)))
                      and all(p["score"] >= pairs[j, di]["score"] for j in range(len(segments))))
            if d["class"] != s["class"]:
                if mutual and p["iou"] >= t["class_min_iou"] and p["coverage"] >= t["class_min_mask_coverage"]:
                    add("class_swap_or_category_swap", [si], [di],
                        "Vi tri khop nhung ten lop khac; xem anh de chon class_swap HOAC category_swap, hoac rejected.",
                        {"bbox_iou": p["iou"], "mask_coverage": p["coverage"]})
            else:
                ratio = rect_area(d["bbox"]) / max(1, rect_area(s["bbox"]))
                if mutual and p["iou"] >= t["drift_min_iou"] and p["coverage"] < t["drift_max_mask_coverage"] \
                        and 0.35 <= ratio <= 3.0:
                    add("bbox_drift", [si], [di],
                        "Box cung lop de nhieu pixel segment ra ngoai; can phan biet lech box voi che khuat/cach gan nhan.",
                        {"bbox_iou": p["iou"], "mask_coverage": p["coverage"], "box_area_ratio": ratio})

        # Thieu box chi khi khong co counterpart cung lop hoac khop hinh hoc tot.
        if s["area"] < max(t["missing_min_pixels"], width * height * t["missing_min_image_fraction"]):
            continue
        same = [pairs[si, j] for j, d in enumerate(detections) if d["class"] == s["class"]]
        same_cov = max((p["coverage"] for p in same), default=0.0)
        same_iou = max((p["iou"] for p in same), default=0.0)
        spatial_counterpart = any(pairs[si, j]["iou"] >= 0.35 and pairs[si, j]["coverage"] >= 0.50
                                  for j in range(len(detections)))
        if same_cov < t["missing_max_mask_coverage"] and same_iou < 0.10 and not spatial_counterpart:
            add("missing_box", [si], [], "Segment thing du lon khong co box tuong ung; can xem vat co can gan box khong.",
                {"best_same_class_coverage": same_cov, "best_same_class_iou": same_iou,
                 "area_fraction": s["area"] / (width * height)})

    result = []
    for kind, evidence in sorted(found.items()):
        possible = ["class_swap", "category_swap"] if kind == "class_swap_or_category_swap" else [kind]
        result.append({"image": name, "type": kind, "possible_types": possible,
                       "status": "needs_review", "evidence": evidence})
    return result


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


def validate(delivery: Path, phase2: dict | None = None) -> tuple[list[dict], list[dict], int]:
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
    class_names = {int(parts[0]): parts[1].strip()
                   for line in inside(root, "classes.txt").read_text(encoding="utf-8-sig").splitlines()
                   if line.strip() for parts in [line.split(maxsplit=1)]}
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

        if phase2 is not None:
            # Nguon thieu/hong ve cau truc khong du tin cay de doi chieu semantic.
            blockers = [kind for kind in ("missing_image", "invalid_class", "bbox_out_of_bounds",
                         "duplicate_box", "invalid_category_id", "thing_overlap") if (kind, name) in evidence]
            if blockers:
                phase2.setdefault("skipped", []).append({"image": name, "reasons": blockers})
            else:
                candidates = semantic_candidates(name, rows, ann, categories, class_names, width, height, pixels)
                phase2.setdefault("candidates", []).extend(candidates)
                if candidates and phase2.get("include_review"):
                    wanted = {s["id"] for c in candidates for e in c["evidence"] for s in e["panoptic"]}
                    regions = mask_spans(width, height, pixels, wanted)
                    phase2.setdefault("assets", {})[name] = {
                        "width": width, "height": height,
                        "image_base64": base64.b64encode(paths["image"].read_bytes()).decode("ascii"),
                        "spans": {str(k): v for k, v in regions.items()},
                        "detections": [{"line": line, "class": class_names[int(cls)],
                                        "bbox": [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]}
                                       for line, cls, x, y, w, h in rows],
                    }

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


def write_review_html(path: Path, document: dict, assets: dict) -> None:
    """Trang HTML offline, khong gui anh di dau, khong tu dong chon verdict."""
    payload = json.dumps({"document": document, "assets": assets}, ensure_ascii=False, allow_nan=False)
    # JSON trong script phai escape '<' de ten file/note khong dong the script.
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    signature = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    page = r'''<!doctype html>
<html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Day 10 — Review detection và panoptic</title>
<style>
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:#f3f6fa;color:#172b40;font:16px system-ui,sans-serif}
header{background:#142e48;color:white;padding:22px 28px}h1{font-size:24px;margin:0 0 8px}header p{margin:5px 0;line-height:1.5}
main{padding:20px;max-width:1500px;margin:auto}.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:14px}
button,.file{background:white;color:#142e48;border:1px solid #a9b9c9;border-radius:6px;padding:9px 13px;cursor:pointer;font:inherit}
button:hover,.file:hover{background:#e5eef8}button:disabled{opacity:.4;cursor:default}.primary{background:#155c9b;color:white}
.primary:hover{background:#104b80}.grid{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(320px,1fr);gap:18px}
.card{background:white;border:1px solid #d4dfeb;border-radius:10px;padding:18px;min-width:0}
.viewport{overflow:auto;max-height:75vh;background:#07131e;padding:8px}canvas{display:block;margin:auto}
label{display:block;margin:12px 0 5px}select,input[type=text],textarea{width:100%;font:inherit;padding:9px;border:1px solid #a9b9c9;border-radius:5px}
textarea{min-height:115px;resize:vertical}#jump{width:auto;max-width:600px}.hint{font-size:14px;color:#40576e;line-height:1.5}
.status{font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#f3f6fa;padding:12px;max-height:320px;overflow:auto}
.inline{display:inline-flex;gap:5px;align-items:center;margin:0}.empty{padding:25px;text-align:center}#message{color:#9b3500;font-weight:600}
@media(max-width:900px){.grid{grid-template-columns:1fr}header{padding:18px}}
</style>
<header><h1>Review detection ↔ panoptic</h1>
<p>Ứng viên là dấu hiệu cần kiểm tra. Thành viên nhóm phải xem ảnh trước khi xác nhận hoặc bác bỏ.</p>
<p>Box detection: <b style="color:#38d9ff">xanh cyan</b> · Segment panoptic: <b style="color:#ffbd4a">cam</b>. Bật/tắt lớp phủ và phóng to để nhìn vật.</p></header>
<main>
<div class="bar"><button id="prev">← Trước</button><select id="jump" aria-label="Chọn ứng viên"></select><button id="next">Sau →</button><span id="progress" class="status"></span></div>
<div class="bar"><button id="csv" class="primary">Xuất review_log.csv</button><button id="backup">Lưu tiến độ JSON</button>
<label class="file" style="margin:0">Nạp tiến độ JSON<input id="restore" type="file" accept=".json" hidden></label></div>
<p id="message" role="status"></p>
<div id="empty" class="empty" hidden>Không có ứng viên theo các ngưỡng hiện tại. Điều này không chứng minh toàn bộ nhãn đều đúng.</div>
<div id="workspace" class="grid">
<section class="card"><h2 id="title" style="font-size:19px;margin-top:0"></h2>
<div class="bar"><label class="inline"><input type="checkbox" id="showBoxes" checked>Box detection</label>
<label class="inline"><input type="checkbox" id="showMasks" checked>Mask panoptic</label>
<label class="inline"><input type="checkbox" id="showContext" checked>Các box khác</label>
<label class="inline">Zoom <input type="range" id="zoom" min="50" max="300" value="100" step="25"><span id="zoomValue">100%</span></label></div>
<div class="viewport"><canvas id="canvas"></canvas></div><p class="hint">Box liên quan được tô sáng; các box khác nét mờ. Mask tô vùng pixel thật; khung cam là bbox của segment.</p>
</section>
<section class="card">
<p id="hypothesis" class="status"></p>
<label for="reviewer">Người xem</label><input id="reviewer" type="text" placeholder="Tên thành viên thực sự xem ảnh">
<label for="type">Loại lỗi đang đánh giá</label><select id="type"></select>
<label for="verdict">Kết luận của người xem</label><select id="verdict"><option value="">Chưa kết luận</option value="confirmed">confirmed — xác nhận có lỗi</option><option value="rejected">rejected — bác bỏ ứng viên</option></select>
<label for="note">Bạn thấy gì trong ảnh? (tối thiểu 20 ký tự)</label><textarea id="note" placeholder="Mô tả vật, vị trí, nguồn nhãn sai hoặc lý do đây là lệch tự nhiên."></textarea>
<p class="hint">Nếu khác lớp: chọn class_swap khi detection sai, category_swap khi panoptic sai. Nếu cả hai nhãn hợp lý, chọn rejected. Không chọn cả hai chỉ vì hai nguồn khác nhau.</p>
<details open><summary>Bằng chứng hình học</summary><pre id="evidence"></pre></details>
</section></div>
<p class="hint">Trang hoạt động offline. Nên lưu tiến độ JSON trước khi đóng trang. Xuất CSV chỉ khi tất cả ứng viên đã có người xem, loại lỗi, kết luận và ghi chú hợp lệ.</p>
</main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const payload=JSON.parse(document.getElementById('payload').textContent), candidates=payload.document.candidates;
const key='day10-review-__SIGNATURE__', $=id=>document.getElementById(id);
const itemKey=c=>c.image+'|'+c.type;
let state={},index=0,loadedImage=null,loadToken=0;
try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){$('message').textContent='Không thể tự lưu trong trình duyệt. Hãy dùng nút Lưu tiến độ JSON.'}
function entry(i){const c=candidates[i];return state[itemKey(c)]||{reviewer:'',type:c.possible_types.length===1?c.possible_types[0]:'',verdict:'',note:''}}
function valid(i){const c=candidates[i],r=entry(i);return r.reviewer.trim()&&r.note.trim().length>=20&&c.possible_types.includes(r.type)&&['confirmed','rejected'].includes(r.verdict)}
function progress(){const n=candidates.filter((c,i)=>valid(i)).length;$('progress').textContent=n+'/'+candidates.length+' đã review';$('prev').disabled=index===0;$('next').disabled=index>=candidates.length-1}
function persist(){try{localStorage.setItem(key,JSON.stringify(state))}catch(e){$('message').textContent='Không tự lưu được; dùng Lưu tiến độ JSON.'}}
function save(){if(!candidates.length)return;state[itemKey(candidates[index])]={reviewer:$('reviewer').value,type:$('type').value,verdict:$('verdict').value,note:$('note').value};persist();progress()}
function draw(){const c=candidates[index];if(!c||!loadedImage)return;const asset=payload.assets[c.image],canvas=$('canvas'),ctx=canvas.getContext('2d');
canvas.width=asset.width;canvas.height=asset.height;canvas.style.width=(asset.width*Number($('zoom').value)/100)+'px';canvas.style.height='auto';$('zoomValue').textContent=$('zoom').value+'%';
ctx.drawImage(loadedImage,0,0,asset.width,asset.height);const det=new Map(),seg=new Map();
for(const ev of c.evidence){for(const d of ev.detection)det.set(d.line,d);for(const s of ev.panoptic)seg.set(s.id,s)}
ctx.lineWidth=Math.max(1,asset.width/400);ctx.font=Math.max(12,asset.width/50)+'px sans-serif';
function label(text,x,y,color){const yy=Math.max(16,y);ctx.fillStyle='#07131e';ctx.fillRect(x,yy-15,ctx.measureText(text).width+6,19);ctx.fillStyle=color;ctx.fillText(text,x+3,yy)}
if($('showBoxes').checked&&$('showContext').checked){for(const d of asset.detections||[]){if(det.has(d.line))continue;const [x0,y0,x1,y1]=d.bbox;ctx.strokeStyle='rgba(50,210,255,.5)';ctx.strokeRect(x0,y0,x1-x0,y1-y0);label('D '+d.line+': '+d.class,x0,y0-3,'#9bdaea')}}
if($('showMasks').checked){ctx.fillStyle='rgba(255,160,40,.30)';for(const s of seg.values()){for(const [y,x0,x1] of asset.spans[String(s.id)]||[])ctx.fillRect(x0,y,x1-x0,1)}
for(const s of seg.values()){const [x0,y0,x1,y1]=s.bbox;ctx.strokeStyle='#ffb331';ctx.setLineDash([5,4]);ctx.strokeRect(x0,y0,x1-x0,y1-y0);ctx.setLineDash([]);label('P '+s.id+': '+s.class,x0,Math.min(asset.height-3,y1+14),'#ffbd4a')}}
if($('showBoxes').checked){for(const d of det.values()){const [x0,y0,x1,y1]=d.bbox;ctx.strokeStyle='#25d4ff';ctx.strokeRect(x0,y0,x1-x0,y1-y0);label('D line '+d.line+': '+d.class,x0,y0-3,'#38d9ff')}}}
function render(){progress();if(!candidates.length){$('empty').hidden=false;$('workspace').hidden=true;$('csv').disabled=true;return}
const c=candidates[index],r=entry(index);$('jump').value=String(index);$('title').textContent=c.image+' — '+(index+1)+'/'+candidates.length;
$('hypothesis').textContent=c.type;$('type').replaceChildren(new Option('Chọn sau khi xem ảnh',''),...c.possible_types.map(t=>new Option(t,t)));
for(const field of ['reviewer','type','verdict','note'])$(field).value=r[field]||'';
$('evidence').textContent=JSON.stringify(c.evidence,null,2);loadedImage=null;const token=++loadToken,img=new Image();
const ctx=$('canvas').getContext('2d');ctx.clearRect(0,0,$('canvas').width,$('canvas').height);
img.onload=()=>{if(token!==loadToken)return;loadedImage=img;draw()};img.onerror=()=>{$('message').textContent='Không đọc được ảnh '+c.image};img.src='data:image/jpeg;base64,'+payload.assets[c.image].image_base64}
for(let i=0;i<candidates.length;i++)$('jump').add(new Option((i+1)+'. '+candidates[i].image+' / '+candidates[i].type,String(i)));
$('prev').onclick=()=>{save();index=Math.max(0,index-1);render()};$('next').onclick=()=>{save();index=Math.min(candidates.length-1,index+1);render()};
$('jump').onchange=()=>{save();index=Number($('jump').value);render()};
for(const field of ['reviewer','type','verdict','note'])$(field).addEventListener('input',save);
for(const field of ['showBoxes','showMasks','showContext','zoom'])$(field).addEventListener('input',draw);
function download(name,text,mime){const url=URL.createObjectURL(new Blob([text],{type:mime})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
$('backup').onclick=()=>{save();download('review_progress.json',JSON.stringify({signature:key,state},null,2),'application/json')};
$('restore').onchange=async()=>{const file=$('restore').files[0];if(!file)return;try{const doc=JSON.parse(await file.text());if(doc.signature!==key||!doc.state||typeof doc.state!=='object')throw Error('Tiến độ thuộc bộ ứng viên khác');
for(const c of candidates){const r=doc.state[itemKey(c)];if(r){if(!['reviewer','type','verdict','note'].every(k=>typeof r[k]==='string'))throw Error('Dòng tiến độ không hợp lệ');if(r.type&&!c.possible_types.includes(r.type))throw Error('Loại lỗi không hợp lệ')}}
state=doc.state;persist();render();$('message').textContent='Đã nạp tiến độ.'}catch(e){$('message').textContent=e.message}};
function csvCell(s){return '"'+String(s).replaceAll('"','""')+'"'}
$('csv').onclick=()=>{save();const missing=candidates.findIndex((c,i)=>!valid(i));if(missing>=0){index=missing;render();$('message').textContent='Ứng viên này chưa đủ: người xem, loại lỗi, kết luận và ghi chú ít nhất 20 ký tự.';return}
const lines=['image,type,verdict,note'];candidates.forEach((c,i)=>{const r=entry(i);lines.push([c.image,r.type,r.verdict,'Người xem: '+r.reviewer.trim()+'; '+r.note.trim()].map(csvCell).join(','))});
download('review_log.csv',lines.join('\r\n')+'\r\n','text/csv;charset=utf-8');$('message').textContent='Đã xuất CSV. Chép vào thư mục run rồi chạy merge_review.py.'};
render();
</script></html>'''
    page = page.replace("__SIGNATURE__", signature).replace("__PAYLOAD__", payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Day 10: loi luat + ung vien det-panoptic (Pha 1-2).")
    parser.add_argument("delivery", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="errors.json hoac rules.json")
    parser.add_argument("--details", type=Path, help="JSON bang chung tung loi, khong dung de cham")
    parser.add_argument("--candidates", type=Path, help="Ung vien cho nguoi review; khong phai errors.json")
    parser.add_argument("--review-html", type=Path, help="Trang xem anh/box/mask offline va xuat review_log.csv")
    args = parser.parse_args(argv)
    try:
        root = args.delivery.resolve(strict=True)
        if args.review_html and not args.candidates:
            raise ValueError("--review-html can di kem --candidates")
        outputs = [p.resolve() for p in (args.out, args.details, args.candidates, args.review_html) if p is not None]
        if any(p.is_relative_to(root) for p in outputs):
            raise ValueError("Output phai nam NGOAI delivery de giu nguyen v1")
        if len(set(outputs)) != len(outputs):
            raise ValueError("Cac duong dan output phai khac nhau")
        phase2 = {"include_review": bool(args.review_html), "candidates": [], "skipped": []} if args.candidates else None
        errors, details, total = validate(root, phase2)
        write_json(args.out, {"errors": errors})
        if args.details:
            write_json(args.details, {"findings": details})
        if args.candidates:
            candidates = phase2["candidates"]
            summary = {"images_total": total, "images_with_candidates": len({c["image"] for c in candidates}),
                       "candidate_groups": len(candidates), "evidence_count": sum(len(c["evidence"]) for c in candidates),
                       "by_type": dict(sorted(Counter(c["type"] for c in candidates).items())),
                       "images_skipped_structural_errors": len(phase2["skipped"])}
            document = {"schema_version": 1, "note": "Gia thuyet can xem anh. Khong tu dong confirmed.",
                        "thresholds": CANDIDATE_THRESHOLDS, "summary": summary,
                        "skipped": phase2["skipped"], "candidates": candidates}
            write_json(args.candidates, document)
            if args.review_html:
                write_review_html(args.review_html, document, phase2.get("assets", {}))
            print(f"Pha 2: {len(candidates)} nhom ung vien / {summary['images_with_candidates']} anh.")
            print("Chi them vao errors.json sau khi nguoi review xac nhan.")
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
