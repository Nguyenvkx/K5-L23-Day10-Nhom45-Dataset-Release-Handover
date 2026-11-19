"""Render candidate review sheets with Pillow; never confirm or edit labels.

Place next to validate.py in validator/. Run from the repository root:
python validator/review_pillow.py runs/r45007/delivery --candidates runs/r45007/candidates.json --out runs/r45007/review_images
"""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from validate import inside


def main():
    ap = argparse.ArgumentParser(description="Create original/detection/panoptic review sheets")
    ap.add_argument("delivery", type=Path)
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    root = args.delivery.resolve(strict=True)
    output = args.out.resolve()
    if output.is_relative_to(root):
        ap.error("Output must be outside delivery to preserve v1")
    if output.exists() and any(output.iterdir()):
        ap.error("Output folder is not empty. Choose a new folder to preserve existing reviews.")
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))["candidates"]
    manifest = json.loads(inside(root, "manifest.json").read_text(encoding="utf-8"))["items"]
    classes = dict(line.split(maxsplit=1) for line in inside(root, "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip())
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    index = []
    pending = []
    for number, candidate in enumerate(candidates, 1):
        name = candidate["image"]
        item = manifest[name]
        with Image.open(inside(root, item["image"])) as source:
            original = source.convert("RGB")
        with Image.open(inside(root, item["mask"])) as source:
            mask = source.convert("RGB")
        width, height = original.size
        if mask.size != original.size:
            raise ValueError(f"Mask size mismatch: {name}")
        detection = original.copy()
        draw = ImageDraw.Draw(detection)
        highlighted = {d["line"] for e in candidate["evidence"] for d in e["detection"]}
        for line_no, line in enumerate(inside(root, item["label"]).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            cls, x, y, w, h = map(float, line.split())
            box = [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]
            color = "cyan" if line_no in highlighted else "#91b8c4"
            draw.rectangle(box, outline=color, width=3 if line_no in highlighted else 1)
            draw.text((max(0,box[0]),max(0,box[1]-12)), f"D{line_no}: {classes[str(int(cls))]}", fill=color, stroke_width=1, stroke_fill="black", font=font)
        segments = {s["id"]: s for e in candidate["evidence"] for s in e["panoptic"]}
        colors = [(255,155,30),(230,60,160),(100,255,90),(160,100,255)]
        palette = {sid: colors[i % len(colors)] for i,sid in enumerate(segments)}
        overlay = Image.new("RGBA", original.size)
        rgb = mask.tobytes()
        overlay.putdata([(*palette[sid],110) if sid in palette else (0,0,0,0)
                         for r,g,b in zip(rgb[0::3],rgb[1::3],rgb[2::3]) for sid in [r+(g<<8)+(b<<16)]])
        panoptic = Image.alpha_composite(original.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(panoptic)
        for sid, segment in segments.items():
            box = segment["bbox"]
            draw.rectangle(box, outline=palette[sid], width=2)
            draw.text((box[0],max(0,box[1]-12)), f"P{sid}: {segment['class']}", fill=palette[sid], stroke_width=1, stroke_fill="black", font=font)
        # Keep native pixels; the reviewer can zoom in their normal image viewer.
        sheet = Image.new("RGB", (width*3,height+60), "#142e48")
        draw = ImageDraw.Draw(sheet)
        draw.text((10,8), f"{number:03d} | {name} | {candidate['type']} | NEEDS HUMAN REVIEW", fill="white", font=font)
        for i,(panel,title) in enumerate([(original,"ORIGINAL"),(detection,"DETECTION: all boxes"),(panoptic,"PANOPTIC: candidate segments")]):
            draw.text((i*width+10,32), title, fill="white", font=font)
            sheet.paste(panel,(i*width,60))
        filename = f"{number:03d}.png"
        sheet.save(output/filename)
        index.append({"sheet":filename, **candidate})
        pending.append({"image":name,"type":candidate["possible_types"][0] if len(candidate["possible_types"])==1 else "", "verdict":"", "note":""})
    (output/"index.json").write_text(json.dumps(index,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (output/"review_log.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["image","type","verdict","note"])
        writer.writeheader()
        writer.writerows(pending)
    print(f"Created {len(index)} review sheets in {output}")
    print("Fill review_log.csv yourself: type, confirmed/rejected, and concrete observation + reviewer name.")
    print("For class disagreement choose class_swap OR category_swap after looking at the image.")


if __name__ == "__main__":
    main()
