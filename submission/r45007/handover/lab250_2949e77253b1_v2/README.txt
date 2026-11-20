Release: lab250_2949e77253b1_v2
Parent: lab250_2949e77253b1
Số ảnh: 242
Số nhãn: 1683
QC v1: 16/250
Kết luận v1: Hold (ngưỡng 2%)
Kết luận v2: Release
Ghi chú:
Lỗi panoptic cần team panoptic xử lý; ảnh vẫn được giữ trong v2 detection:
- 000000001584: category_swap
- 000000022479: instance_merge
- 000000006818: invalid_category_id
- 000000012748: invalid_category_id
- 000000023126: thing_overlap
wrong_hash 000000019432: ảnh mở được và nội dung đúng; cập nhật SHA256 trong manifest v2.
guideline_mismatch 000000001503: guideline v2 lấy theo jobs.csv.
scene_leak scene_514: thống nhất split train trong truyvet.csv của v2.
