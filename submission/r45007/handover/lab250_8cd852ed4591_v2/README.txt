Release: lab250_8cd852ed4591_v2
Parent: lab250_8cd852ed4591
Số ảnh: 242
Số nhãn: 1683
Guideline: v1.2 (J01-J03), v1.3 (J04-J10)
QC v1: 16/250 = 6,4%
Kết luận v1: Hold — tỉ lệ lỗi 16/250 (6,4%) vượt ngưỡng 2% (Day 7), không phát hành v1
Kết luận v2: Release — đã tách 8 ảnh lỗi detection/thiếu file sang rework_queue.csv, sửa các lỗi hash/guideline/split trong v2, giữ nguyên nhãn gốc của 242 ảnh đạt chuẩn
Ghi chú: Sửa scene_leak: scene_514 (chuyển 000000017029 về train) | Sửa wrong_hash: đã mở kiểm tra ảnh 000000019432 còn nguyên vẹn không hỏng, cập nhật sha256 thật trong manifest.csv | Sửa guideline_mismatch: chuẩn hóa guideline của 000000001503 theo đúng jobs.csv trong truyvet.csv | Rework: 8 ảnh giao lại đúng cặp annotator/reviewer (A02/Q01, A03/Q02, A04/Q02, A05/Q02, A05/Q03) theo jobs.csv | Báo team panoptic (5 ảnh chỉ lỗi panoptic, nhãn detection đúng nên ở lại v2): 000000001584 (category_swap), 000000006818 (invalid_category_id), 000000012748 (invalid_category_id), 000000022479 (instance_merge), 000000023126 (thing_overlap)
