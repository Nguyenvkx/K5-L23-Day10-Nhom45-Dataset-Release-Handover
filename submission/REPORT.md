# Báo cáo nhóm 45

Seed luyện tập: 45007. Lô v1: `lab250_2949e77253b1` (250 ảnh). Lô v2: `lab250_2949e77253b1_v2`.

## 1. Số liệu luyện tập

- F1 theo seed (rule / cross_check / overall), do `score_findings.py` in làm tròn 1 chữ số:
  - seed 45007: ≈ 1.0 / ≈ 0.8 / ≈ 0.9
  - Truy vết (tier_a / tier_b / notify): ≈ 1.0 / ≈ 1.0 / ≈ 1.0
- Danh sách nộp: 16 mục (type, target) = 11 lỗi luật + 5 lỗi gán nhãn đã xác nhận.
  - Lỗi luật (11): `missing_image` 2, `wrong_hash` 1, `invalid_class` 1, `bbox_out_of_bounds` 1, `duplicate_box` 1, `scene_leak` 1, `guideline_mismatch` 1, `invalid_category_id` 2, `thing_overlap` 1.
  - Lỗi gán nhãn đã xác nhận (5): `category_swap` 1, `class_swap` 2, `missing_box` 1, `instance_merge` 1.
- Số ứng viên trên lô sạch (`inject_release.py --seed 5 --no-faults`, 50 ảnh): **0 ứng viên và 0 lỗi luật**, ở cả ngưỡng mặc định lẫn ngưỡng đã nới.
- Hàng đợi trên seed luyện tập: **6 ảnh** (5 ảnh ở ngưỡng mặc định, 6 ảnh sau khi nới ngưỡng `bbox_drift`). Theo loại: `class_swap_or_category_swap` 3, `instance_merge` 1, `missing_box` 1, `bbox_drift` 1. Có thêm 8 ảnh bị bỏ qua ở bước đối chiếu vì đã có lỗi cấu trúc (thiếu file, nhãn sai lớp, box tràn mép, box lặp, category_id sai, thing_overlap), nên nguồn nhãn của chúng không đủ tin cậy để so det với panoptic.
- Ngưỡng đã chọn và cách chọn (không dùng `_private`):
  - Cách ghép: không chỉ dùng IoU bbox. Mỗi segment được dựng lại bbox từ pixel thật của mask, rồi ghép 1-1 với box theo điểm `0.65·IoU + 0.35·tỉ lệ pixel segment nằm trong box`, và chỉ nhận cặp khi đây là lựa chọn tốt nhất của cả hai phía.
  - `class_swap/category_swap`: IoU ≥ 0.60 và ≥ 0.85 pixel segment nằm trong box, nhưng tên lớp khác nhau.
  - `bbox_drift`: cùng lớp, IoU ≥ **0.15** (mặc định 0.25) và tỉ lệ pixel trong box < **0.95** (mặc định 0.90). Đây là hai ngưỡng duy nhất nhóm đã chỉnh, vì ở mặc định chưa có ứng viên `bbox_drift` nào trong khi lab có 5 loại lỗi gán nhãn.
  - `missing_box`: segment thing đủ lớn (≥ 128 pixel và ≥ 0.1% diện tích ảnh), không có box cùng lớp nào phủ quá 10% và không có cặp khớp vị trí.
  - `instance_merge`: một segment có phần pixel riêng ≥ 18% trong mỗi box của hai box cùng lớp, hợp hai phần ≥ 85% diện tích segment, hai box chồng nhau IoU ≤ 0.40.
  - Cách chọn: chạy trên lô sạch trước. Ở cả hai bộ ngưỡng lô sạch đều ra 0 ứng viên, tức là nới `bbox_drift` không làm tăng nhiễu tự nhiên. Trên seed luyện tập, việc nới thêm đúng 1 ảnh vào hàng đợi, đủ nhỏ để cả nhóm xem hết bằng mắt. Nhóm dừng ở mức này thay vì nới tiếp vì chưa có số đo nào cho thấy hạ thêm ngưỡng sẽ giữ được hàng đợi nhỏ.
  - Mọi ứng viên đều xem ảnh bằng công cụ `tools/draw_candidates.py` (vẽ box detection màu đỏ, viền segment màu xanh, ảnh gốc bên trái để đối chiếu). Dòng `review_log.csv` có 6 dòng, trong đó 5 dòng `confirmed` vào `errors.json` và 1 ứng viên `bbox_drift` không được xác nhận nên không vào danh sách nộp.

## 2. Vì sao báo hết ứng viên làm F1 giảm?

Ứng viên chỉ là giả thuyết từ hình học: hai nguồn nhãn lệch nhau, nhưng COCO có nhiều lệch tự nhiên (vật bị che một phần, box ôm cả vật lẫn phần che, ranh giới mask mỏng). F1 = 2·TP / (2·TP + FP + FN) tính trên cặp (type, target), nên mỗi ứng viên lệch tự nhiên mà vẫn báo là một FP, bị trừ ngang với một lỗi bị bỏ sót.

Với số của nhóm: hàng đợi có 6 ảnh nhưng chỉ 5 ảnh được xác nhận là lỗi thật khi xem ảnh. Nếu báo hết cả 6, nhóm lỗi gán nhãn có thêm 1 FP, precision của nhóm này tụt xuống 5/6 ≈ 0.83 (giả sử 5 mục kia đều đúng) thay vì 1.0, và F1 tổng giảm theo. Hàng đợi càng thô thì tình hình càng xấu: lab ghi rõ ghép chỉ bằng IoU với ngưỡng lỏng ra nhiều ứng viên, còn lô sạch của nhóm ra 0 nhờ dùng pixel mask. Bước xem ảnh và đánh dấu `rejected` có lý do chính là bộ lọc giữ precision cao. Ngược lại, điểm lỗi gán nhãn của nhóm chỉ ≈ 0.8 cho thấy recall hoặc loại lỗi vẫn còn chỗ sai (xem mục 6).

## 3. Quyết định và lineage

- QC v1: **16/250 = 6.4%**. Mỗi ảnh mang tối đa một lỗi và `scene_leak` (`scene_514`) tính một mục cho cả cảnh. Ngưỡng Day 7 là 2%, nên 6.4% > 2% → **Hold** (không Release v1).
- Lineage (`runs/lineage.jsonl`): event `decision` HOLD cho `lab250_2949e77253b1`, rồi event `release` cho `lab250_2949e77253b1_v2` với `--parent lab250_2949e77253b1`.
- Lệnh `lineage.py impact --release-id lab250_2949e77253b1` trả về `R-021: affected=True action=review_or_retrain`. Run R-021 đã train trên đúng release v1 đang bị Hold. Sửa dữ liệu ở v2 không tự sửa trọng số model đã học từ nhãn sai (ví dụ nhãn thiếu `missing_box`, nhãn `class_swap`, các cảnh rò train/val), nên R-021 phải được xem lại kết quả hoặc train lại trên v2.
- Kết quả bàn giao v2 (sản phẩm detection): **242 ảnh, 1683 nhãn**, manifest 484 file.
  - Tách 8 ảnh vào `rework_queue.csv`, giao đúng người gán/người kiểm theo `jobs.csv`: `missing_image` (000000000776, 000000016228), `bbox_out_of_bounds` (000000001268), `invalid_class` (000000015079), `duplicate_box` (000000021167), `class_swap` (000000007784, 000000025057), `missing_box` (000000014831).
  - Sửa ngay trong v2: `wrong_hash` 000000019432 (mở ảnh kiểm, ảnh còn nguyên nên chỉ ghi lại hash trong manifest), `guideline_mismatch` 000000001503 (lấy v1.2 theo job J01 trong `jobs.csv`), `scene_leak` `scene_514` (đưa cả cảnh về `train`).
  - Giữ lại trong v2 và báo team panoptic (ghi trong README.txt, phần Ghi chú): `invalid_category_id` (000000006818, 000000012748), `thing_overlap` (000000023126), `category_swap` (000000001584), `instance_merge` (000000022479). Nhãn detection của các ảnh này vẫn đúng.
  - Kết luận v2: Release cho sản phẩm detection, kèm các việc còn lại ở trên (rework giao lại, báo team panoptic, R-021 xem lại).
- v1 được giữ nguyên: `delivery/` không bị sửa hash, chỉ làm việc trong `handover/`.

## 4. Hash chứng minh gì, không chứng minh gì?

Hash SHA-256 chứng minh **tính toàn vẹn**: byte của file hôm nay đúng bằng byte lúc hash được ghi. Nhờ vậy:

- phát hiện được file bị đổi hoặc hỏng sau khi lập manifest;
- file zip của v2 là tái lập được (thứ tự file và thời gian cố định), cùng nội dung thì cùng mã `.sha256` trên mọi máy. Mã `.sha256` của v2 phải được tính lại bằng `pack_release.py` mỗi lần sửa file (sau khi chốt nhóm đã có mã `cd5ccabc…ef19f` cho 489 file).

Hash **không** chứng minh:

- nhãn đúng hay sai, vì file nhãn sai cũng có hash khớp. Ảnh `class_swap`, `missing_box`, `instance_merge` đều qua được kiểm hash;
- bản thân hash trong manifest đã đúng từ đầu. Ca `wrong_hash` 000000019432 cho thấy ảnh vẫn mở được, đúng nội dung, chỉ hash ghi trong manifest sai. Vì vậy phải mở ảnh kiểm trước khi kết luận ảnh hỏng hay manifest sai (ở đây sửa manifest, không rework);
- ảnh có đúng nguồn, có an toàn về quyền riêng tư (ví dụ khuôn mặt) hay không.

Tương tự, `check_handover.py` PASS chỉ nói lô nhất quán về định dạng, không nói nhãn đúng.

## 5. Truy vết

Ticket của team AI: model hay bỏ sót `cup` bị che hoặc ở mép ảnh, quan sát ở `scene_809` và `scene_931`, nhờ tìm thêm ảnh khác bị như vậy. Changelog: guideline **v1.3** đổi duy nhất lớp `cup` (vật bị che trên 50% hoặc cắt ở mép ảnh vẫn phải gán box), còn **v1.2** cho phép bỏ qua.

Theo `jobs.csv`: J01, J02, J03 dùng v1.2 (cũ), J04–J10 dùng v1.3. Hai cảnh trong ticket đều thuộc job J02 (v1.2).

- **tier_a** (3 ảnh): ảnh thuộc `scene_809`, `scene_931`, job guideline cũ, có ít nhất một box lớp `cup` trong `labels/`: 000000002157, 000000002431, 000000003661.
- **tier_b** (4 ảnh): ảnh khác ngoài hai cảnh trên, cùng điều kiện (job guideline cũ, có box `cup`), team AI chưa báo: 000000002592, 000000002685, 000000003845, 000000003934.
- **notify**: người gán và người kiểm của các job chứa ảnh tầng a và b. Tất cả đều thuộc J02, gán bởi A03, kiểm bởi Q02, nên notify = A03, Q02.

Vì sao lấy guideline theo **job** (`jobs.csv`) chứ không theo guideline ghi ở từng ảnh: quy tắc nào được áp dụng khi gán nhãn do job quyết định, còn trường `guideline_version` trong `metadata.json` của ảnh có thể chính là lỗi `guideline_mismatch`. Ví dụ ảnh 000000001503 ghi v1.3 trong khi job J01 dùng v1.2. Nếu tin metadata ảnh thì sẽ bỏ sót hoặc thêm nhầm ảnh vào tier_a/tier_b và báo nhầm người. Theo kết quả tự đo, truy vết đạt F1 ≈ 1.0 ở cả ba phần.

## 6. Bug hoặc điểm chưa rõ trong lab (nếu có)

- **Điểm lỗi gán nhãn chỉ ≈ 0.8** dù mọi mục nhóm báo đều đã xem ảnh. Có hai khả năng: còn một lỗi gán nhãn nhóm chưa bắt được (nhiều khả năng ở loại `bbox_drift`), hoặc một mục bị ghi nhầm chiều `class_swap`/`category_swap`. Vì `score_findings.py` chỉ in F1 làm tròn và không được dò từng ảnh bằng `_private`, nhóm chưa xác định được nguyên nhân. Ảnh `000000025057` đã xem lại: vật được khoanh là một đĩa ném (frisbee), panoptic đúng và detection ghi `bottle` sai nên `class_swap` là đúng chiều.
- **Thư mục lạ trong `delivery/`**: lô `runs/r45007/delivery/` có thêm `runs/r45007/errors.json/validation_report.csv` và `validation_report.json`, sinh ra khi một lần chạy công cụ kiểm tra ghi nhầm vào bên trong `delivery/`. `check_handover.py` H1 vẫn PASS vì chỉ kiểm `release_id` và hash của nhãn, mask, panoptic, nên không bắt được file thừa này. Cần xoá hoặc chuyển ra ngoài trước khi nộp để v1 đúng nghĩa giữ nguyên. Validator của nhóm có kiểm tra không cho ghi output vào trong `delivery/`.
- **Lineage có hai event `decision` HOLD trùng nhau** cho `lab250_2949e77253b1` do chạy lệnh hai lần. Không ảnh hưởng kết quả `impact` nhưng nên ghi rõ.
- **Chỉ số `H13`** tính số ảnh lỗi theo tập target khác nhau trong `errors.json`, trong đó `scene_leak` tính một mục cho cả cảnh dù cảnh có nhiều ảnh; số 16/250 phải đọc theo quy ước này.
- Cảnh báo nhỏ: Pillow 13 báo `getdata` bị deprecated (sẽ bị gỡ ở Pillow 14); công cụ vẽ ảnh vẫn chạy bình thường.