**Ngày chạy:** 08/11/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** 3.x / 2.x / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` 

**Thay đổi so với notebook nguồn:** Không 

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab; `rank`=1 ; `score`0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh**, vì `class_id`, `class_name` và `score` thể hiện kết quả phân loại tổng thể của model đối với hình ảnh, không xác định vị trí hay **bounding box** của một vật thể riêng lẻ.
3. Class list **đi kèm taxonomy/checkpoint** (ImageNet-1K với `yolo11n-cls.pt`), không phải model tự nghĩ ra tên lớp
4. Vì `class_id` giúp ổn định cho máy, `class_name` giúp người đọc dễ hiểu và `taxonomy_name` cho biết đang dùng lớp nào (tránh nhầm khi đổi dataset hoặc checkpoint)
5. Guideline cần nói rõ tiêu chí chọn lớp một cách cụ thể đối với trường hợp ảnh nhiều chủ đề (ví dụ: chủ thể lớn nhất / chủ đề chính của cảnh / lớp xuất hiện rõ nhất), để mọi annotator chọn thống nhất **một** lớp.
6. `Score` của model phản ánh những gì model nghĩ là đúng, `Ground Truth` phản ánh những gì đã xác minh là đúng theo tiêu chí đánh giá, dù kết quả score có tốt đến đâu thì vẫn chỉ là prediction, chưa đạt đến độ chính xác tuyệt 

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=bowl ; `score`0.719063 ; `bbox_xyxy`=  [
      32.65,
      342.12,
      100.16,
      384.93
    ]; `bbox_width`=67.51 ; `bbox_height`=42.81
2. Lớp `bowl`, score 0.719063; box từ góc trên-trái khoảng pixel (32.65, 342.12) đến góc dưới-phải (100.16, 384.93); rộng khoảng 68 px, cao khoảng 43 px.
3. Đối với ngưỡng 0.2 => số prediction sẽ nhiều hơn (điền số), với ngưỡng 0.6 => số prediction sẽ thấp hơn (điền số)
4. Khi giảm threshold, model giữ lại nhiều **bounding box** hơn, nên **reviewer** phải kiểm tra thêm số lượng box để xác định box nào đúng, box nào sai hoặc trùng lặp.
5. Quy tắc box: Box phải được vẽ sát vật thể, không cắt mất phần nhìn thấy của object; không vẽ một box chung cho hai object tách rời.
6. Trên ảnh `kitchen`, vật bị cắt mép bàn/đồ dùng gần biên ảnh: tôi chọn **guideline** nếu rule đã có (ví dụ “còn ≥ 50% object trong khung thì vẫn gán”); nếu không chắc object là gì vì che khuất nặng thì **escalation** hỏi Lab Coach, không tự bịa lớp.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-003 ; `class_name`=bowl ; `score`=0.698698 ; số điểm polygon=`polygon_point_count`=70 ; 3–5 điểm đầu của `polygon_xy`=[
        84.0,
        289.0
      ],
      [
        83.0,
        290.0
      ],
      [
        81.0,
        290.0
      ],
      [
        80.0,
        291.0
      ],
      [
        77.0,
        291.0
      ]
2. `Polygon` cung cấp thông tin về **đường viền** và **hình dạng thực tế** của vật thể, giúp xác định chính xác vùng pixel thuộc về vật thể, thay vì chỉ khoanh bằng hình chữ (bounding box)
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
4. `instance_id` dùng để phân biệt từng cá thể vật thể trong cùng một ảnh, không đại diện cho loại vật thể `class_id` hay theo dõi vật thể qua các frame `tracking_id`.
5. Chỗ 2 cái bát (bowl) chạm nhau trên bàn: nếu guideline chưa nói rõ tách biên như thế nào thì **escalation**; nếu guideline đã giải thích chi tiết thì làm theo guideline.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground Truth là dữ liệu nhãn chuẩn được gán cho ảnh hoặc từng vật thể, thể hiện đối tượng thuộc class nào, vị trí và hình dạng của chúng tùy theo tác vụ. | 

Classification: Khó xác định class khi ảnh chứa nhiều vật thể. Detection: Bounding box lệch và trùng. Segmentation: Polygon có thể không sát đường viền hoặc nhầm instance.
Ví dụ: Đối với 2 vật thể chồng lên nhau trong ảnh, khó để có thể segmentation  |

| Annotator xác định đối tượng, gán nhãn class và tạo nhãn tương ứng (label, bounding box hoặc polygon/mask) theo từng tác vụ. | 

Reviewer kiểm tra độ chính xác của nhãn, class, vị trí bounding box, đường viền polygon và tính nhất quán của instance ID.|

| --- | --- | --- | --- | --- |
Phân loại ảnh | Một lớp cho cả ảnh theo taxonomy ImageNet-1K | Top-1 có thể không khớp trực giác cảnh giao thông (prediction ≠ GT) | Đọc guideline chọn đúng 1 lớp cho ảnh | Đúng taxonomy, đúng phạm vi cấp ảnh, không chép score thành nhãn |

| Phát hiện vật thể | Mỗi object: class + box xyxy | Hạ threshold ra thêm hộp nhiễu; object cắt mép | Vẽ box theo rule, đủ object trong guideline | Box có ôm sát không, thiếu/thừa object, ca che khuất |

| Instance segmentation | Mỗi instance: class + polygon | Biên hai vật chạm nhau / vùng mờ | Vẽ mask bám biên từng instance | Mask có tràn object khác không, đủ instance cùng lớp |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ dùng ảnh COCO công khai có sẵn trong notebook, không tải ảnh cá nhân, khuôn mặt, biển số, dữ liệu nội bộ lên Github/Colab
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Dừng làm bài lab và báo với Lab Coach để có thể điều 

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS