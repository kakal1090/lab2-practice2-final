# Mở bài và chuẩn bị nộp

## 1. Bạn cần mở gì trước?

1. Mở `report/BAO_CAO_PRACTICE_2.pdf` để đọc toàn bộ bài.
2. Mở `results/comparison.csv` để kiểm tra bảng số liệu gốc.
3. Mở `PRACTICE_2_HOAN_CHINH.ipynb` bằng Colab. Các ô đầu dùng để xem kết quả có sẵn, không cần train lại. Chỉ bật `RUN_NEW_EXPERIMENTS=True` khi muốn chạy lại thật.
4. Đọc `VALIDATION.md` để biết những kiểm tra đã thực hiện.

Đây là một bộ thực nghiệm độc lập: ảnh 96×96, đủ CIFAR-10, cùng split và ngân sách giữa các lượt. Không thay kết quả DenseNet 224×224 của bạn khác bằng số liệu này mà vẫn giữ mô tả 224. Khi nộp, chọn một bộ điều kiện nhất quán làm bảng so sánh chính.

## 2. Phân công trình bày cho ba người

### Người 1 - Bạn: dữ liệu, ResNet18 baseline và tổng hợp

- Giải thích CIFAR-10 có 10 lớp; chia 45.000 train / 5.000 validation từ tập train gốc; giữ 10.000 test riêng.
- Chỉ ra `split_indices`, `loaders_data`, `make_model` trong `src/experiment.py`.
- Giải thích Resize, ToTensor, normalize ImageNet; train DataLoader shuffle, validation/test không shuffle.
- Mở `results/resnet_baseline/architecture.txt` và `parameters.json`: lớp `fc` đổi từ 1.000 thành 10 đầu ra; chỉ `fc.weight`, `fc.bias` được học.
- Trình bày CrossEntropyLoss, SGD momentum 0.9, vòng train/validation, cách chọn checkpoint tốt nhất.
- Đọc đúng hàng ResNet baseline trong bảng; lấy checkpoint theo validation, không theo test.
- Điền tên/MSSV, gộp nội dung, kiểm tra và duyệt PR.

### Người 2 - DenseNet121 và so sánh kiến trúc

- Đọc `results/densenet_baseline/config.json`, `architecture.txt`, `parameters.json`, `history.csv`, `test_metrics.json`.
- Giải thích DenseNet kết nối đặc trưng qua phép nối; ResNet dùng residual addition. Chỉ ra DenseNet thay `classifier` bằng `Linear(1024,10)`.
- So sánh số tham số tổng và số tham số được học của hai kiến trúc.
- Đối chiếu đường loss/accuracy và test của hai baseline trên cùng dữ liệu 96×96, split, LR, batch và epoch.
- Nhận xét chênh lệch bằng điểm phần trăm; không gọi đó là mức tăng phần trăm tương đối.
- Giải thích confusion matrix và ít nhất một lớp khó. Nhận xét từ số liệu; nếu muốn giải thích nguyên nhân thị giác phải xem ảnh thật.
- Đọc phần thời gian: head training đã cache feature, nên phải tính cả chi phí extraction khi thảo luận tốc độ.

### Người 3 - Hyperparameter và fine-tuning

- Đọc ba thư mục `resnet_lr_01`, `resnet_batch64`, `resnet_finetune` và đối chiếu `resnet_baseline`.
- LR: chỉ đổi 0.001 → 0.01. Chỉ ra loss/accuracy có ổn định hay dao động qua từng epoch.
- Batch: chỉ đổi 32 → 64. Giải thích số optimizer steps/epoch giảm 1.407 → 704, dù cùng số ảnh và epoch.
- Fine-tuning: mở `layer4` và `fc`, giữ các lớp trước frozen/eval. Chỉ ra tên tham số trainable trong `parameters.json`.
- Giải thích vì sao fine-tuning không dùng cache feature của toàn bộ backbone và vì sao tốn thời gian hơn.
- So sánh validation tốt nhất, best epoch và đường học; không lấy test để chọn thêm LR.
- Nêu giới hạn một seed/5 epoch, không khẳng định chênh lệch nhỏ có ý nghĩa thống kê.

## 3. Những mục phải hoàn tất trước khi nộp

- Bổ sung họ tên, MSSV, lớp, giảng viên, ngày nộp theo mẫu riêng của lớp. Chỉnh bản Markdown/HTML hoặc chuyển nội dung sang mẫu báo cáo của lớp; không sửa số liệu thủ công.
- Xem TensorBoard thật: từ thư mục gốc chạy `python -m tensorboard.main --logdir results`, hoặc dùng ô TensorBoard trong notebook.
- Nếu lớp yêu cầu ảnh màn hình TensorBoard, chụp đủ `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy`, để thấy tên các run và trục epoch; ghi chú smoothing nếu bật. Event logs đã có trong gói.
- Nếu lớp yêu cầu notebook có output Colab, chạy các ô xem kết quả/kiến trúc/TensorBoard và lưu notebook. Các kết quả trong gói được chạy thật bằng CPU, không ghi là đã train trên Colab T4.
- Kiểm tra định dạng/hạn nộp riêng của giảng viên vì slide không quy định các mục đó.

## 4. Đưa vào GitHub để review

Từ thư mục repo đã clone trên máy, tạo nhánh mới trước khi chép bộ này vào một thư mục riêng, ví dụ `submission_complete_96/`. Không chép đè trực tiếp kết quả cũ của nhóm.

```bash
git switch main
git pull --ff-only origin main
git switch -c feat/practice2-complete-96
```

Chép nội dung thư mục `practice2_complete` đã giải nén vào `submission_complete_96/`. Không đưa `data/`, `cache/`, checkpoint `.pt` hoặc môi trường `.venv` lên GitHub. Có thể giữ checkpoint trong bản ZIP/Drive để tái dùng; `.gitignore` trong gói loại checkpoint khỏi commit.

```bash
git status
git add submission_complete_96
git diff --cached --stat
git commit -m "Complete Practice 2 experiments and report at 96px"
git push -u origin feat/practice2-complete-96
```

Mở Pull Request từ nhánh này vào `main`. Nếu người 2/3 bổ sung phân tích, mỗi bạn tạo nhánh và PR riêng; trưởng nhóm review rồi merge. Không tự merge vào `main`. Bảo vệ main bằng rule yêu cầu pull request và review; nếu nhóm muốn chỉ trưởng nhóm có quyền merge, cần cấu hình quyền/rules của repo tương ứng.

Trước khi commit phải xem `git status` và `git diff --cached --stat` để chắc không chọn nhầm dữ liệu hoặc thay đổi ngoài bài. Nếu repo đang có thay đổi chưa lưu, xử lý chúng trước khi đổi nhánh.
