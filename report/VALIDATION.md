# Kiểm tra đã thực hiện

## Thực nghiệm thật

- Chạy đủ 5 cấu hình × 5 epoch bằng CPU, full CIFAR-10; 45.000 train, 5.000 validation; test riêng 10.000 ảnh cho mỗi mô hình cuối.
- ResNet18 baseline: best validation 84,24%, test 82,88%.
- DenseNet121 baseline: best validation 85,94%, test 84,61%.
- Fine-tuning ResNet18: best validation 91,06% ở epoch 4, test 90,05%; macro F1 0,9008.
- Lựa chọn cấu hình được ghi trong `selection_before_test.json` trước khi đánh giá test. Hai lượt LR/batch không có test vì không được chọn; ô trống không phải accuracy 0.
- `execution.log` là log chạy thật. `environment.json` lưu hash SHA-256 code đã thực thi; `runtime_details.json` lưu phiên bản package và lệnh thực tế.

## Dữ liệu

- Archive CIFAR-10 chính thức, 170.498.071 bytes, MD5 `c58f30108f718f92721af3b95e74349a`, khớp torchvision.
- `data_verification.json` lưu kiểm tra. Archive đầy đủ kèm trong ZIP; dữ liệu đã giải nén và cache feature không kèm để giảm dung lượng.
- Unit test kiểm tra split tái lập, không giao nhau, đúng 500 validation/lớp.

## Kiểm tra code

Lệnh đã chạy: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v`.

Ba test đã qua: split phân tầng; cached feature khớp full forward cho cả ResNet/DenseNet và frozen state không đổi; phạm vi train/eval và tham số fine-tuning đúng layer4+fc. Dữ liệu ngẫu nhiên chỉ dùng trong unit test, không dùng làm kết quả thực nghiệm.

Multiprocessing DataLoader không khả dụng trong môi trường thực thi này, nên lượt hoàn chỉnh dùng `--workers 0`. Đây là thay đổi cách nạp batch; không giảm dữ liệu hay epoch. Notebook dùng cùng thiết lập để tránh lỗi này.

## Kiểm tra kết quả

`python -m src.verify_results`: PASS cho cả 5 run.

- CSV đủ 5 epoch, giá trị hữu hạn; accuracy trong [0,1].
- Best epoch/accuracy khớp lịch sử validation.
- Bốn scalar TensorBoard khớp từng epoch của CSV (sai số float32 dưới 1e-5).
- Test đủ 10.000 dòng dự đoán/model; confusion matrix 10×10, mỗi lớp đúng 1.000 ảnh; tổng đường chéo khớp accuracy.
- Cấu hình thắng đúng max validation; hash source khớp code đã chạy.

Bằng chứng máy đọc: `results/verification.json`.

## Kiểm tra notebook và báo cáo

- Notebook đạt nbformat schema; mọi code cell parse được bằng AST. Notebook chưa được chạy bằng giao diện Colab trong lượt này; số liệu có sẵn được chạy bằng CPU như mô tả.
- HTML có đủ 4 bảng (yêu cầu, kết quả, thời gian, per-class), đúng số hàng; 8 hình được nhúng trực tiếp, không phụ thuộc file ảnh ngoài.
- PDF 9 trang A4 đã render kiểm tra bố cục, font tiếng Việt, bảng, trục/nhãn đồ thị, confusion matrices, ngắt trang và footer.
- Không thay đổi thủ công accuracy/loss trong báo cáo. `make_report` đọc CSV/JSON; `make_pdf` đọc bản Markdown vừa sinh.

## Phạm vi nộp

Bộ này dùng 96×96, độc lập với thực nghiệm 224×224 cũ của nhóm. Một seed/5 epoch là giới hạn đã nêu trong báo cáo. Điền thông tin nhóm và kiểm tra mẫu/hạn nộp riêng của giảng viên trước khi nộp.
