> BẢN DÀNH CHO GITHUB: giữ code, notebook, báo cáo, metrics và TensorBoard logs nhỏ. Không kèm dataset/checkpoint. Bộ thực nghiệm 96×96 này nằm riêng, không ghi đè bài 224×224 trước đó.
>
> Muốn mở notebook theo chế độ xem kết quả, dùng ZIP đầy đủ đã nhận. Muốn tái lập bằng code từ GitHub, chạy trong bản sao/thư mục kết quả sạch theo hướng dẫn dưới đây. Dataset tải bằng `src.download_data`; các checkpoint sẽ được tạo khi train.

# Practice 2 — Bài hoàn chỉnh về pretrained neural networks

Bám Chapter 3, Practice 2 (slide 23), ví dụ CIFAR-10 (slide 24).

## Mở bài trước

- `report/BAO_CAO_PRACTICE_2.html`: báo cáo tự chứa ảnh, mở bằng trình duyệt.
- `report/BAO_CAO_PRACTICE_2.pdf`: bản PDF để đọc/nộp.
- `report/BAO_CAO_PRACTICE_2.md`: bản chỉnh sửa nội dung.
- `PRACTICE_2_HOAN_CHINH.ipynb`: notebook Colab để xem kết quả đã chạy hoặc tái lập toàn bộ.
- `results/comparison.csv`: bảng tổng hợp số liệu thật.
- `results/<run>/tensorboard/`: event logs thật, không phải chỉ ảnh minh họa.

## Phạm vi thực nghiệm

Dùng toàn bộ CIFAR-10: 45.000 train, 5.000 validation, 10.000 test. Pretrained ImageNet; ảnh resize 96×96; normalization ImageNet; seed 42; SGD momentum 0.9; 5 epoch mỗi lượt. Hai kiến trúc: ResNet18 và DenseNet121. Năm lượt: baseline mỗi kiến trúc, đổi LR, đổi batch, fine-tuning layer4+fc. Test hai baseline và cấu hình thắng validation sau khi khóa lựa chọn.

96×96 là lựa chọn của bài thực nghiệm độc lập này để chạy được bằng CPU. Đề không bắt buộc 224. KHÔNG trộn số liệu này với DenseNet121 224×224 đã chạy trước đó trong repo nhóm. Muốn gộp vào một bảng công bằng, phải chạy cùng pipeline/kích thước/split cho các model.

## Đáp ứng đề

1. Nạp pretrained từ torchvision.models (weights enum hiện hành).
2. In kiến trúc, lưu tổng tham số/trainable và tên tham số được học.
3. Thay classifier thành 10 lớp; baseline frozen backbone, fine-tuning layer4+fc.
4. transforms và DataLoader cho ảnh, chia dữ liệu không rò rỉ test.
5. CrossEntropyLoss, SGD, vòng lặp train/validation thật.
6. Thử hai kiến trúc và hai hyperparameter trong lượt tách biệt.
7. TensorBoard ghi bốn scalar theo epoch.
8. Test cuối có accuracy, per-class precision/recall/F1, confusion matrix và dự đoán từng ảnh.

Các lựa chọn số model/epoch/split là thiết kế thực nghiệm; đề không ghi định mức tối thiểu hoặc rubric điểm. Báo cáo nêu giới hạn một seed và ảnh 96.

## Tái lập

Python >=3.11. GPU dùng cặp torch/torchvision phù hợp CUDA theo hướng dẫn PyTorch; CPU cũng chạy được. Cài:

```bash
python -m pip install -r requirements.txt
python -m src.download_data
python -u -m src.experiment --image-size 96 --epochs 5 --threads 4 --workers 0
python -m src.verify_results
python -m src.make_report
python -m src.make_pdf
python -m tensorboard.main --logdir results
```

Chạy mới trong thư mục sạch để không trộn run/config. Notebook tạo thư mục tái lập mới. `experiment` bỏ qua run có summary chỉ khi config giống hệt; nếu run dang dở chưa có summary, nó train lại từ đầu (không phải resume optimizer). Không dùng file kết quả cũ với config khác.

Tải dữ liệu: archive từ nguồn CIFAR chính thức, Range từng phần và checksum MD5 toàn bộ. Nếu server tạm lỗi, chạy lại để lấy phần còn thiếu; không xóa các part đã tải. Nếu nguồn chậm/không hỗ trợ, upload file Python tar.gz đầy đủ, đặt `data/cifar-10-python.tar.gz`, rồi chạy `python -m src.download_data` để xác minh/giải nén. Không cần đưa dataset lên GitHub.

## Tối ưu tính toán có giải thích

Lượt chỉ học head dùng cache feature từ backbone frozen/eval và preprocessing cố định; đã kiểm tra forward khớp với full model. Fine-tuning không dùng cache vì trọng số thay đổi. Thời gian trong bảng tách feature extraction và optimization; không so sánh riêng optimization để tuyên bố tốc độ model.

## Kiểm tra

```bash
python -m unittest discover -s tests -v
```

Xem `report/VALIDATION.md` để biết kiểm tra và lượt thực nghiệm đã thực sự hoàn thành. Không dùng smoke/fake accuracy trong báo cáo.

## Nộp nhóm

Đọc và hiểu báo cáo, bổ sung tên/MSSV nếu cần. Mỗi người trình bày phần mình: dữ liệu/ResNet, DenseNet/so sánh, tuning/fine-tuning. Giữ nguyên nguồn số liệu, không điền kết quả cho thí nghiệm chưa chạy. Tạo nhánh mới/PR để review, không ghi đè kết quả cũ của nhóm.

## Phân biệt hai gói bài

Gói GitHub này không có dataset/checkpoint. Bản ZIP đầy đủ đã cung cấp riêng kèm `data/cifar-10-python.tar.gz` từ nguồn CIFAR chính thức, đã đối chiếu MD5. Đây là lý do gói tải tương đối lớn; Colab có thể dùng lại file này và không tải từ máy chủ chậm. Không commit dataset lên GitHub. Gói giữ checkpoint tốt nhất của cấu hình thắng validation; checkpoint các lượt còn lại được loại để giảm dung lượng, nhưng giữ nguyên toàn bộ metrics, predictions, kiến trúc, config và event logs. Chạy lại code sẽ tạo đủ checkpoint mới.

Bắt đầu theo `report/HUONG_DAN_DOC_VA_NOP.md`. Báo cáo mô tả môi trường chạy thật là CPU; notebook là công cụ đọc/tái lập, không khẳng định kết quả đã train trên Colab.
