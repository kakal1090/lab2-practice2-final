# Mô tả Lab 2

## Thông tin nhóm cần hoàn tất

- Thành viên 1: [Họ tên và MSSV]
- Thành viên 2: [Họ tên và MSSV]
- Thành viên 3: [Họ tên và MSSV]
- Lớp: [Điền]
- Giảng viên: [Điền]
- Ngày nộp: [Điền]

## Mục tiêu

Sử dụng pretrained ImageNet cho phân loại CIFAR-10; thay classifier thành 10 lớp; so sánh ResNet18 và DenseNet121; thử learning rate, batch size và fine-tuning; theo dõi TensorBoard và đánh giá trên tập test tách biệt.

## Phạm vi

5 cấu hình, mỗi cấu hình 5 epoch; ảnh 96×96, seed 42; 45.000 train, 5.000 validation, 10.000 test. Kết quả tham chiếu được ghi nhận chạy bằng CPU. Không trộn với thực nghiệm 224×224 khác. Các tuyên bố về đề trong đối chiếu dưới đây kế thừa báo cáo gốc; không thay thế việc đọc đề chính thức.

## Đối chiếu yêu cầu trong báo cáo gốc

| Yêu cầu slide 23 | Bằng chứng trong bài |
|---|---|
| Cài PyTorch/torchvision | requirements.txt, results/environment.json |
| Nạp pretrained | ResNet18 và DenseNet121, IMAGENET1K_V1 |
| Xem kiến trúc và tham số | results/[run]/architecture.txt, parameters.json; notebook in mô hình |
| Thay lớp cuối và freeze/fine-tune | fc/classifier 10 lớp; head hoặc layer4+fc |
| transforms và DataLoader | src/experiment.py: loaders_data, DataLoader ảnh và feature batches |
| Loss/optimizer/training loop | CrossEntropyLoss, SGD momentum 0.9, run_epoch |
| Đánh giá test | test_metrics.json, confusion matrix, classification report |
| Thử nhiều kiến trúc | resnet_baseline / densenet_baseline |
| Đổi hyperparameters | resnet_lr_01 / resnet_batch64 |
| TensorBoard | results/[run]/tensorboard, bốn scalar theo epoch |

## Tệp truy cập nhanh

- [Notebook](../PRACTICE_2_HOAN_CHINH.ipynb)
- [Code](../src/experiment.py)
- [Báo cáo HTML](../report/BAO_CAO_PRACTICE_2.html)
- [Bảng số liệu](../results/comparison.csv)
