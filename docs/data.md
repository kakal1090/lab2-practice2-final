# Dữ liệu CIFAR 10

## Nguồn và cấu trúc

Nguồn: https://www.cs.toronto.edu/~kriz/cifar.html . Dùng CIFAR-10 Python qua torchvision.

10 lớp theo thứ tự nhãn: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Mỗi lớp có 1.000 ảnh test.

CIFAR-10 có 50.000 ảnh train gốc và 10.000 ảnh test, mỗi ảnh RGB 32×32, 10 lớp. Từ train gốc, tách phân tầng 45.000 train và 5.000 validation (500/lớp) với seed 42. Dùng đầy đủ dữ liệu, không dùng tập con. Chỉ số split lưu trong results/split.json. Test tách biệt hoàn toàn, không chọn cấu hình bằng test.

Ảnh được resize 96×96, ToTensor và normalize mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225). Không augmentation. Kích thước 96 là lựa chọn thực nghiệm phù hợp CPU; đề không bắt buộc 224. Cùng preprocessing cho cả năm lượt. Không trộn kết quả với thí nghiệm cũ 224×224 trong repo nhóm.



## Tải và kiểm tra

Gói này không chứa dataset. Từ thư mục gốc, sau khi cài requirements:

```bash
python -m src.download_data
```

Nếu đã có archive, đặt `cifar-10-python.tar.gz` vào `data/` rồi chạy lệnh trên. Archive gốc được ghi nhận có 170.498.071 byte, MD5 `c58f30108f718f92721af3b95e74349a`. Xem [data_verification.json](../results/data_verification.json).

Chia tập nằm ở `split_indices`, tiền xử lý tại `loaders_data` trong [experiment.py](../src/experiment.py). Chỉ số tập nằm trong [split.json](../results/split.json). Không dùng test để chọn tham số hoặc epoch.
