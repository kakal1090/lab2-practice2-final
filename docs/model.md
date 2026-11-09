# Kiến trúc và transfer learning

ResNet18 dùng residual connections. Thay fc 512→1000 bằng Linear(512,10); mode head chỉ học fc.weight/fc.bias, tổng 5,130 tham số trên 11,181,642 tham số mô hình. DenseNet121 nối đặc trưng qua dense connections; classifier mới Linear(1024,10), học 10,250/6,964,106 tham số.

Fine-tuning ResNet18 mở layer4 và fc, tổng 8,398,858 tham số. Các lớp trước frozen và eval; layer4 và fc ở train. Với mode head, toàn bộ backbone ở eval để không đổi BatchNorm running statistics.

Các lượt head dùng cache đặc trưng để tiết kiệm CPU. Backbone frozen, eval và preprocessing xác định nên cùng ảnh luôn có cùng feature; học head trên cache tương đương học head trên output backbone cố định, trong sai số tính toán. Đã kiểm tra forward cached và full model khớp. Không cache backbone cho fine-tuning vì trọng số thay đổi. Đây là tối ưu tính toán, không bổ sung dữ liệu hay thay đổi bài toán.

## Vị trí triển khai

Trong [experiment.py](../src/experiment.py): `make_model` nạp pretrained, đóng băng và thay head; `training_mode` quản lý train/eval; `features` và `extract` phục vụ cache.

Mỗi run lưu `architecture.txt` và `parameters.json` trong `results/<run>/`. `requires_grad=False` ngăn cập nhật tham số; `eval()` giữ hành vi đánh giá của backbone, bao gồm thống kê BatchNorm. Hai thao tác có mục đích khác nhau.
