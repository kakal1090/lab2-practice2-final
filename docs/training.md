# Huấn luyện và tái lập

## Thiết lập chung

Mỗi lượt 5 epoch, SGD momentum 0.9, weight_decay=0, CrossEntropyLoss trên logits (không thêm softmax trước loss). Các lượt khởi tạo từ pretrained ImageNet và cùng khởi tạo head theo seed 42; fine-tuning không lấy checkpoint baseline để tránh thêm ngân sách học. Chọn best checkpoint theo validation accuracy; nếu hòa giữ epoch xuất hiện đầu tiên.

## Cấu hình

| Run | Phần được học | LR | Batch |
|---|---|---:|---:|
| resnet_baseline | fc | 0.001 | 32 |
| densenet_baseline | classifier | 0.001 | 32 |
| resnet_lr_01 | fc | 0.01 | 32 |
| resnet_batch64 | fc | 0.001 | 64 |
| resnet_finetune | layer4 và fc | 0.001 | 32 |

`run_epoch` thực hiện forward, tính cross entropy trên logits, zero_grad/backward/step khi train. Validation không cập nhật gradient. Ghi train_loss, val_loss, train_accuracy, val_accuracy sau mỗi epoch. Best checkpoint theo validation accuracy; hòa giữ epoch đầu tiên.

Các lượt chỉ học head cache đặc trưng backbone cố định để tiết kiệm CPU. Fine-tuning không dùng cache toàn backbone vì trọng số thay đổi. Phải tính chi phí extraction khi so sánh thời gian.

## Chạy mới

Python >=3.11. Dùng thư mục làm việc riêng. Trước khi train mới, đổi tên `results/` có sẵn thành `results_reference/` để giữ kết quả gốc: chương trình có thể bỏ qua run có summary cùng cấu hình. Notebook phần 5 tự tạo thư mục mới nên không cần đổi tên trong Colab.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.download_data
python -u -m src.experiment --image-size 96 --epochs 5 --threads 4 --workers 0
python -m src.verify_results
python -m src.make_report
python -m src.make_pdf
```

Dataset và pretrained weights cần mạng. Môi trường tái lập có thể khác môi trường gốc; GPU có thể cho sai khác nhỏ. Xem [runtime_details.json](../results/runtime_details.json), [environment.json](../results/environment.json), [execution.log](../results/execution.log).
