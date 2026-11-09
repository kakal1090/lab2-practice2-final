# Demo và thuyết trình Lab 2

## Mở file nào

- Trình bày chính: [BAO_CAO_PRACTICE_2.html](../report/BAO_CAO_PRACTICE_2.html), mở bằng Chrome sau khi giải nén. Dùng bản PDF làm dự phòng.
- Chạy demo: [PRACTICE_2_HOAN_CHINH.ipynb](../PRACTICE_2_HOAN_CHINH.ipynb) trên Colab.
- Giải thích code: [experiment.py](../src/experiment.py).

## Cách chạy từ GitHub

Notebook giữ nguyên cách lấy bài từ repo, không yêu cầu upload ZIP. Sau khi docs được merge vào main, mở:

https://colab.research.google.com/github/kakal1090/lab2-practice2-final/blob/main/PRACTICE_2_HOAN_CHINH.ipynb

1. Lưu bản sao vào Drive nếu cần giữ output cá nhân.
2. Chạy phần 1 để clone main; kiểm tra commit in ra để biết phiên bản. Phần này cần mạng và quyền đọc repo.
3. Chạy phần 2 để đọc bảng và báo cáo.
4. Chạy phần 3 để in kiến trúc và tham số được học.
5. Chạy phần 4 để mở TensorBoard; chọn các run, xem bốn scalar, smoothing 0 và đủ 5 epoch.
6. Giữ `RUN_NEW_EXPERIMENTS = False` ở phần 5 khi chỉ demo kết quả.

Nếu chỉ upload notebook vào Colab nhưng chưa merge docs lên main, phần 1 vẫn lấy phiên bản main cũ. File ZIP mới không tự cập nhật repo.

## Kết quả mong đợi

ResNet baseline test 82,88%; DenseNet baseline 84,61%; ResNet fine-tuning 90,05%. Fine-tuning best validation 91,06%, epoch 4. Trainable: ResNet head 5.130, DenseNet head 10.250, fine-tuning 8.398.858.

Giới thiệu: “Đây là kết quả đã huấn luyện trước bằng CPU. Nhóm đang chạy notebook để đọc số liệu, xem kiến trúc và TensorBoard từ log đã lưu.”

## Khi cô hỏi code

`make_model`: pretrained, freeze, thay head. `split_indices` và `loaders_data`: chia tập và transforms. `training_mode`: train/eval. `run_epoch`: vòng train/validation. `extract`: cache feature.

## Tái lập tùy chọn

Đọc [training.md](training.md). Chạy lại phần 1 nếu đổi runtime, bật `RUN_NEW_EXPERIMENTS=True`, chạy các ô phần 5 theo thứ tự. Dataset và weights sẽ tải khi cần. Khi hoàn tất, tải kết quả ở phần 6. Muốn xem kết quả mới bằng phần 2–4, chạy `project = new_project` sau khi train xong rồi chạy lại các phần đó.

## Phân công thuyết trình đề xuất

- Người 1: mục tiêu, dữ liệu, ResNet18, loss và optimizer.
- Người 2: DenseNet121, freeze, tham số và so sánh baseline.
- Người 3: tuning, fine-tuning, TensorBoard, confusion matrix và hạn chế.

Tổng khoảng 12–15 phút; nhóm tự xác nhận tên và đóng góp thực tế.

## Dự phòng

Nếu mất mạng, dùng HTML/PDF cục bộ và ảnh TensorBoard đã chuẩn bị. Nếu `project` chưa có, chạy phần 1 thành công trước. Notebook trong ZIP chưa có output từ phiên Colab cá nhân: tải bản đã chạy nếu lớp yêu cầu.

ZIP không có dataset hoặc checkpoint. Demo upload ảnh và suy luận mới chưa được triển khai; nếu cô yêu cầu, cần checkpoint đúng hoặc train lại. Không dùng ImageNet chưa fine-tuning để giả làm mô hình đạt 90,05%.
