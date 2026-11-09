# Đọc bài và chuẩn bị nộp

## Tài liệu chính

Đọc [mục lục docs](../docs/README.md). Dữ liệu, mô hình, huấn luyện và đánh giá có file riêng để đối chiếu cấu trúc cô yêu cầu.

## Cần hoàn tất

Điền thông tin nhóm trong [labs2-description.md](../docs/labs2-description.md) và bản báo cáo dùng nộp. Phân công tại [plan-progress.status.md](../docs/plan-progress.status.md) chỉ là đề xuất, cần xác nhận thực tế. Không thay số liệu thủ công. Nếu chạy lại công cụ sinh báo cáo, kiểm tra thông tin nhóm vì nguồn template có thể sinh lại chỗ trống.

## Trình bày

Dùng HTML báo cáo làm chính và notebook phần 1–4 để demo kiến trúc, TensorBoard. Notebook clone repo main; không upload ZIP. Kết quả gốc được ghi nhận train CPU. Đọc [demo.md](../docs/demo.md).

## Đưa lên repo hiện có

Làm theo [HUONG_DAN_PUSH_GITHUB.md](../HUONG_DAN_PUSH_GITHUB.md). Bổ sung docs ở root repo ngang src/report/results, không đặt cả bộ vào thư mục con khác. Notebook và code thực nghiệm không cần chép đè. Kiểm tra diff trước commit và mở PR theo quy trình nhóm.

## Phạm vi

Gói không có dataset/checkpoint, không có output Colab cá nhân. Nếu cô yêu cầu ảnh TensorBoard hoặc notebook output, lưu từ phiên đã chạy. Nếu cô yêu cầu inference ảnh mới, cần checkpoint đúng và phần demo tương ứng. Không cần train lại chỉ để có docs.
