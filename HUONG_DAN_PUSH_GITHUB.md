# Đưa bộ tài liệu lên GitHub

Repo dự kiến: https://github.com/kakal1090/lab2-practice2-final . Hướng dẫn này giữ lịch sử repo bằng cách clone repo có sẵn và tạo nhánh. Không chạy git init trong thư mục ZIP.

## Chuẩn bị

Cài Git nếu chưa có. Giải nén `Lab2_GitHub_Ready.zip`. Điền thông tin trong docs và báo cáo theo thực tế. Chưa cần chỉnh notebook hoặc chạy train lại.

## Clone vào thư mục riêng

Mở Git Bash ở nơi muốn lưu repo, ví dụ ổ D:

```bash
cd /d
git clone https://github.com/kakal1090/lab2-practice2-final.git lab2-docs-update
cd lab2-docs-update
git switch -c docs/lab2-structure-update
```

Nếu thư mục lab2-docs-update đã tồn tại, dùng tên thư mục mới hoặc dùng repo đã clone và kiểm tra git status trước. Nếu nhánh đã có, đổi sang tên nhánh mới.

## Chép các mục đã thay đổi

Dùng File Explorer, chép từ bên TRONG thư mục `Lab2_GitHub_Ready` sang bên TRONG `D:\lab2-docs-update`:

- Thư mục docs
- README.md
- BAT_DAU_O_DAY.txt
- HUONG_DAN_PUSH_GITHUB.md
- KIEM_TRA_BO_SUNG.md
- report/HUONG_DAN_DOC_VA_NOP.md
- report/VALIDATION.md

Đặt hai file report vào thư mục report đã có. Cho phép thay thế đúng các file này. Không chép nguyên thư mục Lab2_GitHub_Ready thành thư mục con trong repo. Code, metrics, notebook và báo cáo chính trong ZIP này không thay đổi so với ZIP nguồn; không cần chép đè chúng lên phiên bản main mới hơn.

Nếu bạn tự điền thông tin vào báo cáo .md/.html/.pdf, chép thêm đúng những file đã chỉnh và thêm chúng vào git add ở bước sau.

## Kiểm tra và commit

```bash
git status
git diff --stat
git add docs README.md BAT_DAU_O_DAY.txt HUONG_DAN_PUSH_GITHUB.md KIEM_TRA_BO_SUNG.md report/HUONG_DAN_DOC_VA_NOP.md report/VALIDATION.md
git diff --cached --stat
git diff --cached
```

Nhấn q nếu màn hình diff đang ở chế độ xem cuộn. Kiểm tra chỉ có thay đổi dự kiến rồi chạy:

```bash
git commit -m "Add Lab 2 documentation structure and fix instructions"
git push -u origin docs/lab2-structure-update
```

Nếu Git yêu cầu user.name/user.email, đặt danh tính thật cho repo rồi commit lại. Nếu cần đăng nhập, hoàn tất luồng xác thực GitHub trên máy. Nếu lỗi 403, kiểm tra tài khoản có quyền push hoặc dùng fork/nhờ chủ repo cấp quyền. Không dùng force push để xử lý lỗi này.

## Tạo và merge pull request

Mở repo trên GitHub, chọn Compare & pull request. Kiểm tra base main và compare docs/lab2-structure-update. Tạo PR, xem Files changed. Người có quyền kiểm tra và merge theo quy định nhóm. Nếu repo có bảo vệ nhánh, hoàn thành review/checks cần thiết.

Push nhánh xong chưa có nghĩa main đã cập nhật. Link Colab hiện đọc main; chỉ sau khi PR được merge, mở lại notebook và chạy phần 1 để clone phiên bản mới. Kiểm tra docs có trên main và commit được in trong Colab.

Nguồn hướng dẫn GitHub: https://docs.github.com/en/get-started/using-git/pushing-commits-to-a-remote-repository và https://docs.github.com/en/pull-requests/get-started/pull-request-quickstart .
