# Plagiarism & AI-Generated Text Detection System

Hệ thống kiểm tra **đạo văn** và **phát hiện văn bản do AI tạo** cho tiếng Việt, gồm:

- **Frontend**: React UI
- **Backend**:
  - `backend/api`: Flask APIs (plagiarism + AI detection)
  - `backend/node-auth`: Node.js service (auth/history)

> Ghi chú: Repo này **không commit** các thư mục dữ liệu/model lớn (`backend/data/`, `backend/model/`). Bạn cần có sẵn chúng ở máy (hoặc tải từ nguồn riêng của bạn) để chạy đầy đủ.

## Chạy nhanh

### Windows
- Chạy script: `start.bat`

### Linux / macOS
- Chạy script: `./start.sh`

Hoặc xem hướng dẫn chi tiết tại [SETUP_GUIDE.md](SETUP_GUIDE.md).

## Endpoints (tham khảo)

- Plagiarism API: `http://localhost:5000/health`
- Node Auth API: `http://localhost:5001/health`
- AI Detection API: `http://localhost:5002/health`

## Cấu trúc thư mục

- `frontend/`: React app
- `backend/api/`: Flask APIs
- `backend/node-auth/`: Node.js auth service

## Biến môi trường (tuỳ chọn)

- `PHOBERT_PARAPHRASE_MODEL_DIR`: trỏ tới thư mục model PhoBERT paraphrase (nếu không dùng mặc định trong backend).
