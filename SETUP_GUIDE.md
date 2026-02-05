

## Chuẩn bị dữ liệu & model (bắt buộc)

Do repo đã cấu hình `.gitignore` để **không commit** các thư mục lớn `backend/data/` và `backend/model/`, bạn cần tải chúng thủ công trước khi chạy hệ thống.

- Link Google Drive: https://drive.google.com/drive/u/0/folders/13l2pWqHqDA5zh8Zk_2Lfy4Z4ONfj-OWm

### Cách làm

1. Mở link Drive ở trên và tải về các thư mục (hoặc file) tương ứng.
2. Giải nén (nếu có) và đặt đúng vào project theo cấu trúc:

```
backend/
	data/
		vn_plagiarism_corpus.json
		vn_plagiarism_queries.json
		corpus_chunks.pkl
		chunk_embeddings_normalized.npy
		chunk_faiss_index.faiss
		chunk_metadata.pkl
		combiner_logreg.joblib
		...
	model/
		detector_phobert/
			config.json
			model.safetensors
			tokenizer_config.json
			vocab.txt
			...
		phobert_finetuned/
			model.safetensors
			...
```

3. Kiểm tra nhanh (Windows): chạy `start.bat` sẽ tự báo thiếu file nếu chưa đủ.

> Lưu ý: Nếu bạn đặt model PhoBERT paraphrase ở nơi khác, có thể set biến môi trường `PHOBERT_PARAPHRASE_MODEL_DIR` để trỏ tới thư mục đó.

### Cách 1: Sử dụng Script Tự Động (Khuyến Nghị)

**Windows:**
```bash
# Mở Git Bash hoặc Command Prompt
start.bat
```
Script sẽ tự động:
- Kiểm tra các file model
- Khởi động 3 backend services
- Khởi động React frontend

### Cách 2: Chạy Thủ Công (Development)

Mở **4 terminal** riêng biệt:

**Terminal 1 - Plagiarism API:**
```bash
# Kích hoạt virtual environment (nếu dùng)
source .venv/Scripts/activate  # Windows Git Bash
# source .venv/bin/activate    # Linux/Mac

# Chạy API
cd backend/api
python plagiarism_api.py
```
Server chạy tại: `http://localhost:5000`

**Terminal 2 - AI Detection API:**
```bash
# Kích hoạt virtual environment (nếu dùng)
source .venv/Scripts/activate  # Windows Git Bash

# Chạy API
cd backend/api
python ai_detection_api.py
```
Server chạy tại: `http://localhost:5002`

**Terminal 3 - Node.js Auth API:**
```bash
cd backend/node-auth
npm start
# Hoặc dùng nodemon cho development:
# npm run dev
```
Server chạy tại: `http://localhost:5001`

**Terminal 4 - React Frontend:**
```bash
cd frontend
npm start
```
Server chạy tại: `http://localhost:3000`

### Kiểm Tra Services

Sau khi khởi động, kiểm tra các endpoints:

```bash
# Plagiarism API
curl http://localhost:5000/health

# AI Detection API
curl http://localhost:5002/health

# Node.js Auth API
curl http://localhost:5001/health
```
