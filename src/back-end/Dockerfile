FROM python:3.11-slim

WORKDIR /app

# Cài đặt các gói phụ thuộc
COPY requirements_backend.txt .
RUN pip install --no-cache-dir -r requirements_backend.txt

# Sao chép mã nguồn
COPY rag_backend.py .
COPY .env .

# Cài đặt các thư viện bổ sung cho sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Expose port
EXPOSE 30002

# Khởi động ứng dụng
CMD ["uvicorn", "rag_backend:app", "--host", "0.0.0.0", "--port", "30002"] 