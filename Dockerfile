# Faz 3 (v2 Planı): Hem 'web' hem de 'worker' servisleri için ortak temel imaj
# Bu dosya, tüm sistem ve Python bağımlılıklarını kurar.
FROM python:3.11-slim

# Proje kodu için çalışma dizini
WORKDIR /app

# Sistem bağımlılıklarını kur (OCRmyPDF ve PyMuPDF için kritik)
# 'tesseract-ocr': OCR motoru
# 'libgl1': OpenCV gibi kütüphaneler için (PyMuPDF'in bazı bağımlılıkları)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    liblept5 \
    libtesseract4 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt'yi kopyala ve kur
# Önce bağımlılıkları kurmak, Docker'ın katman önbelleğini (layer caching)
# daha verimli kullanmasını sağlar. Kod değiştiğinde tekrar 'pip install' yapmaz.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje kodunun tamamını (src, scripts vb.) kopyala
COPY . .

# Streamlit portunu dışarı aç
EXPOSE 8501

# Not: 'ENTRYPOINT' veya 'CMD' docker-compose.yml içinde 'command' ile tanımlanıyor.