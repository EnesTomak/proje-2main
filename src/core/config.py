"""
Merkezi Yapılandırma Modülü.

Bu modül, .env dosyasındaki tüm ortam değişkenlerini okur ve
tüm uygulama genelinde (Streamlit, Celery) kullanılmak üzere
tek bir 'settings' objesi olarak dışa aktarır.

Bu, API anahtarlarının ve ayarların kodun içine dağılmasını engeller.
"""

import os
from dotenv import load_dotenv

# .env dosyasını yükle (eğer varsa)
# Bu, özellikle Docker dışında yerel geliştirme yaparken kullanışlıdır.
# Docker Compose, değişkenleri 'environment' bölümünden zaten sağlar.
load_dotenv()

class Settings:
    """
    Tüm proje ayarlarını tutan Pydantic-benzeri bir sınıf.
    .env dosyasından veya ortam değişkenlerinden değerleri yükler.
    """
    def __init__(self):
        # --- API Anahtarları ---
        self.GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
        if not self.GOOGLE_API_KEY:
            # Kritik hata: API anahtarı olmadan sistem çalışamaz.
            raise ValueError("HATA: GOOGLE_API_KEY ortam değişkeni bulunamadı. Lütfen .env dosyanızı kontrol edin.")

        # --- Celery & Redis Yapılandırması ---
        self.CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
        self.CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

        # YENİ (Faz 13 - Hibrit): MLflow deney (RAGAS) kayıt yolu
        # 'docker-compose.yml'deki 'file://' yoluyla eşleşir
        self.MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlflow/mlflow-db")

        # --- Kalıcı Depolama Yolları (Container İçi) ---
        self.DB_PERSIST_DIR: str = os.getenv("DB_PERSIST_DIR", "/app/chroma_db_local")
        self.PENDING_DIR: str = "/app/pending_files"
        self.PROCESSED_DIR: str = "/app/processed_files"
        self.FAILED_DIR: str = "/app/failed_files"
        
        # --- RAG Zinciri Ayarları ---
        self.EMBEDDING_MODEL: str = "models/text-embedding-004"
        self.LLM_MODEL: str = "gemini-2.5-flash-lite"
        self.RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.BASE_RETRIEVER_K: int = 25 # Re-ranker'a gönderilecek belge sayısı
        self.RERANKER_TOP_N: int = 5    # Re-ranker'dan sonra LLM'e gönderilecek belge sayısı

# Ayarları başlat ve tüm projede kullanmak üzere dışa aktar
# Diğer dosyalardan kullanımı: from src.core.config import settings
settings = Settings()
