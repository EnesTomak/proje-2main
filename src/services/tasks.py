"""
Asenkron Görev Yöneticisi (Celery Worker) (v2.5 Veri Kalitesi Kontrollü).

Bu modül, 'web' (Streamlit) servisinden bağımsız olarak çalışan
'worker' servisi tarafından kullanılır.

'process_pdf_task' görevi, ağır ve uzun süren PDF işleme
boru hattını (OCR -> Veri Kalitesi -> İşleme -> Parçalama -> Gömme) yürütür.
"""

import os
import shutil
import logging
import ocrmypdf
from celery import Celery
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Merkezi ayarlarımızı ve loglama yapılandırmamızı import ediyoruz
from src.core.config import settings
from src.utils.logging_config import setup_logging

# 'Bileşenleri' (tools) import ediyoruz
from src.components.document_processor import extract_pages_from_pdf
from src.components.text_splitter import chunk_documents
from src.components.vectorstore_manager import add_documents_to_store

# Loglamayı başlat
setup_logging()
logger = logging.getLogger(__name__)

# --- Dil Tespiti (langdetect) Yapılandırması ---
# 'DetectorFactory.seed = 0', langdetect'in deterministik (tutarlı)
# sonuçlar vermesini sağlar. Bu, test edilebilirlik için 'senior' bir pratiktir.
try:
    DetectorFactory.seed = 0
except Exception as e:
    logger.warning(f"langdetect seed ayarlanamadı: {e}")


# Celery uygulamasını başlat
celery = Celery(
    'tasks',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)
celery.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True
)

@celery.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
    task_soft_time_limit=900 # 15 dakika
)
def process_pdf_task(self, pending_filepath: str):
    """
    (ANA GÖREV) Bir PDF dosyası üzerinde tüm RAG işleme boru hattını çalıştırır.
    """
    
    filename = os.path.basename(pending_filepath)
    processed_pdf_path = os.path.join(settings.PROCESSED_DIR, filename)
    failed_pdf_path = os.path.join(settings.FAILED_DIR, filename)

    # 'finally' bloğunda kullanılmak üzere 'current_step' tanımla
    current_step = "ADIM 1: OCR" 
    
    try:
        logger.info(f"[TASK START: {self.request.id}] '{filename}' işleniyor...")

        # --- ADIM 1: OCR (Ağır I/O) ---
        logger.info(f"Adım 1/5: OCR (Tesseract) uygulanıyor -> {processed_pdf_path}")
        try:
            ocrmypdf.ocr(
                pending_filepath,
                processed_pdf_path,
                force_ocr=True,
                deskew=True,
                language='eng+tur' # İngilizce ve Türkçe dillerini tanı
            )
            logger.info(f"OCR Tamamlandı: {filename}")
        except ocrmypdf.exceptions.EncryptedPdfError:
            logger.warning(f"'{filename}' şifreli (parola korumalı) ve atlanıyor.")
            shutil.move(pending_filepath, failed_pdf_path)
            return {"status": "failed", "reason": "encrypted"}
        except ocrmypdf.exceptions.InputFileError as ocr_error:
            # Bozuk veya geçersiz PDF'ler burada yakalanır
            logger.warning(f"Bozuk PDF Hatası (InputFileError): {filename} - {ocr_error}.")
            shutil.move(pending_filepath, failed_pdf_path)
            return {"status": "failed", "reason": "corrupted_pdf"}
        except Exception as ocr_error:
            logger.warning(f"Genel OCR Hatası: {filename} - {ocr_error}. Orijinal dosya ile devam ediliyor.")
            shutil.copy(pending_filepath, processed_pdf_path)

        # --- ADIM 2: Akıllı Metin Çıkarma (CPU Yüklü) ---
        current_step = "ADIM 2: Metin Çıkarma (PyMuPDF)"
        logger.info(f"Adım 2/5: Akıllı metin/meta veri çıkarma (PyMuPDF) başlıyor...")
        pages = extract_pages_from_pdf(processed_pdf_path)

        # --- YENİ ADIM 2.5: Veri Kalitesi Kontrolü (Fail-Fast) ---
        current_step = "ADIM 2.5: Veri Kalitesi Kontrolü (langdetect)"
        logger.info(f"Adım 2.5/5: Veri kalitesi (Dil/İçerik) kontrol ediliyor...")
        
        if not pages:
            # 'document_processor' hiç sayfa döndürmezse (örn. boş PDF)
            raise ValueError(f"'{filename}' dosyasından hiçbir metin/sayfa çıkarılamadı (Dosya boş veya bozuk).")

        # Dil tespiti için ilk 2 sayfadan bir örneklem al
        # (Daha sağlam bir tespit için 1 sayfadan fazlasını kullanmak iyidir)
        sample_text = " ".join([p.page_content for p in pages[:2]])

        if len(sample_text) < 100:
            # Eğer metin çok kısaysa (örn. 100 karakterden az),
            # dil tespiti güvenilir olmaz ve muhtemelen PDF boştur.
            raise ValueError(f"'{filename}' dosyasındaki metin (100 karakterden az) dil tespiti için çok kısa.")

        try:
            lang = detect(sample_text)
            logger.info(f"'{filename}' için tespit edilen dil: {lang}")
            
            # Sadece İngilizce ('en') ve Türkçe ('tr') makaleleri kabul et
            if lang not in ['en', 'tr']:
                raise ValueError(f"'{filename}' desteklenmeyen bir dilde ({lang}) tespit edildi. Sadece 'en' ve 'tr' destekleniyor.")
                
        except LangDetectException as e:
            # 'langdetect' bir dil bulamazsa (örn. sadece rakamlar varsa)
            logger.warning(f"Dil tespiti başarısız oldu: {filename} - Hata: {e}")
            raise ValueError(f"'{filename}' dosyasının dili tespit edilemedi (muhtemelen metin içermiyor).")
        
        # --- ADIM 3: Metin Parçalama (CPU Yüklü) ---
        current_step = "ADIM 3: Metin Parçalama"
        logger.info(f"Adım 3/5: {len(pages)} sayfa parçalara (chunks) bölünüyor...")
        chunks = chunk_documents(pages)
        if not chunks:
            raise ValueError(f"'{filename}' dosyasından hiçbir parça (chunk) oluşturulamadı.")

        # --- ADIM 4: Gömme ve Veritabanına Ekleme (Ağ & I/O Yüklü) ---
        current_step = "ADIM 4: Veritabanına Ekleme"
        logger.info(f"Adım 4/5: {len(chunks)} parça vektör veritabanına ekleniyor (Google API)...")
        new_docs_added = add_documents_to_store(chunks)
        
        # --- ADIM 5: Temizlik ---
        current_step = "ADIM 5: Temizlik"
        # İşlem tamamlandı, 'pending' klasöründeki orijinal dosyayı sil.
        os.remove(pending_filepath)

        logger.info(f"[TASK SUCCESS: {self.request.id}] '{filename}' tamamlandı. {new_docs_added} yeni parça eklendi.")
        return {"status": "success", "file": filename, "chunks_added": new_docs_added}

    except Exception as e:
        # Bu 'try...except' bloğu, Celery'nin yeniden denemeleri (retry)
        # başarısız olduktan sonra çalışır veya bizim fırlattığımız
        # 'ValueError' (örn. Veri Kalitesi) hatalarını yakalar.
        logger.critical(f"[TASK FAILED: {self.request.id}] '{filename}' {current_step} aşamasında kalıcı olarak başarısız oldu: {e}", exc_info=True)
        
        # Orijinal dosyayı (eğer hala oradaysa) 'failed' klasörüne taşı
        if os.path.exists(pending_filepath):
            shutil.move(pending_filepath, failed_pdf_path)
        # 'processed' klasöründe (OCR sonrası) bir kopyası oluştuysa onu da sil
        if os.path.exists(processed_pdf_path) and current_step != "ADIM 1: OCR":
             os.remove(processed_pdf_path)
        
        # Celery'ye görevin başarısız olduğunu bildir
        raise e

