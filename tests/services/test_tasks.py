"""
Birim Testleri - src.services.tasks (Celery Worker)

Bu test betiği, 'process_pdf_task' Celery görevinin
orkestrasyon (boru hattı) mantığını doğrular.

Kritik "senior" pratikler:
- 'monkeypatch' kullanarak, görevin çağırdığı *tüm* harici ve ağır
  işlemler (ocrmypdf, document_processor, vectorstore_manager)
  'mock' (taklit) edilir.
- Test, bu harici fonksiyonların çağrılıp çağrılmadığını ve
  dosyaların (pending -> processed / failed) doğru taşınıp taşınmadığını
  doğrular.
"""

import pytest
import sys
import os
import shutil
from unittest.mock import MagicMock, patch

# 'src' paketini testlerin bulabilmesi için
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

# Test edilecek modül
from src.services import tasks
# 'conftest.py'den paylaşılan fixture'ı import et
from tests.conftest import sample_document_list

# --- Test Fonksiyonu ---

# '@patch' dekoratörleri, test süresince belirtilen fonksiyonları
# otomatik olarak 'MagicMock' objeleriyle değiştirir.
@patch('src.services.tasks.add_documents_to_store')
@patch('src.services.tasks.chunk_documents')
@patch('src.services.tasks.extract_pages_from_pdf')
@patch('src.services.tasks.ocrmypdf.ocr')
@patch('src.services.tasks.os.remove')
@patch('src.services.tasks.shutil.move')
def test_process_pdf_task_success_path(
    mock_shutil_move: MagicMock,
    mock_os_remove: MagicMock,
    mock_ocrmypdf_ocr: MagicMock,
    mock_extract_pages: MagicMock,
    mock_chunk_docs: MagicMock,
    mock_add_to_store: MagicMock,
    monkeypatch,
    sample_document_list
):
    """
    Test 1: 'process_pdf_task' görevinin BAŞARILI bir senaryoda
           tüm adımları (OCR -> Extract -> Chunk -> Add)
           doğru sırada çağırdığını ve dosyaları doğru taşıdığını test eder.
    """
    
    # --- Hazırlık (Arrange) ---
    
    # 1. Görev bağlamını (self) taklit et (Celery 'bind=True' için)
    mock_task_self = MagicMock()
    mock_task_self.request.id = "test-task-id-123"

    # 2. Sahte (mock) fonksiyonların ne döndüreceğini ayarla
    mock_extract_pages.return_value = sample_document_list # 3 sahte sayfa
    mock_chunk_docs.return_value = sample_document_list * 2 # 6 sahte parça
    mock_add_to_store.return_value = 6 # 6 yeni belgenin eklendiğini simüle et

    # 3. Dosya yollarını tanımla
    test_filename = "test_doc.pdf"
    pending_path = f"/app/pending_files/{test_filename}"
    processed_path = f"/app/processed_files/{test_filename}"
    
    # 'settings' objesini de mock'layarak yolları kontrol altına al
    mock_settings = MagicMock()
    mock_settings.PROCESSED_DIR = "/app/processed_files"
    mock_settings.FAILED_DIR = "/app/failed_files"
    monkeypatch.setattr(tasks, 'settings', mock_settings)
    
    # --- Eylem (Act) ---
    # Test edilen fonksiyonu (görevi) çalıştır
    result = tasks.process_pdf_task(mock_task_self, pending_path)

    # --- Doğrulama (Assert) ---
    
    # 1. Tüm adımlar 1'er kez çağrıldı mı?
    mock_ocrmypdf_ocr.assert_called_once_with(
        pending_path, processed_path, force_ocr=True, deskew=True, language='eng+tur'
    )
    mock_extract_pages.assert_called_once_with(processed_path)
    mock_chunk_docs.assert_called_once_with(sample_document_list)
    mock_add_to_store.assert_called_once_with(sample_document_list * 2)

    # 2. Dosya operasyonları doğru mu?
    # Başarılı olduğu için 'pending' dosyası silinmeli
    mock_os_remove.assert_called_once_with(pending_path)
    # Başarılı olduğu için 'shutil.move' (hata) çağrılmamalı
    mock_shutil_move.assert_not_called()

    # 3. Görev sonucu (return value) doğru mu?
    assert result["status"] == "success"
    assert result["file"] == test_filename
    assert result["chunks_added"] == 6


@patch('src.services.tasks.add_documents_to_store')
@patch('src.services.tasks.chunk_documents')
@patch('src.services.tasks.extract_pages_from_pdf')
@patch('src.services.tasks.ocrmypdf.ocr')
@patch('src.services.tasks.os.remove')
@patch('src.services.tasks.shutil.move')
def test_process_pdf_task_failure_path(
    mock_shutil_move: MagicMock,
    mock_os_remove: MagicMock,
    mock_ocrmypdf_ocr: MagicMock,
    mock_extract_pages: MagicMock,
    mock_chunk_docs: MagicMock,
    mock_add_to_store: MagicMock,
    monkeypatch
):
    """
    Test 2: 'process_pdf_task' görevinin BAŞARISIZ bir senaryoda
           (örn. 'extract_pages' hata verdiğinde)
           dosyayı 'failed' klasörüne taşıdığını test eder.
    """
    
    # --- Hazırlık (Arrange) ---
    mock_task_self = MagicMock()
    mock_task_self.request.id = "test-task-id-456"
    
    # 2. Adımda (extract_pages) bir hata fırlatmayı simüle et
    mock_extract_pages.side_effect = ValueError("Bozuk PDF simülasyonu")
    
    test_filename = "failed_doc.pdf"
    pending_path = f"/app/pending_files/{test_filename}"
    processed_path = f"/app/processed_files/{test_filename}"
    failed_path = f"/app/failed_files/{test_filename}"

    mock_settings = MagicMock()
    mock_settings.PROCESSED_DIR = "/app/processed_files"
    mock_settings.FAILED_DIR = "/app/failed_files"
    monkeypatch.setattr(tasks, 'settings', mock_settings)
    
    # 'os.path.exists'in 'pending_path' için True döndürmesini sağla
    monkeypatch.setattr(tasks.os.path, 'exists', lambda path: path == pending_path)

    # --- Eylem (Act) & Doğrulama (Assert) ---
    # Görevin 'ValueError' hatası fırlatmasını bekliyoruz
    with pytest.raises(ValueError, match="Bozuk PDF simülasyonu"):
        tasks.process_pdf_task(mock_task_self, pending_path)

    # --- Doğrulama (Assert) ---
    
    # 1. Boru hattı erken durdu mu?
    mock_ocrmypdf_ocr.assert_called_once() # Adım 1 çalıştı
    mock_extract_pages.assert_called_once() # Adım 2 çalıştı (ve hata verdi)
    mock_chunk_docs.assert_not_called() # Adım 3 çalışmamalı
    mock_add_to_store.assert_not_called() # Adım 4 çalışmamalı

    # 2. Dosya operasyonları doğru mu?
    # Hata oluştuğu için 'os.remove' çağrılmamalı
    mock_os_remove.assert_not_called()
    # Hata oluştuğu için 'shutil.move' (hata) çağrılmalı
    mock_shutil_move.assert_called_once_with(pending_path, failed_path)
