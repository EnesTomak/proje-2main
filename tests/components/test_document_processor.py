"""
Birim Testleri (Unit Tests) - src.components.document_processor
(Yeniden yapılandırılmış konum: tests/components/)

Bu test betiği, 'pytest' tarafından çalıştırılmak üzere tasarlanmıştır.
'document_processor' modülümüzün temel işlevlerini doğrular.
"""

import pytest
import os
import sys
from typing import List
from langchain.docstore.document import Document

# Testlerin 'src' paketini bulabilmesi için proje kök dizinini 'sys.path'e ekle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

# Test edilecek modülü import et
try:
    from src.components.document_processor import extract_pages_from_pdf
    from src.utils.logging_config import setup_logging
    # Testler sırasında loglamayı da etkinleştir
    setup_logging()
except ImportError:
    print("HATA: 'src' modülleri import edilemedi.")
    print("Lütfen testi projenin kök dizininden 'pytest' komutuyla çalıştırın.")
    sys.exit(1)


# --- Test Fixture'ı (Test Verisi) ---

@pytest.fixture(scope="module")
def sample_pdf_path() -> str:
    """
    Test için kullanılacak örnek PDF dosyasının yolunu sağlar.
    Bu fixture, dosyanın var olup olmadığını kontrol eder.
    """
    # 'tests/fixtures/' klasörüne (Faz 9.1'de oluşturduk) yönlendir
    path = os.path.join(PROJECT_ROOT, "tests", "fixtures", "sample_article.pdf")
    
    if not os.path.exists(path):
        pytest.skip(f"Test verisi bulunamadı: {path}. Lütfen 'tests/fixtures/' klasörüne 'sample_article.pdf' ekleyin.")
    
    return path

# --- Test Fonksiyonları ---

def test_extract_pages_returns_list_of_documents(sample_pdf_path):
    """
    Test 1: Fonksiyonun doğru tipte (List[Document]) bir çıktı
           verdiğini ve boş olmadığını doğrular.
    """
    pages: List[Document] = extract_pages_from_pdf(sample_pdf_path)
    
    assert isinstance(pages, list)
    assert len(pages) > 0 
    assert isinstance(pages[0], Document)

def test_document_metadata_is_correct(sample_pdf_path):
    """
    Test 2: Çıkarılan belgelerin meta verilerinin (kaynak, sayfa no)
           doğru atandığını doğrular.
    """
    pages: List[Document] = extract_pages_from_pdf(sample_pdf_path)
    
    first_page = pages[0]
    assert "source" in first_page.metadata
    assert "page" in first_page.metadata
    
    assert first_page.metadata["source"] == "sample_article.pdf"
    assert first_page.metadata["page"] == 1
    
    if len(pages) > 1:
        last_page = pages[-1]
        assert last_page.metadata["page"] == len(pages)

def test_smart_section_detection_works(sample_pdf_path):
    """
    Test 3: (En Kritik Test) Akıllı bölüm (section) tespitinin
           çalıştığını doğrular.
    """
    pages: List[Document] = extract_pages_from_pdf(sample_pdf_path)
    
    detected_sections = set()
    for page in pages:
        section = page.metadata.get("section", "Unknown")
        detected_sections.add(section)
        
    print(f"Test PDF'inde bulunan bölümler: {detected_sections}")

    assert len(detected_sections) > 1 
    assert "Introduction" in detected_sections or "Abstract" in detected_sections
