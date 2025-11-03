"""
Akıllı Belge İşleme Modülü (PyMuPDF tabanlı).

Bu modül, ham PDF dosyalarını işlemekten sorumludur.
- PyMuPDF (fitz) kullanarak metin ve temel görüntü/tablo varlıklarını çıkarır.
- Font boyutu/stili analizi yaparak (get_dominant_font_size, is_bold)
  bilimsel makalelerdeki bölüm (section) başlıklarını akıllıca tespit eder.
- Sonuçları, LangChain Document formatında, zengin meta verilerle
  döndürür.

Bu modül, Faz 5'teki (tasks.py) Celery worker'ı tarafından çağrılacaktır.
"""

import fitz  # PyMuPDF
import re
import statistics
import logging
import os
from typing import List, Dict, Any, Tuple
from langchain.docstore.document import Document

# Merkezi loglama yapılandırmasını uygula
from src.utils.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# Bölüm başlıklarını yakalamak için derlenmiş Regex
# (Hem İngilizce hem Türkçe yaygın terimler)
SECTION_REGEX = re.compile(
    r"^(Abstract|Özet|Giriş|Introduction|Methods|Methodology|Yöntemler|Results|Bulgular|Sonuçlar|Discussion|Tartışma|Conclusion)",
    flags=re.IGNORECASE | re.MULTILINE
)

def _get_dominant_font_stats(page: fitz.Page) -> Tuple[float, float]:
    """
    Sayfadaki en yaygın (dominant) font boyutunu (medyan) ve 
    en sık kullanılan boyutu (mod) döndürür.
    Medyan -> normal paragraf metni
    Mod -> potansiyel başlıklar (eğer medyandan büyükse)
    """
    sizes = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:  # Sadece metin blokları
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sizes.append(span.get("size", 0.0))
    
    if not sizes:
        return 10.0, 10.0  # Varsayılan font boyutu (eğer sayfa boşsa)

    try:
        median_size = float(statistics.median(sizes))
        mode_size = float(statistics.mode(sizes))
    except statistics.StatisticsError:
        # 'mode' bulunamazsa medyeni kullan
        median_size = float(statistics.median(sizes))
        mode_size = median_size
        
    return median_size, mode_size

def _is_bold(span: Dict[str, Any]) -> bool:
    """Bir metin parçasının (span) kalın (bold) olup olmadığını heuristic olarak belirler."""
    flags = span.get("flags", 0)
    fontname = span.get("font", "").lower()
    # PyMuPDF flag'lerinde 2**4 (16) bold bayrağıdır
    return (flags & 16) != 0 or "bold" in fontname

def extract_pages_from_pdf(pdf_path: str) -> List[Document]:
    """
    Bir PDF dosyasını işler ve sayfa sayfa, zengin meta verilere sahip
    LangChain Document nesnelerinin bir listesini döndürür.
    
    Argümanlar:
        pdf_path (str): İşlenecek PDF dosyasının yolu.

    Döndürür:
        List[Document]: Her bir sayfası Document objesi olan bir liste.
                       Metadata şunları içerir: source, page, section.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"Dosya bulunamadı: {pdf_path}")
        raise FileNotFoundError(f"Dosya bulunamadı: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    processed_pages: List[Document] = []
    current_section = "Unknown"  # Başlangıçta bölüm bilinmiyor
    
    try:
        for page_num, page in enumerate(doc):
            dominant_size, mode_size = _get_dominant_font_stats(page)
            # Başlık boyutu genellikle normal paragraf metninden (medyan)
            # en az 1 puan büyüktür.
            title_font_size_threshold = dominant_size + 1.0

            page_text = ""
            contains_image = bool(page.get_images()) # Görüntü var mı?
            
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT).get("blocks", [])
            for block in blocks:
                if block.get("type") != 0:  # Sadece metin blokları
                    continue
                
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue

                        span_size = span.get("size", 0.0)
                        
                        # "Senior" Dokunuş: Başlık tespiti
                        # Eğer metin kalınsa VEYA normal metinden belirgin şekilde büyükse
                        if _is_bold(span) or span_size > title_font_size_threshold:
                            match = SECTION_REGEX.search(text)
                            if match:
                                # Yeni bir bölüm başlığı bulduk
                                current_section = match.group(0).capitalize()
                                logger.debug(f"Yeni bölüm '{current_section}' bulundu, Sayfa: {page_num + 1}")

                        page_text += text + " " # Spanları boşlukla birleştir
                    page_text += "\n" # Satırları yeni satırla birleştir
            
            # Basit tablo tespiti (heuristic): Çok fazla '|' veya '\t' var mı?
            contains_table = page_text.count("|") > 10 or page_text.count("\t") > 10

            metadata = {
                "source": os.path.basename(pdf_path),
                "page": page_num + 1,
                "section": current_section,
                "contains_image": contains_image,
                "contains_table": contains_table
            }
            
            processed_pages.append(
                Document(page_content=page_text, metadata=metadata)
            )
            
    except Exception as e:
        logger.error(f"PDF işlenirken hata: {pdf_path} (Sayfa: {page_num + 1}) - Hata: {e}", exc_info=True)
    finally:
        doc.close()
        
    logger.info(f"'{os.path.basename(pdf_path)}' dosyasından {len(processed_pages)} sayfa başarıyla işlendi.")
    return processed_pages
