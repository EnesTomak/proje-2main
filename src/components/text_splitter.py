"""
Metin Bölücü (Text Splitter) Modülü.

Bu modül, LangChain'in 'Document' nesnelerini alır ve onları
vektör veritabanına yüklenmeye uygun, daha küçük 'parçalara' (chunks)
böler.

Bu, 'document_processor' (sayfaları üretir) ve
'vectorstore_manager' (parçaları saklar) arasındaki köprüdür.
"""

import logging
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Merkezi loglama ve ayarlar
from src.utils.logging_config import setup_logging
from src.core.config import settings
setup_logging()
logger = logging.getLogger(__name__)

def chunk_documents(pages: List[Document]) -> List[Document]:
    """
    Bir liste 'Document' (sayfa) alır ve onları daha küçük 'chunk' (parça)
    Document'larına böler.

    Meta veriler (sayfa, kaynak, bölüm) otomatik olarak korunur
    ve yeni parçalara kopyalanır.

    Argümanlar:
        pages (List[Document]): 'document_processor'dan gelen sayfa listesi.

    Döndürür:
        List[Document]: Vektör veritabanına hazır, parçalanmış döküman listesi.
    """
    if not pages:
        logger.warning("Parçalanacak (chunk) sayfa bulunamadı.")
        return []

    # RecursiveCharacterTextSplitter, anlamsal olarak metni bölmek için
    # en iyi ve en esnek yöntemdir.
    # ('\n\n' -> '\n' -> '. ' -> ' ' sıralamasıyla bölmeyi dener)
    text_splitter = RecursiveCharacterTextSplitter(
        # chunk_size: Her bir parçanın maksimum karakter sayısı.
        # Bu, embedding modelinin (örn. text-embedding-004) bağlam
        # penceresinden (context window) daha küçük olmalıdır.
        chunk_size=1500,
        
        # chunk_overlap: Parçalar arasında anlamsal bütünlüğün
        # kaybolmaması için bırakılan ortak karakter sayısı.
        chunk_overlap=250,
        
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""] # Bölme öncelik sırası
    )
    
    logger.info(f"{len(pages)} adet sayfa, daha küçük parçalara (chunks) bölünüyor...")
    
    # 'split_documents' metodu, 'pages' listesindeki her bir Document'ı
    # alır, metin içeriğini böler ve meta verileri yeni oluşturulan
    # parçalara (chunks) kopyalar.
    chunked_docs = text_splitter.split_documents(pages)
    
    logger.info(f"{len(pages)} sayfa, toplam {len(chunked_docs)} adet parçaya (chunk) bölündü.")
    
    return chunked_docs
