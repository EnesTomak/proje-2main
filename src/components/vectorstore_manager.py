"""
Güvenli Vektör Veritabanı Yöneticisi (ChromaDB).

Bu modül, ChromaDB ile olan tüm etkileşimleri merkezileştirir.
Sorumlulukları:
- ChromaDB'yi ve Google Embedding modelini "lazy loading" (gecikmeli yükleme)
  ile yalnızca ihtiyaç duyulduğunda başlatmak.
- 'settings' modülünden yapılandırmaları okumak.
- (KRİTİK) 'hashlib' kullanarak içerik tabanlı bir imza (hash)
  oluşturarak veritabanına mükerrer (duplicate) belge eklenmesini
  engellemek (idempotency).
"""

import os
import logging
import hashlib
import json
from typing import List, Optional
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Merkezi ayarları ve loglamayı import et
from src.core.config import settings
from src.utils.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# --- Global (Lazy Loaded) Değişkenler ---
# Bu, uygulamanın her çağrıda modeli/veritabanını yeniden yüklemesini engeller.
_embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
_db: Optional[Chroma] = None
_hash_index: Optional[Dict[str, Dict]] = None

def _get_hash_index_path() -> str:
    """Hash index dosyasının yolunu merkezi olarak döndürür."""
    # Index dosyasını, veritabanı klasörünün *içinde* saklıyoruz.
    return os.path.join(settings.DB_PERSIST_DIR, "doc_hash_index.json")

def _load_hash_index() -> Dict[str, Dict]:
    """
    Disk'ten belge imzalarını (hash) içeren JSON index'i yükler.
    Bu index, mükerrer kayıtları takip etmek için kullanılır.
    """
    global _hash_index
    if _hash_index is not None:
        return _hash_index

    index_path = _get_hash_index_path()
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                _hash_index = json.load(f)
            logger.info(f"{len(_hash_index)} adetlik belge imza (hash) index'i diskten yüklendi.")
        except json.JSONDecodeError:
            logger.warning(f"Hash index dosyası ({index_path}) bozuk. Yeni index oluşturuluyor.")
            _hash_index = {}
    else:
        logger.info("Hash index dosyası bulunamadı. Yeni index oluşturuluyor.")
        _hash_index = {}
    return _hash_index

def _save_hash_index():
    """Hash index'i diske (JSON) kaydeder."""
    if _hash_index is None:
        return
    
    index_path = _get_hash_index_path()
    try:
        os.makedirs(settings.DB_PERSIST_DIR, exist_ok=True)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(_hash_index, f, ensure_ascii=False, indent=2)
    except IOError as e:
        logger.error(f"Hash index diske kaydedilemedi: {index_path} - Hata: {e}", exc_info=True)

def _get_document_signature(doc: Document) -> str:
    """
    Bir Document objesi için benzersiz ve tutarlı bir imza (SHA-256 hash) oluşturur.
    İmza, (kaynak + sayfa + içeriğin başı) karmasıdır.
    """
    source = doc.metadata.get("source", "")
    page = str(doc.metadata.get("page", ""))
    # İçeriğin ilk 200 karakteri, değişikliği tespit etmek için yeterlidir
    head = (doc.page_content or "")[:200]
    
    raw_signature = f"{source}|{page}|{head}"
    return hashlib.sha256(raw_signature.encode("utf-8")).hexdigest()

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Embedding modelini 'lazy load' ile başlatır ve döndürür."""
    global _embeddings
    if _embeddings is None:
        logger.info(f"Google Embedding Modeli ({settings.EMBEDDING_MODEL}) başlatılıyor...")
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY
        )
    return _embeddings

def get_vectorstore() -> Chroma:
    """
    Merkezi ChromaDB örneğini 'lazy load' ile başlatır ve döndürür.
    Bu fonksiyon, veritabanına bir şey eklemez, sadece okuma/erişim sağlar.
    (örn. RAG zinciri tarafından kullanılır)
    """
    global _db
    if _db is None:
        logger.info(f"Mevcut ChromaDB veritabanı yükleniyor: {settings.DB_PERSIST_DIR}")
        if not os.path.exists(settings.DB_PERSIST_DIR):
            logger.warning("Veritabanı dizini bulunamadı. Boş bir veritabanı başlatılıyor.")
            # Bu, en azından RAG zincirinin hata vermeden boş başlamasını sağlar
        
        _db = Chroma(
            persist_directory=settings.DB_PERSIST_DIR,
            embedding_function=get_embeddings()
        )
    return _db

def add_documents_to_store(chunked_docs: List[Document]) -> int:
    """
    (KRİTİK FONKSİYON) Belgeleri veritabanına 'idempotent' (güvenli) bir şekilde ekler.
    
    Mükerrer (duplicate) belgeleri filtreler ve sadece yenilerini ekler.
    
    Argümanlar:
        chunked_docs (List[Document]): 'tasks.py'den gelen parçalanmış belgeler.

    Döndürür:
        int: Veritabanına *gerçekten* eklenen yeni belge sayısı.
    """
    if not chunked_docs:
        logger.warning("Veritabanına eklemek için hiç belge gelmedi.")
        return 0

    db = get_vectorstore()
    hash_index = _load_hash_index()

    docs_to_add: List[Document] = []
    hashes_to_add: List[str] = []

    for doc in chunked_docs:
        signature = _get_document_signature(doc)
        if signature not in hash_index:
            docs_to_add.append(doc)
            hashes_to_add.append(signature)
        # else:
            # logger.debug(f"Mükerrer belge atlanıyor: {doc.metadata['source']} (Sayfa: {doc.metadata['page']})")

    if not docs_to_add:
        logger.info("Eklenecek yeni (mükerrer olmayan) belge bulunamadı.")
        return 0

    try:
        logger.info(f"Veritabanına {len(docs_to_add)} adet yeni belge parçası ekleniyor...")
        db.add_documents(docs_to_add)
        
        # Sadece veritabanı eklemesi başarılı olursa hash index'i güncelle
        for i, sig in enumerate(hashes_to_add):
            hash_index[sig] = {
                "source": docs_to_add[i].metadata.get("source"),
                "page": docs_to_add[i].metadata.get("page")
            }
        
        _save_hash_index()
        logger.info(f"{len(docs_to_add)} adet yeni belge eklendi ve hash index güncellendi.")
        return len(docs_to_add)

    except Exception as e:
        logger.critical(f"ChromaDB'ye belge eklenirken KRİTİK HATA oluştu: {e}", exc_info=True)
        # Veritabanı eklemesi başarısız olursa hash index'i kaydetme!
        return 0
