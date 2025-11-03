"""
İki Aşamalı Geri Getirme için Cross-Encoder Re-Ranker Modülü.

Bu modül, RAG boru hattının 2. Aşamasını gerçekleştirir.
- 1. Aşamada (vektör araması) gelen 'k' adet belgeyi (örn. 25) alır.
- Küçük, uzmanlaşmış bir Cross-Encoder model kullanarak bu belgeleri
  kullanıcının orijinal sorusuna göre yeniden puanlar.
- En alakalı 'n' adet belgeyi (örn. 5) döndürür.

Bu, LLM'e giden bağlamın (context) kalitesini ve doğruluğunu
dramatik biçimde artırır.

Kullanılan model (cross-encoder/ms-marco-MiniLM-L-6-v2), RAG
re-ranking için özel olarak eğitilmiştir.
"""

import logging
import math
from typing import List
from langchain.docstore.document import Document
from sentence_transformers import CrossEncoder

# Merkezi ayarlar ve loglama
from src.core.config import settings
from src.utils.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# --- Global (Lazy Loaded) Değişken ---
_reranker_model: CrossEncoder | None = None

def _get_reranker_model() -> CrossEncoder:
    """
    Cross-Encoder modelini 'lazy load' ile (sadece ilk
    ihtiyaç duyulduğunda) başlatan ve döndüren fonksiyon.
    Model, 'settings' üzerinden yapılandırılır.
    """
    global _reranker_model
    if _reranker_model is None:
        model_name = settings.RERANKER_MODEL
        logger.info(f"Cross-Encoder Re-ranker modeli ({model_name}) başlatılıyor...")
        try:
            # Modeli CPU veya (varsa) GPU üzerine yükle
            _reranker_model = CrossEncoder(model_name)
            logger.info("Re-ranker modeli başarıyla yüklendi.")
        except Exception as e:
            logger.critical(f"Re-ranker modeli ({model_name}) yüklenemedi. Hata: {e}", exc_info=True)
            raise e
    return _reranker_model

class CrossEncoderReranker:
    """
    LangChain'in 'ContextualCompressionRetriever'ı ile uyumlu,
    'sentence-transformers' kullanan Re-Ranker sınıfı.
    
    'rerank' metodu, 'batch' (toplu) modda çalışarak yüksek verimlilik sağlar.
    """
    def __init__(self, top_n: int = settings.RERANKER_TOP_N, batch_size: int = 64):
        """
        Re-ranker'ı başlatır.
        
        Argümanlar:
            top_n (int): LLM'e gönderilecek en iyi belge sayısı (ayarlardan gelir).
            batch_size (int): Modelin aynı anda kaç belgeyi puanlayacağı.
        """
        self.model = _get_reranker_model()
        self.top_n = top_n
        self.batch_size = batch_size

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Belge listesini bir sorguya göre yeniden sıralar.
        
        Argümanlar:
            query (str): Kullanıcının orijinal sorusu.
            docs (List[Document]): Vektör deposundan gelen (örn. 25 adet) belge listesi.
            
        Döndürür:
            List[Document]: Alaka düzeyine göre sıralanmış en iyi 'top_n' (örn. 5 adet) belge.
        """
        if not docs:
            logger.warning("Re-ranker'a yeniden sıralanacak belge gelmedi.")
            return []
            
        logger.info(f"Re-ranker {len(docs)} belgeyi '{query}' sorgusuna göre sıralıyor...")

        # Modelin 'predict' metodu [sorgu, belge_metni] çiftleri bekler
        doc_contents = [d.page_content for d in docs]
        
        # --- Batch (Toplu) Puanlama ---
        # 'v2.1' planına uygun olarak, 25 belgeyi tek tek değil,
        # 'batch_size' (örn. 64) gruplar halinde modele göndererek
        # GPU/CPU'dan tam verim alıyoruz.
        scores = self._score_in_batches(query, doc_contents)
        
        # Puanları (scores) ve orijinal indeksleri (index) eşleştir
        # Örn: [(0, 0.98), (1, 0.02), (2, 0.75), ...]
        indexed_scores = list(enumerate(scores))

        # Puanlara göre büyükten küçüğe sırala
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # En yüksek puanlı 'top_n' adet belgenin orijinal indekslerini al
        selected_indices = [idx for idx, score in indexed_scores[:self.top_n]]
        
        # Orijinal 'docs' listesinden bu indekslere karşılık gelen belgeleri seç
        reranked_docs = [docs[i] for i in selected_indices]

        logger.info(f"Re-ranker {len(docs)} belgeyi {len(reranked_docs)} belgeye indirdi.")
        return reranked_docs

    def _score_in_batches(self, query: str, texts: List[str]) -> List[float]:
        """
Half-private' yardımcı metot. Puanlamayı 'batch'ler halinde yapar.
        """
        scores = []
        num_texts = len(texts)
        # Gerekli batch (grup) sayısını hesapla
        num_batches = math.ceil(num_texts / self.batch_size)

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, num_texts)
            
            # Modelin beklediği format: [ [sorgu, metin1], [sorgu, metin2], ... ]
            batch_pairs = [[query, texts[j]] for j in range(start_idx, end_idx)]
            
            if batch_pairs:
                # 'predict' metodu puan listesi döndürür
                batch_scores = self.model.predict(
                    batch_pairs, 
                    show_progress_bar=False # Logları kirletmemesi için ilerleme çubuğunu kapat
                )
                scores.extend(batch_scores)
                
        return [float(s) for s in scores]

# LangChain'in ContextualCompressionRetriever'ı ile doğrudan
# uyumlu olması için bu sınıfı kullanan bir fonksiyon
# (Bu, 'rag_pipeline.py'de kullanılacak)
def get_compression_retriever(base_retriever):
    """
    LangChain'in ContextualCompressionRetriever'ını bizim
    CrossEncoderReranker'ımız ile yapılandırır.
    """
    from langchain.retrievers import ContextualCompressionRetriever

    # 'base_compressor' olarak sınıfımızın bir örneğini veriyoruz.
    # LangChain, bu objenin 'rerank' metodunu (veya 'compress_documents')
    # otomatik olarak çağıracaktır.
    compressor = CrossEncoderReranker(top_n=settings.RERANKER_TOP_N)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    return compression_retriever
