"""
RAG Sorgu Boru Hattı Modülü (v2.5 - RAGAS Uyumlu).

Bu modül, "Proje 2main"in ana RAG (Retrieval-Augmented Generation)
zincirini oluşturur ve yönetir.

Sorumlulukları:
- Tüm bileşenleri (LLM, prompt, retriever, re-ranker) 'lazy load' ile başlatmak.
- LangChain Expression Language (LCEL) kullanarak modüler bir sorgu
  boru hattı ('chain') oluşturmak.
- Dinamik meta veri filtrelemeyi (bölüme göre) desteklemek.
- İki aşamalı geri getirmeyi (Retrieval + Re-ranking) yönetmek.
- (YENİ v2.5) Hem 'answer' (yanıt), 'context_docs' (RAGAS için ham belgeler)
  hem de 'formatted_context' (Streamlit için formatlanmış metin) döndürmek.

Bu modül, 'src/services/app.py' (Streamlit) ve 'scripts/evaluate.py' (RAGAS)
tarafından çağrılır.
"""

import logging
from typing import Dict, Any, List
from langchain.schema.runnable import Runnable, RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Merkezi ayarlar ve loglama
from src.core.config import settings
from src.utils.logging_config import setup_logging

# Gerekli bileşenleri import et
from src.components.vectorstore_manager import get_vectorstore
# 'get_compression_retriever' yerine 'CrossEncoderReranker' sınıfını
# doğrudan kullanacağız, çünkü LCEL'de 'invoke' uyumluluğu için
# 'ContextualCompressionRetriever'ı atlamak daha temiz olabilir.
# (Veya 'get_compression_retriever' içindeki sınıfı
# 'Runnable' olarak güncelleyebiliriz - Şimdilik basit tutalım)
from src.components.reranker import CrossEncoderReranker # (Dosya 14)
from src.components.vectorstore_manager import get_embeddings # RAGAS için

setup_logging()
logger = logging.getLogger(__name__)

# --- Global (Lazy Loaded) Değişkenler ---
_llm: ChatGoogleGenerativeAI | None = None
_prompt_template: ChatPromptTemplate | None = None
_rag_chain: Runnable | None = None


def get_llm() -> ChatGoogleGenerativeAI:
    """Gemini LLM'i 'lazy load' ile başlatır."""
    global _llm
    if _llm is None:
        logger.info(f"Gemini LLM Modeli ({settings.LLM_MODEL}) başlatılıyor...")
        _llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.0,
            convert_system_message_to_human=True 
        )
    return _llm

def _get_prompt_template() -> ChatPromptTemplate:
    """
    "Nihai v2" planına uygun, hassas "birebir çıkarım" (extraction)
    istem şablonunu 'lazy load' ile başlatır.
    """
    global _prompt_template
    if _prompt_template is None:
        logger.info("Hassas çıkarım (extraction) istem şablonu oluşturuluyor...")
        
        PROMPT_TEMPLATE = """
SEN: Sen, moleküler biyoloji ve genetik alanında uzman bir araştırma asistanısın.
GÖREVİN: Sana verilen SORU'yu yanıtlamak için SADECE sana sağlanan BAĞLAM'daki cümleleri kullanmaktır.

KURALLAR:
1. Cevabın, BAĞLAM'da bulunan **orijinal cümlenin veya cümlelerin birebir aynısı** olmalıdır.
2. ASLA özetleme yapma, yorum katma, bilgileri birleştirme veya cümleleri yeniden yazma.
3. Eğer BAĞLAM içinde soruyu doğrudan yanıtlayan bir cümle bulamazsan, 'İlgili spesifik cümle sağlanan bağlamda bulunamadı.' yanıtını ver.
4. ASLA halüsinasyon görme veya bağlam dışı bilgi kullanma.
5. Yanıtın sadece çıkarılan cümlelerden oluşmalıdır. Ekstra bir giriş cümlesi kullanma.

BAĞLAM:
---
{context}
---

SORU:
{question}
"""
        _prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return _prompt_template

def _format_docs_with_sources(docs: List[Document]) -> str:
    """
    Re-ranker'dan gelen belgeleri, LLM'in okuyabileceği
    temiz bir 'BAĞLAM' formatına (Streamlit için) dönüştürür.
    """
    if not docs:
        return "Sağlanan bağlamda ilgili bilgi bulunamadı."
        
    formatted_context = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata
        source_info = (
            f"Kaynak: {metadata.get('source', 'Bilinmiyor')}, "
            f"Sayfa: {metadata.get('page', 'N/A')}, "
            f"Bölüm: {metadata.get('section', 'N/A')}"
        )
        formatted_context.append(f"--- Belge {i+1} ({source_info}) ---\n{doc.page_content}")
    
    return "\n\n".join(formatted_context)


def setup_rag_chain() -> Runnable:
    """
    (ANA FONKSİYON - YENİ v2.5) Tüm RAG bileşenlerini birleştirerek
    çalıştırılabilir bir LangChain (LCEL) zinciri oluşturur ve döndürür.

    Bu zincir, RAGAS ve Streamlit'in ihtiyaçlarını karşılamak üzere
    paralel olarak 3 çıktı üretir: 'answer', 'context_docs', 'formatted_context'.
    
    Döndürür:
        Runnable: 'invoke' edilebilecek (çağrılabilecek) LCEL zinciri.
    """
    global _rag_chain
    if _rag_chain is not None:
        return _rag_chain

    logger.info("Yeni bir RAG Boru Hattı (LCEL Zinciri - v2.5 RAGAS Uyumlu) oluşturuluyor...")

    try:
        # --- Bileşenleri Başlat ---
        llm = get_llm()
        prompt = _get_prompt_template()
        vectorstore = get_vectorstore()
        reranker = CrossEncoderReranker(top_n=settings.RERANKER_TOP_N)

        # --- Zincir Mantığını Tanımla ---

        # Adım 1: Temel Geri Getirici (Retriever)
        # Bu fonksiyon input_dict alır (question, section)
        def retrieve_base_docs(input_dict: Dict[str, Any]) -> List[Document]:
            """
            Aşama 1: Dinamik filtrelemeyi uygular ve 'k' (örn. 25) adet
            belgeyi ChromaDB'den hızlıca alır.
            """
            query = input_dict["question"]
            section = input_dict.get("section", "Tüm Bölümler")
            
            search_kwargs = {"k": settings.BASE_RETRIEVER_K} # k=25
            
            if section and section != "Tüm Bölümler":
                search_kwargs["filter"] = {"section": section}
                logger.info(f"Dinamik filtre uygulandı: Bölüm = '{section}'")
            else:
                logger.info("Dinamik filtre uygulanmadı (Tüm Bölümler).")

            base_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )
            return base_retriever.invoke(query)

        # Adım 2: Yeniden Sıralayıcı (Re-Ranker)
        # Bu, input_dict ve Aşama 1'den gelen 'base_docs'u alır
        def rerank_docs(input_with_docs: Dict[str, Any]) -> List[Document]:
            """
            Aşama 2: 'base_docs'u alır ve 'question'a göre yeniden
            sıralayarak en iyi 'n' (örn. 5) adedi döndürür.
            """
            query = input_with_docs["question"]
            base_docs = input_with_docs["base_docs"]
            return reranker.rerank(query=query, docs=base_docs)
        
        # --- LCEL Zincirlerini Kur ---

        # Bu, 'input_dict'i alır ve RAGAS'ın ihtiyaç duyduğu
        # 'context_docs' (ham belgeler) listesini üretir.
        # Akış: input_dict -> (retrieve_base_docs) -> base_docs
        #        input_dict + base_docs -> (rerank_docs) -> context_docs (List[Document])
        context_docs_chain = {
            "base_docs": RunnableLambda(retrieve_base_docs),
            "question": RunnableLambda(lambda x: x["question"])
        } | RunnableLambda(rerank_docs)
        
        # Bu, 'context_docs'u (ham belgeler) alır ve Streamlit'in
        # ihtiyaç duyduğu 'formatted_context' (formatlanmış metin)
        # dizesini üretir.
        formatted_context_chain = context_docs_chain | RunnableLambda(_format_docs_with_sources)

        # Bu, 'formatted_context' ve 'question'ı alır ve
        # LLM'in ürettiği 'answer' (yanıt) dizesini üretir.
        answer_generation_chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        # Bu, 'formatted_context_chain' ve 'answer_generation_chain'i
        # paralel olarak besler.
        answer_chain = (
            {
                "context": formatted_context_chain,
                "question": RunnableLambda(lambda input_dict: input_dict["question"])
            }
            | answer_generation_chain
        )

        # (ANA ZİNCİR) 'final_chain', 'app.py' ve 'evaluate.py'nin
        # ihtiyaç duyduğu tüm çıktıları paralel olarak üretir.
        # input_dict'i alır ve bir dict döndürür.
        _rag_chain = RunnableParallel(
            {
                # Streamlit ve RAGAS için yanıt
                "answer": answer_chain, 
                
                # RAGAS için ham belgeler
                "context_docs": context_docs_chain, 
                
                # Streamlit'in 'expander'ı için formatlanmış metin
                "formatted_context": formatted_context_chain 
            }
        )
        
        logger.info("RAG Boru Hattı (LCEL Zinciri) başarıyla oluşturuldu (v2.5 RAGAS Uyumlu).")
        return _rag_chain

    except Exception as e:
        logger.critical(f"RAG Boru Hattı (LCEL Zinciri) oluşturulurken KRİTİK HATA oluştu: {e}", exc_info=True)
        return None

