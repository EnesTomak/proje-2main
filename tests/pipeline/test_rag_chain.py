"""
Entegrasyon Testleri - src.pipeline.rag_chain

Bu test betiği, ana RAG boru hattının (LCEL zinciri) iş mantığını doğrular.

Kritik "senior" pratikler:
- 'pytest.monkeypatch' kullanarak harici bağımlılıkları (LLM, DB, Re-ranker)
  'mock' (taklit) ederiz.
- Bu, API çağrıları yapmadan veya gerçek bir veritabanına ihtiyaç duymadan
  zincirin kendisinin (filtreleme, prompt formatlama) test edilmesini sağlar.
"""

import pytest
import sys
import os
from langchain.schema.runnable import Runnable
from langchain.schema.output_parser import StrOutputParser

# 'src' paketini testlerin bulabilmesi için
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

# Test edilecek modül
from src.pipeline import rag_chain
# 'conftest.py'den paylaşılan fixture'ı import et (pytest bunu bilir)
from tests.conftest import sample_document_list 

# --- Sahte (Mock) Fonksiyonlar ve Sınıflar ---

class MockLLM:
    """_get_llm() fonksiyonunu taklit eden sahte LLM."""
    def invoke(self, prompt):
        # LLM'in bir string döndürmesini simüle et
        return f"Mocked Answer based on: {prompt}"

class MockRetriever:
    """get_vectorstore().as_retriever() 'ı taklit eden sahte Retriever."""
    def __init__(self, docs):
        self.docs = docs
    def invoke(self, query):
        # Sorgu ne olursa olsun, fixture'daki belgeleri döndür
        return self.docs

class MockCompressionRetriever:
    """get_compression_retriever() 'ı taklit eden sahte Re-ranker."""
    def __init__(self, docs):
        self.docs = docs
    def invoke(self, query):
        # Re-ranker'ın belgeleri 2'ye düşürdüğünü simüle et
        return self.docs[:2] # Sadece ilk 2 belgeyi döndür

# --- Test Fonksiyonu ---

def test_setup_rag_chain_integration(monkeypatch, sample_document_list):
    """
    'setup_rag_chain' fonksiyonunun tüm RAG zincirini (LCEL)
    doğru bir şekilde kurduğunu ve çalıştırdığını test eder.
    
    'monkeypatch', harici çağrıları sahte fonksiyonlarımızla değiştirir.
    'sample_document_list', 'conftest.py'den gelen fixture'dır.
    """
    
    # --- Mock'ları Ayarla (Monkeypatching) ---
    
    # 1. LLM'i mock'la: 'rag_chain' içindeki '_get_llm' çağrıldığında
    #    bizim sahte 'MockLLM'imizi döndür.
    monkeypatch.setattr(rag_chain, "_get_llm", lambda: MockLLM())
    
    # 2. Vektör Deposunu mock'la: 'get_vectorstore' çağrıldığında
    #    'as_retriever' metoduna sahip sahte bir obje döndür.
    mock_db = type('MockDB', (object,), {
        'as_retriever': lambda **kwargs: MockRetriever(sample_document_list)
    })()
    monkeypatch.setattr(rag_chain, "get_vectorstore", lambda: mock_db)

    # 3. Re-ranker'ı mock'la: 'get_compression_retriever' çağrıldığında
    #    sahte 'MockCompressionRetriever'ımızı döndür.
    monkeypatch.setattr(
        rag_chain, 
        "get_compression_retriever", 
        lambda retriever: MockCompressionRetriever(sample_document_list)
    )

    # ÖNEMLİ: Zinciri (rag_chain) mock'lar ayarlandıktan *sonra*
    # 'setup_rag_chain'i çağırmalıyız.
    # Önceki 'session_state'i temizle (eğer varsa)
    rag_chain._rag_chain = None 
    
    # --- Test Edilen Fonksiyonu Çalıştır ---
    chain = rag_chain.setup_rag_chain()
    
    assert chain is not None
    assert isinstance(chain, Runnable) # LCEL objesi mi?

    # --- Zinciri Çalıştır (Invoke) ---
    test_input = {
        "question": "What is CRISPR?",
        "section": "Tüm Bölümler"
    }
    result = chain.invoke(test_input)

    # --- Doğrulamalar (Assertions) ---
    
    # 1. Yanıt (answer) doğru formatta mı?
    assert "answer" in result
    assert "Mocked Answer" in result["answer"] # MockLLM'den gelen yanıt mı?

    # 2. Bağlam (context) doğru formatta mı?
    assert "context" in result
    # MockCompressionRetriever'ın 2 belge döndürdüğünü test et
    assert "Belge 1" in result["context"]
    assert "Belge 2" in result["context"]
    assert "Belge 3" not in result["context"] # Re-ranker'ın 3.'yü elediğini doğrula
    
    # 'sample_document_list'ten gelen meta veriler bağlamda mı?
    assert "article1.pdf" in result["context"]
    assert "Introduction" in result["context"]
    assert "Methods" in result["context"]
    assert "Discussion" not in result["context"] # 3. belge elendiği için
