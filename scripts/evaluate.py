"""
Gelişmiş RAG Değerlendirme Betiği (v2.5 - RAGAS + MLflow).

Bu betik, "Proje 2main" RAG boru hattının kalitesini, "Tier 1"
(FAANG seviyesi) yol haritasına uygun olarak, RAGAS framework'ü
kullanarak ölçer.

Sorumlulukları:
1.  'data/evaluation_set.jsonl' dosyasındaki test setini (soru + altın cevap)
    yükler.
2.  v2.5 RAG zincirimizi ('rag_chain.py') çağırarak her soru için
    'answer' (yanıt) ve 'context_docs' (ham belgeler) üretir.
3.  RAGAS metriklerini (faithfulness, answer_relevancy, context_precision,
    context_recall) hesaplar.
4.  Tüm parametreleri (örn. llm_model, chunk_size) ve metrikleri (örn.
    ragas_score) MLflow'a ('mlflow/' klasörüne) kaydeder.
5.  Sonuçların bir özetini 'eval_results.csv' dosyasına kaydeder.

Çalıştırma (proje kök dizininden):
$ python -m scripts.evaluate
(Veya Docker container içinde çalıştırmak için - Önerilen):
$ docker-compose exec web python -m scripts.evaluate
"""

import logging
import time
import csv
import statistics
from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset
import mlflow
import ragas
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

import sys
import os

# Betiğin 'src' paketini bulabilmesi için proje kök dizinini 'sys.path'e ekle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Projemizin 'src' klasöründeki bileşenlerini import et
try:
    from src.core.config import settings
    from src.utils.logging_config import setup_logging
    # v2.5 RAGAS Uyumlu RAG ZİNCİRİ (Dosya 25)
    from src.pipeline.rag_chain import setup_rag_chain, get_llm
    from src.components.vectorstore_manager import get_embeddings
except ImportError as e:
    print(f"HATA: 'src' modülleri import edilemedi. Hata: {e}")
    print("Lütfen bu betiği projenin kök dizininden (proje-2main/)")
    print("şu komutla çalıştırın: python -m scripts.evaluate")
    sys.exit(1)

# --- Loglamayı Başlat ---
setup_logging()
logger = logging.getLogger(__name__)

# --- Test Seti (Altın Standart) Yolu ---
# İdeal v2 planına göre, test seti artık 'data/' klasöründe
TEST_SET_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation_set.jsonl")

# --- RAGAS Metriklerini Yapılandır ---
# RAGAS, metrikleri (örn. answer_relevancy) hesaplamak için
# LLM ve Embedding modellerine ihtiyaç duyar.
ragas_metrics = [
    faithfulness,       # Yanıt, sağlanan bağlama ne kadar sadık? (Halüsinasyon tespiti)
    answer_relevancy,   # Yanıt, soruyla ne kadar alakalı?
    context_precision,  # Getirilen bağlamın ne kadarı soruyla alakalı?
    context_recall,     # 'Altın cevap' için gerekli bağlamın ne kadarı getirildi?
]

def load_evaluation_set(path: str) -> List[Dict[str, str]]:
    """
    Değerlendirme setini (soru, altın cevap) bir JSONL dosyasından yükler.
    
    Dosya formatı (her satır bir JSON objesi):
    {"question": "CRISPR-Cas9 nedir?", "ground_truth": "CRISPR-Cas9, gen düzenlemesi için..."}
    """
    if not os.path.exists(path):
        logger.warning(f"Değerlendirme seti bulunamadı: {path}")
        logger.warning("Lütfen 'data/evaluation_set.jsonl' dosyasını oluşturun.")
        # Değerlendirme betiğinin çökmemesi için sahte (dummy) bir set döndür
        return [
            {
                "question": "What is the role of Cas9 in CRISPR?",
                "ground_truth": "Cas9 nuclease is an enzyme that uses guide RNA to find and cut specific strands of DNA, causing a double-strand break."
            },
            {
                "question": "What is genetic engineering?",
                "ground_truth": "Genetic engineering, also called genetic modification, is the direct manipulation of an organism's genes using biotechnology."
            }
        ]
        
    logger.info(f"Değerlendirme seti yükleniyor: {path}")
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except Exception as e:
        logger.error(f"Değerlendirme seti ({path}) yüklenemedi: {e}", exc_info=True)
        return [] # Boş liste döndür
    return data

def run_rag_pipeline(rag_chain: Any, test_set: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Tüm test seti üzerinde RAG zincirini çalıştırır ve
    RAGAS'ın ihtiyaç duyduğu (question, ground_truth, answer, contexts)
    veriyi toplar.
    """
    logger.info(f"{len(test_set)} adet soru üzerinde RAG boru hattı çalıştırılıyor...")
    results = []
    latencies = []

    for item in test_set:
        query = item["question"]
        logger.info(f"Sorgu: '{query}'")
        
        start_time = time.perf_counter()
        
        try:
            # --- RAG ZİNCİRİNİ ÇAĞIR (v2.5) ---
            response = rag_chain.invoke({
                "question": query,
                "section": "Tüm Bölümler" # Değerlendirme için filtreleme yapma
            })
            
            end_time = time.perf_counter()
            latencies.append(end_time - start_time)

            # RAGAS'ın beklediği formatı doldur
            results.append({
                "question": query,
                "ground_truth": item["ground_truth"],
                "answer": response["answer"],
                # RAGAS, 'context_docs'u 'contexts' anahtarı altında
                # ve string listesi olarak bekler
                "contexts": [doc.page_content for doc in response["context_docs"]]
            })
        except Exception as e:
            logger.error(f"RAG Zinciri sorgusu '{query}' başarısız oldu: {e}", exc_info=True)
            results.append({
                "question": query,
                "ground_truth": item["ground_truth"],
                "answer": "ERROR",
                "contexts": []
            })
        
    avg_latency = statistics.mean(latencies) if latencies else 0
    logger.info(f"RAG boru hattı tamamlandı. Ortalama gecikme (latency): {avg_latency:.4f} saniye")
    
    return pd.DataFrame(results), avg_latency

def run_ragas_evaluation(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    RAGAS kullanarak toplanan sonuçlar üzerinde
    'senior seviye' metrikleri hesaplar.
    """
    logger.info("RAGAS değerlendirmesi başlıyor... (Bu işlem zaman alabilir)")
    
    # RAGAS, 'datasets' kütüphanesinin formatını bekler
    dataset = Dataset.from_pandas(results_df)
    
    try:
        # RAGAS'a değerlendirme için hangi modelleri kullanacağını söyle
        ragas.set_metrics_config({
            "llm": get_llm(),
            "embeddings": get_embeddings()
        })
        
        # Değerlendirmeyi çalıştır
        score = ragas.evaluate(
            dataset=dataset,
            metrics=ragas_metrics
        )
        
        logger.info("RAGAS değerlendirmesi tamamlandı.")
        
        # RAGAS skorunu (tüm metriklerin ortalaması) ve
        # her bir metriği 'score.to_pandas()' ile al
        return score.to_pandas()
        
    except Exception as e:
        logger.error(f"RAGAS değerlendirmesi başarısız oldu: {e}", exc_info=True)
        # Başarısız olursa boş bir DataFrame döndür
        return pd.DataFrame()

def run_evaluation_and_log_to_mlflow(output_csv: str = "eval_results.csv"):
    """
    (ANA FONKSİYON) Değerlendirmeyi çalıştırır ve
    sonuçları MLflow'a (ve CSV'ye) kaydeder.
    """
    logger.info("MLOps Değerlendirme Süreci (v2.5) başlatılıyor...")
    
    # --- 1. MLflow'u Yapılandır ---
    # 'config.py' (Dosya 8) ve 'docker-compose.yml' (Dosya 1)
    # içinde tanımladığımız 'file://' yolunu kullanır.
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Proje 2main - RAG Değerlendirme")
    
    with mlflow.start_run(run_name="RAGAS_v2.5_Run") as run:
        logger.info(f"MLflow deneyi başlatıldı. URI: {mlflow.get_tracking_uri()}")
        logger.info(f"Run ID: {run.info.run_id}")

        # --- 2. Parametreleri Logla (Tier 1 Kanıt) ---
        logger.info("MLflow'a parametreler loglanıyor...")
        mlflow.log_params({
            "llm_model": settings.LLM_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "base_retriever_k": settings.BASE_RETRIEVER_K,
            "reranker_top_n": settings.RERANKER_TOP_N,
            "chunk_size": 1500, # (Şimdilik config.py'de yok, manuel)
            "chunk_overlap": 250 # (Şimdilik config.py'de yok, manuel)
        })

        # --- 3. Test Setini Yükle ---
        test_set = load_evaluation_set(TEST_SET_PATH)
        if not test_set:
            logger.critical("Değerlendirme seti boş veya yüklenemedi. İşlem durduruluyor.")
            return

        # --- 4. RAG Zincirini Kur ve Çalıştır ---
        rag_chain = setup_rag_chain()
        if not rag_chain:
            logger.critical("RAG Zinciri başlatılamadı. İşlem durduruluyor.")
            return
            
        results_df, avg_latency = run_rag_pipeline(rag_chain, test_set)
        
        # --- 5. RAGAS Metriklerini Hesapla ---
        ragas_results_df = run_ragas_evaluation(results_df)

        if ragas_results_df.empty:
            logger.critical("RAGAS değerlendirmesi başarısız oldu, metrikler loglanamıyor.")
            return

        # --- 6. Metrikleri Logla (Tier 1 Kanıt) ---
        logger.info("MLflow'a metrikler loglanıyor...")
        # Ana RAGAS skoru (tüm metriklerin harmonik ortalaması)
        ragas_score = ragas_results_df["ragas_score"].mean()
        mlflow.log_metric("ragas_score", ragas_score)
        
        # Diğer tüm RAGAS metrikleri
        mlflow.log_metric("faithfulness", ragas_results_df["faithfulness"].mean())
        mlflow.log_metric("answer_relevancy", ragas_results_df["answer_relevancy"].mean())
        mlflow.log_metric("context_precision", ragas_results_df["context_precision"].mean())
        mlflow.log_metric("context_recall", ragas_results_df["context_recall"].mean())
        
        # Performans Metriği
        mlflow.log_metric("latency_avg_seconds", avg_latency)

        # --- 7. Sonuçları CSV'ye Kaydet ---
        output_path = os.path.join(PROJECT_ROOT, output_csv)
        try:
            ragas_results_df.to_csv(output_path, index=False, encoding="utf-8")
            logger.info(f"RAGAS sonuçları başarıyla '{output_path}' dosyasına kaydedildi.")
            # MLflow'a 'artifact' (kanıt) olarak da kaydet
            mlflow.log_artifact(output_path)
        except IOError as e:
            logger.error(f"Sonuçlar CSV dosyasına yazılamadı: {e}", exc_info=True)

        logger.info("MLflow deneyi başarıyla tamamlandı.")
        print("\n--- Değerlendirme Özeti (RAGAS) ---")
        print(ragas_results_df.mean(numeric_only=True))
        print("-----------------------------------")
        print(f"\nDeney sonuçlarını görmek için proje kök dizininde şu komutu çalıştırın:\nmlflow ui --backend-store-uri {settings.MLFLOW_TRACKING_URI.replace('file://', '')}")

if __name__ == "__main__":
    """
    Bu betiği bir modül olarak çalıştırmak için (proje kök dizininden):
    $ python -m scripts.evaluate
    
    VEYA Docker içinden (Önerilen, çünkü tüm bağımlılıklar oradadır):
    $ docker-compose exec web python -m scripts.evaluate
    """
    run_evaluation_and_log_to_mlflow()

