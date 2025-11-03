"""
Hiperparametre Optimizasyonu (v2.5 - Optuna + MLflow).

Bu betik, "Proje 2main" RAG boru hattının "Tier 1" (FAANG seviyesi)
yol haritasındaki (Madde 3) gereksinimleri karşılar.

Sorumlulukları:
1.  Optuna kullanarak RAG boru hattı için en iyi hiperparametreleri
    (retriever_k, reranker_top_n) arar.
2.  Her bir "deneme" (trial) için 'scripts/evaluate.py'deki RAGAS
    değerlendirme mantığını çalıştırır.
3.  Optuna'nın 'MLflowCallback' entegrasyonunu kullanarak TÜM denemeleri
    (parametreler ve metrikler) 'mlflow/' klasörüne otomatik olarak kaydeder.

Çalıştırma (proje kök dizininden):
$ python -m scripts.optimize_hyperparams

VEYA Docker içinden (Önerilen, çünkü tüm bağımlılıklar oradadır):
$ docker-compose exec web python -m scripts.optimize_hyperparams
"""

import logging
import optuna
import mlflow
from optuna.integration.mlflow import MLflowCallback
import sys
import os
from typing import Dict, Any, List
from datasets import Dataset
import ragas
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Betiğin 'src' paketini bulabilmesi için proje kök dizinini 'sys.path'e ekle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Projemizin 'src' klasöründeki bileşenlerini import et
try:
    from src.core.config import settings
    from src.utils.logging_config import setup_logging
    from src.components.vectorstore_manager import get_vectorstore, get_embeddings
    from src.components.reranker import CrossEncoderReranker
    from src.pipeline.rag_chain import _get_llm, _get_prompt_template, _format_docs_with_sources
    # 'evaluate.py'den (Dosya 20) değerlendirme seti yükleyiciyi 'import' ediyoruz
    # (Bu fonksiyonu 'src/components/evaluation_utils.py'ye taşımak 'v3' olurdu,
    # şimdilik 'v2.5' için 'evaluate'den import etmek pragmatiktir)
    from scripts.evaluate import load_evaluation_set, TEST_SET_PATH
    
    # LangChain importları (RAG zincirini dinamik olarak kurmak için)
    from langchain.schema.runnable import Runnable, RunnableParallel, RunnableLambda, RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser
    from langchain.retrievers import ContextualCompressionRetriever

except ImportError as e:
    print(f"HATA: 'src' veya 'scripts' modülleri import edilemedi. Hata: {e}")
    print("Lütfen bu betiği projenin kök dizininden (proje-2main/)")
    print("şu komutla çalıştırın: python -m scripts.optimize_hyperparams")
    sys.exit(1)

# --- Loglamayı Başlat ---
setup_logging()
logger = logging.getLogger(__name__)

# --- RAGAS Metriklerini Yapılandır ---
ragas_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

# --- Global Değişkenler (Test Seti ve Vektör Deposu) ---
# Optimizasyon süresince (50 deneme) bu verilerin sadece bir kez yüklenmesi gerekir
try:
    EVALUATION_SET = load_evaluation_set(TEST_SET_PATH)
    VECTORSTORE = get_vectorstore()
    RAGAS_LLM = _get_llm()
    RAGAS_EMBEDDINGS = get_embeddings()
except Exception as e:
    logger.critical(f"Optimizasyon başlatılamadı: Gerekli bileşenler (Veritabanı/Test Seti) yüklenemedi. Hata: {e}")
    sys.exit(1)


def setup_dynamic_rag_chain(retriever_k: int, reranker_top_n: int) -> Runnable:
    """
    Optuna'nın verdiği 'deneme' (trial) parametrelerine göre
    RAG zincirini dinamik olarak oluşturan fonksiyon.
    """
    
    # 1. Temel Geri Getirici (Dinamik k ile)
    base_retriever = VECTORSTORE.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retriever_k} # Optuna'dan gelen 'k'
    )
    
    # 2. Re-Ranker (Dinamik top_n ile)
    compressor = CrossEncoderReranker(top_n=reranker_top_n) # Optuna'dan gelen 'top_n'
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # 3. Dinamik Filtreleme Mantığı (Filtresiz)
    def dynamic_retriever_no_filter(input_dict: Dict[str, Any]) -> List[Document]:
        query = input_dict["question"]
        # Optimizasyon sırasında 'Bölüm' filtresi KULLANMIYORUZ
        # (tüm metriklerin tutarlı olması için)
        compression_retriever.base_retriever.search_kwargs["filter"] = {} 
        return compression_retriever.invoke(query)

    # 4. LCEL Zinciri (Dosya 25'in kopyası, ama dinamik)
    context_chain = (
        RunnableLambda(dynamic_retriever_no_filter)
        .with_config(run_name="DynamicRetriever")
    )
    
    answer_chain = (
        {
            "context_docs": context_chain,
            "question": RunnablePassthrough()
        }
        | RunnableLambda(lambda x: {
            "context": _format_docs_with_sources(x["context_docs"]),
            "question": x["question"]
        })
        | _get_prompt_template()
        | _get_llm()
        | StrOutputParser()
    )

    final_chain = RunnableParallel(
        {
            "answer": answer_chain,
            "context_docs": context_chain 
        }
    )
    return final_chain

def objective(trial: optuna.Trial) -> float:
    """
    Optuna'nın optimize edeceği ana "hedef" (objective) fonksiyonu.
    
    Her 'deneme' (trial) için bu fonksiyon çalışır, RAGAS skorunu hesaplar
    ve Optuna'ya döndürür.
    """
    
    # --- 1. Hiperparametre Arama Alanını Tanımla ---
    # 'v3' yol haritası (Madde 3) önerilerine göre:
    retriever_k = trial.suggest_int('retriever_k', 10, 50, step=5)
    reranker_top_n = trial.suggest_int('reranker_top_n', 3, 10, step=1)
    
    # (Opsiyonel: LLM sıcaklığını da optimize edebiliriz)
    # temperature = trial.suggest_float('temperature', 0.0, 0.7, step=0.1)
    
    logger.info(f"[Optuna Trial #{trial.number}] Başlatılıyor...")
    logger.info(f"Parametreler: retriever_k={retriever_k}, reranker_top_n={reranker_top_n}")

    try:
        # --- 2. RAG Zincirini Dinamik Parametrelerle Kur ---
        rag_chain = setup_dynamic_rag_chain(retriever_k, reranker_top_n)

        # --- 3. RAG Boru Hattını Test Seti Üzerinde Çalıştır ---
        # (Bu mantık 'evaluate.py' (Dosya 20) ile aynıdır)
        results_data = []
        for item in EVALUATION_SET:
            response = rag_chain.invoke({"question": item["question"]})
            results_data.append({
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": response["answer"],
                "contexts": [doc.page_content for doc in response["context_docs"]]
            })
        
        results_df = pd.DataFrame(results_data)
        dataset = Dataset.from_pandas(results_df)

        # --- 4. RAGAS Metriklerini Hesapla ---
        ragas.set_metrics_config({
            "llm": RAGAS_LLM,
            "embeddings": RAGAS_EMBEDDINGS
        })
        
        score = ragas.evaluate(
            dataset=dataset,
            metrics=ragas_metrics
        )
        
        ragas_score = score["ragas_score"] # Optimize edeceğimiz ana hedef
        logger.info(f"[Optuna Trial #{trial.number}] Tamamlandı. RAGAS Skoru: {ragas_score:.4f}")

        # (MLflowCallback bu metrikleri otomatik loglayacak)
        
        return ragas_score # Optuna'ya bu denemenin skorunu döndür

    except Exception as e:
        # Eğer deneme başarısız olursa (örn. RAGAS çökerse),
        # Optuna'ya bu denemenin "kötü" olduğunu söyle (0.0)
        logger.error(f"[Optuna Trial #{trial.number}] Başarısız Oldu: {e}", exc_info=True)
        return 0.0

def run_optimization():
    """
    (ANA FONKSİYON) Optuna optimizasyon sürecini başlatır ve MLflow'a bağlar.
    """
    logger.info("Hiperparametre Optimizasyon Süreci (v2.5) başlatılıyor...")
    
    # --- MLflow Entegrasyonu (Tier 1 Kanıt) ---
    # MLflowCallback, Optuna'nın her 'trial' sonucunu otomatik olarak
    # 'settings.MLFLOW_TRACKING_URI' (yani 'mlflow/' klasörümüz) içine kaydeder.
    mlflow_callback = MLflowCallback(
        tracking_uri=settings.MLFLOW_TRACKING_URI,
        experiment_name="Proje 2main - Optuna Optimizasyonu",
        metric_name="ragas_score"
    )

    # Optimizasyon "Çalışması" (Study) Oluştur
    study = optuna.create_study(
        study_name="rag_hyperparam_search",
        direction="maximize" # 'ragas_score'u maksimize etmeyi hedefliyoruz
    )
    
    # Optimizasyonu başlat (MLflow'a bağlayarak)
    study.optimize(
        objective, 
        n_trials=50, # Toplam 50 farklı parametre kombinasyonu dene
        callbacks=[mlflow_callback] # Her denemeyi MLflow'a logla
    )

    # --- Sonuçları Yazdır ---
    logger.info("Optimizasyon tamamlandı!")
    logger.info(f"En iyi deneme (Best Trial) #{study.best_trial.number}:")
    logger.info(f"  En Yüksek RAGAS Skoru: {study.best_value:.4f}")
    logger.info("  En İyi Parametreler:")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")

    print("\n--- Optimizasyon Tamamlandı ---")
    print(f"En iyi RAGAS Skoru: {study.best_value:.4f}")
    print(f"En iyi parametreler: {study.best_params}")
    print(f"\nTüm 50 denemenin sonuçlarını görmek için proje kök dizininde şu komutu çalıştırın:\nmlflow ui --backend-store-uri {settings.MLFLOW_TRACKING_URI.replace('file://', '')}")

if __name__ == "__main__":
    """
    Bu betiği bir modül olarak çalıştırmak için (proje kök dizininden):
    $ python -m scripts.optimize_hyperparams
    
    VEYA Docker içinden (Önerilen, çünkü tüm bağımlılıklar oradadır):
    $ docker-compose exec web python -m scripts.optimize_hyperparams
    """
    run_optimization()
