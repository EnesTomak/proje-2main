# 🔬 Proje 2main  -  Akıllı Araştırma Asistanı

**Proje 2main**, bilimsel PDF makalelerinden birebir ve atıf yapılabilir (citable) cümleleri çıkarmak için geliştirilmiş, yüksek doğruluklu bir **RAG (Retrieval-Augmented Generation)** sistemidir.

Bu proje, "temel" bir RAG sohbet botunun ötesine geçerek, **üretim seviyesi (production-grade)** bir mimariyi (Asenkron işleme, CI/CD, Test ve Metrikler) uygular.

---

## 🚀 Temel Özellikler

* **Asenkron PDF İşleme:** Celery & Redis kullanarak, 50 sayfalık 10 PDF yüklendiğinde bile "donmayan", ağır OCR ve gömme (embedding) işlemlerini arka planda yürüten bir arayüz.
* **Yüksek Doğruluklu Çıkarım:** Sadece vektör araması değil, Cross-Encoder (Re-ranker) kullanarak iki aşamalı (two-stage) bir geri getirme (retrieval) stratejisi uygular.
* **Hassas İstem Mühendisliği:** Gemini LLM'ini "özetleme" yapmaktan alıkoyan ve sadece birebir alıntı (extraction) yapmaya zorlayan özel bir istem (prompt) kullanır.
* **Akıllı Meta Veri Filtreleme:** Kullanıcıların, makalelerin "Giriş", "Tartışma" veya "Yöntemler" gibi spesifik bölümlerine göre arama yapmasına olanak tanır.
* **Kanıtlanmış Performans:** Sistemin doğruluğu, `scripts/evaluate.py` betiği kullanılarak precision@k metriği ile nicel olarak ölçülmüş ve kanıtlanmıştır.
* **Test Edilmiş Kod Kalitesi:** Sistem, pytest ile yazılmış birim (unit) ve entegrasyon (integration) testleri ile güvence altına alınmıştır.
* **Otomatik CI/CD Boru Hattı:** Jenkinsfile ile kod değişikliklerinin otomatik olarak test edilmesi ve (isteğe bağlı) dağıtılması.

---

## 🏛️ "Senior Seviye" Mimari

Bu proje, birbirinden bağımsız çalışan, ölçeklenebilir **3 ana servisten** oluşur ve `docker-compose.yml` ile yönetilir:

```
[Kullanıcı] -> [Web: Streamlit (app.py)] -> [Redis (Kuyruk)] <- [Worker: Celery (tasks.py)] -> [ChromaDB / Google AI API]
```

Servisler:

* **web (Streamlit):** Kullanıcı arayüzünü sunar, Celery görevlerini tetikler ve RAG zincirini çağırır.
* **worker (Celery):** Ağır PDF işleme (OCR, PyMuPDF, Gömme, ChromaDB'ye ekleme) görevlerini Redis kuyruğundan alıp asenkron olarak çalıştırır.
* **redis (Redis):** web ve worker servisleri arasında görev kuyruğu (broker) olarak görev yapar.

---

## 🛠️ Kurulum ve Çalıştırma (Yerel Geliştirme)

> Bu projeyi yerel makinenizde çalıştırmak için Docker Desktop'ın kurulu olması gerekir.

### 1. Projeyi Klonlayın

```bash
git clone https://github.com/sizin-kullanici-adiniz/proje-2main.git
cd proje-2main
```

### 2. Güvenli Yapılandırmayı Oluşturun

```bash
cp .env.example .env
```

Şimdi `.env` dosyasını açın ve `GOOGLE_API_KEY="..."` satırını düzenleyin.

### 3. Docker Compose ile Tüm Sistemi Başlatın

```bash
docker-compose up --build
```

### 4. Uygulamayı Kullanın

Tarayıcınızı açın ve [http://localhost:8501](http://localhost:8501) adresine gidin.

---

## 📈 Değerlendirme ve Metrikler (Kanıt)

Sistemimizin kalitesini kanıtlamak için `scripts/evaluate.py` betiğini kullanarak, **Baseline RAG** (sadece vektör arama) ile **Proje 2main (RAG + Re-ranker)** stratejilerini karşılaştırdık.

**Metrik:** Precision@5 (Bulunan 5 sonuçtan kaç tanesi "altın" anahtar kelimeleri içeriyor?)

| Strateji                           | Ortalama Doğruluk (P@5) | Ortalama Gecikme (s) |
| ---------------------------------- | ----------------------- | -------------------- |
| Baseline RAG (Sadece Vektör Arama) | %58.3                   | 0.12 s               |
| Proje 2main (RAG + Re-ranker)      | %81.6                   | 0.45 s               |

**Sonuç:** Sistemimiz (RAG + Re-ranker), ~330ms'lik bir gecikme maliyetiyle, doğruluk oranını (Precision@5) %23.3 puan (veya %40 oransal) artırmıştır.

---

## 🧪 Test ve Kalite Güvencesi (CI)

Projenin kalitesi ve sürdürülebilirliği pytest ile yazılmış birim (unit) ve entegrasyon (integration) testleri ile güvence altına alınmıştır.

Bu testler, Jenkinsfile CI/CD boru hattı tarafından her 'push' işleminde otomatik olarak çalıştırılır.

Yerel olarak test çalıştırmak için:

```bash
docker-compose up -d
docker-compose exec web pytest
```

---

## 📂 Proje Dizin Yapısı (İdeal v2 Mimarisi)

```bash
proje-2main/
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── Jenkinsfile
├── LICENSE
├── README.md
├── requirements.txt
│
├── data/
│   └── .gitkeep
│
├── notebooks/
│   └── .gitkeep
│
├── scripts/
│   ├── __init__.py
│   └── evaluate.py
│
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   ├── reranker.py
│   │   ├── text_splitter.py
│   │   └── vectorstore_manager.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── rag_chain.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   └── tasks.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── logging_config.py
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── components/
    │   ├── __init__.py
    │   └── test_document_processor.py
    ├── pipeline/
    │   ├── __init__.py
    │   └── test_rag_chain.py
    └── services/
        ├── __init__.py
        └── test_tasks.py
```

---

📘 **Lisans:** MIT
📅 **Sürüm:** 1.0
👩‍🔬 **Geliştirici:** Enes

---