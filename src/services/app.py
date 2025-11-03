"""
Proje 2main (Tuana) - Streamlit Web Arayüzü (Ana Uygulama).

Bu dosya, 'docker-compose.yml'deki 'web' servisi tarafından çalıştırılır.
(streamlit run src/services/app.py)

Sorumlulukları:
1.  Kullanıcı arayüzünü (Sidebar, Chat) çizmek.
2.  Dosya yüklemelerini kabul etmek ve 'pending_files' klasörüne kaydetmek.
3.  'src.services.tasks.py' içindeki Celery görevlerini tetiklemek (PDF işleme).
4.  Kalıcı 'volume' klasörlerini (pending, processed, failed) izleyerek
    dosya işleme durumunu kullanıcıya göstermek.
5.  'src.pipeline.rag_chain.py' içindeki (v2.5 RAGAS Uyumlu) RAG zincirini
    ('lazy load' ile) başlatmak ve kullanıcı sorgularını bu zincire göndermek.
6.  RAG zincirinden gelen 'answer' (yanıt) ve 'formatted_context' (kaynaklar)
    verilerini ekrana basmak.
"""

import streamlit as st
import os
import logging
from typing import Dict, Any

# Merkezi yapılandırma, loglama ve ayarlar
# (Dosya 8 - v2.5 Sürümü)
from src.core.config import settings
from src.utils.logging_config import setup_logging

# Asenkron görevler (PDF işleme)
# (Dosya 16)
from src.services.tasks import process_pdf_task

# RAG sorgu boru hattı (YENİ KONUM - Faz 11.1)
# (Dosya 25 - v2.5 RAGAS Uyumlu Sürüm)
from src.pipeline.rag_chain import setup_rag_chain

# --- 1. Loglamayı Başlat ---
setup_logging()
logger = logging.getLogger(__name__)

# --- 2. Sayfa Yapılandırması ---
st.set_page_config(
    page_title="Proje 2main (Tuana)",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🔬 Proje 2main (Tuana) - Genetik Araştırma Asistanı")

# --- 3. Durum Yönetimi (Session State) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! Analiz için lütfen sol panelden PDF'lerinizi yükleyin."}
    ]
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None # 'lazy load' edilecek

# --- 4. Helper Fonksiyon: Dosya Durumlarını Oku ---
def get_file_counts() -> Dict[str, int]:
    """Kalıcı 'volume' dizinlerindeki dosya sayılarını döndürür."""
    try:
        # 'settings' objesinden (Dosya 8) tanımlı yolları kullan
        pending = len([f for f in os.listdir(settings.PENDING_DIR) if f.endswith('.pdf')])
        processed = len([f for f in os.listdir(settings.PROCESSED_DIR) if f.endswith('.pdf')])
        failed = len([f for f in os.listdir(settings.FAILED_DIR) if f.endswith('.pdf')])
        return {"pending": pending, "processed": processed, "failed": failed}
    except FileNotFoundError:
        # Docker volume'leri henüz oluşmamışsa (ilk çalıştırma),
        # 'exist_ok=True' ile oluştur.
        logger.warning("Kalıcı depolama klasörleri bulunamadı, şimdi oluşturuluyor...")
        os.makedirs(settings.PENDING_DIR, exist_ok=True)
        os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
        os.makedirs(settings.FAILED_DIR, exist_ok=True)
        return {"pending": 0, "processed": 0, "failed": 0}

# --- 5. KENAR ÇUBUĞU (SIDEBAR) ---
with st.sidebar:
    st.header("1. Belge Yükleme")
    st.markdown("PDF dosyalarınızı buraya yükleyin. `Worker` servisi dosyaları otomatik olarak işleyecektir.")
    
    uploaded_files = st.file_uploader(
        "Bilimsel makalelerinizi (PDF) seçin",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        files_added_to_queue = 0
        for uploaded_file in uploaded_files:
            # Dosyayı (container içi) 'pending' klasörüne kaydet
            pending_path = os.path.join(settings.PENDING_DIR, uploaded_file.name)
            processed_path = os.path.join(settings.PROCESSED_DIR, uploaded_file.name)
            failed_path = os.path.join(settings.FAILED_DIR, uploaded_file.name)

            # Eğer dosya zaten işlenmemişse veya beklemiyorsa kuyruğa ekle
            if not (os.path.exists(pending_path) or os.path.exists(processed_path) or os.path.exists(failed_path)):
                try:
                    with open(pending_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # --- CELERY GÖREVİNİ TETİKLE ---
                    logger.info(f"'{uploaded_file.name}' dosyası 'pending' klasörüne eklendi. Celery görevi tetikleniyor...")
                    process_pdf_task.delay(pending_path) # Asenkron görevi (Dosya 16) çağır
                    files_added_to_queue += 1
                    
                except Exception as e:
                    logger.error(f"'{uploaded_file.name}' dosyası kuyruğa eklenemedi: {e}", exc_info=True)
                    st.sidebar.error(f"'{uploaded_file.name}' kuyruğa eklenemedi: {e}")
            
            # (Dosya zaten varsa bir şey yapma)
        
        if files_added_to_queue > 0:
            st.sidebar.success(f"{files_added_to_queue} adet yeni dosya işlem kuyruğuna eklendi.")
            st.rerun() # Sayfayı yenileyerek dosya sayaçlarını güncelle

    st.divider()

    # --- Dosya Durum Paneli ---
    st.header("Dosya İşleme Durumu")
    file_counts = get_file_counts()
    st.info(f"Kuyrukta Bekleyen: {file_counts['pending']}")
    st.success(f"İşlenmiş (Hazır): {file_counts['processed']}")
    if file_counts['failed'] > 0:
        st.error(f"İşlenemeyen (Hatalı): {file_counts['failed']}")
    
    if st.button("Durumu Yenile"):
        st.rerun()

    st.divider()

    # --- Arama Filtreleri ---
    st.header("2. RAG Sorgu Filtreleri")
    section_filter = st.selectbox(
        "Aramayı belirli bir bölümle sınırla:",
        (
            "Tüm Bölümler", 
            "Abstract", "Giriş", "Introduction", 
            "Methods", "Yöntemler", "Methodology",
            "Results", "Bulgular", "Sonuçlar",
            "Discussion", "Tartışma", "Conclusion", "Unknown"
        ),
        key="section_filter" # Seçimin 'session_state'de saklanmasını sağlar
    )

# --- 6. ANA SOHBET ARAYÜZÜ ---

# 6.1. Sohbet geçmişini göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6.2. Yeni kullanıcı girdisi al
if prompt := st.chat_input("İşlenmiş belgeler hakkında bir soru sorun..."):
    
    # Kullanıcı mesajını geçmişe ekle ve göster
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 6.3. Asistan yanıtını oluştur
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyorum... (RAG zinciri çalışıyor, Re-ranker sıralıyor, Gemini yanıtlıyor...)"):
            try:
                # RAG zincirini 'lazy load' et (eğer yüklü değilse)
                if st.session_state.rag_chain is None:
                    # Sorgulamadan önce en az bir dosyanın işlenmiş olması gerekir
                    if get_file_counts()['processed'] == 0:
                        st.warning("Sorgulama yapmadan önce lütfen en az bir belgenin işlenmesini bekleyin.")
                        st.stop()
                    
                    logger.info("RAG Zinciri ilk defa başlatılıyor... (setup_rag_chain çağrılıyor)")
                    # 'rag_pipeline.py'deki (Dosya 25) ana fonksiyonu çağır
                    st.session_state.rag_chain = setup_rag_chain()
                    
                    if st.session_state.rag_chain is None:
                        st.error("Kritik Hata: RAG Zinciri başlatılamadı. Lütfen sunucu loglarını kontrol edin.")
                        logger.critical("setup_rag_chain() 'None' döndürdü. app.py durduruluyor.")
                        st.stop()
                        
                    logger.info("RAG Zinciri başarıyla kuruldu ve 'session_state'e yüklendi.")

                # --- RAG ZİNCİRİNİ ÇAĞIR ---
                rag_input = {
                    "question": prompt,
                    "section": st.session_state.section_filter # Sidebar'dan gelen filtre
                }
                logger.info(f"RAG Zinciri çağrılıyor: {rag_input}")
                
                # 'setup_rag_chain'de (Dosya 25) tanımladığımız 'RunnableParallel' zincirini çağırıyoruz.
                # Bu, {'answer': ..., 'context_docs': ..., 'formatted_context': ...} içeren bir dict döndürecek.
                response: Dict[str, Any] = st.session_state.rag_chain.invoke(rag_input)
                
                # YENİ (v2.5) RAG ÇIKTILARI:
                answer = response.get("answer", "Yanıt alınamadı.")
                # Streamlit'in 'expander'ı için formatlanmış metni al:
                context_for_display = response.get("formatted_context", "Kaynak bilgisi bulunamadı.")
                
                # Yanıtı ekrana yaz
                st.markdown(answer)
                
                # Kaynakları (context) gizli bir expander içinde göster
                with st.expander("Görüntülenen yanıtın kaynakları (bağlamı)"):
                    st.info(context_for_display)

                # Yanıtı sohbet geçmişine ekle
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                logger.critical(f"RAG Zinciri sorgusunda HATA oluştu: {e}", exc_info=True)
                st.error(f"Sorgu işlenirken bir hata oluştu: {e}")

