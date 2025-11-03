"""
Merkezi Loglama Yapılandırma Modülü.

Bu modül, tüm servisler (Streamlit, Celery) tarafından çağrılmak üzere
standart bir loglama yapılandırması sağlar.

Kullanımı:
Herhangi bir modülün en başına şu satırı ekleyin:
    from src.utils.logging_config import setup_logging
    setup_logging()
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Bu standart bir log mesajıdır.")
"""

import logging
import sys

# Loglama yapılandırmasının birden fazla kez çalışmasını engellemek için
# bir 'flag' (bayrak) kullanıyoruz.
_logging_configured = False

def setup_logging():
    """
    Proje geneli için standart loglama formatını ayarlar.
    Bu fonksiyon her servisin (app.py, tasks.py) en başında çağrılmalıdır.
    """
    global _logging_configured
    if _logging_configured:
        return

    # Log formatı: ZAMAN - MODÜL ADI - SEVİYE - MESAJ
    log_format = (
        "%(asctime)s - [%(levelname)s] - [%(name)s.%(funcName)s:%(lineno)d] - %(message)s"
    )
    
    # Tüm logları 'stdout' (standart çıktı) üzerine yönlendiriyoruz.
    # Docker Compose, bu çıktıları toplayıp 'docker-compose logs' ile
    # görmemizi sağlayacak. Dosyaya loglama yapmaya gerek yok.
    logging.basicConfig(
        level=logging.INFO,  # Log seviyesi (INFO, DEBUG, WARNING, ERROR)
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Bazı kütüphanelerin (örn. 'httpx') çok fazla 'gürültülü' log
    # basmasını engellemek için seviyelerini yükseltiyoruz.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    _logging_configured = True
    
    logger = logging.getLogger(__name__)
    logger.info("Loglama sistemi başarıyla yapılandırıldı.")
