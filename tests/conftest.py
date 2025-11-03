    """
Pytest Merkezi Fixture (Yardımcı) Dosyası - conftest.py

Bu dosya, 'tests/' dizinindeki tüm testler tarafından paylaşılan
'fixture'ları (test verileri, sahte objeler vb.) tanımlar.

Pytest bu dosyayı otomatik olarak bulur ve 'fixture'ları
test fonksiyonlarına enjekte eder.
"""

import pytest
import sys
import os
from langchain.docstore.document import Document
from typing import List

# 'src' paketini testlerin bulabilmesi için
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

@pytest.fixture(scope="session")
def sample_document_list() -> List[Document]:
    """
    Tüm testlerde kullanılabilecek, 'Document' objelerinden oluşan
    sahte (mock) bir veritabanı sağlar.
    """
    doc1 = Document(
        page_content="CRISPR-Cas9 is a genome editing tool. It uses a Cas9 nuclease to create double-strand breaks.",
        metadata={"source": "article1.pdf", "page": 1, "section": "Introduction"}
    )
    doc2 = Document(
        page_content="The methodology involves a guide RNA (gRNA) that directs Cas9 to the target DNA sequence.",
        metadata={"source": "article1.pdf", "page": 2, "section": "Methods"}
    )
    doc3 = Document(
        page_content="We discuss the ethical implications of genetic engineering.",
        metadata={"source": "article2.pdf", "page": 5, "section": "Discussion"}
    )
    return [doc1, doc2, doc3]
