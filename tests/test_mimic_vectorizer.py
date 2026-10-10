"""
Testes Unitários e de Integração para o Vetorizador MIMIC-IV-ECG e Base Vetorial Qdrant.
Valida geração de embeddings de biossinais, streaming em lotes e busca de casos clínicos (k-NN).
"""

import pytest
import numpy as np
from src.data.mimic_vectorizer import MIMICIVVectorizer
from src.data.ecg_knowledge_rag import ECGKnowledgeVectorDB


def test_mimic_vectorizer_initialization():
    vectorizer = MIMICIVVectorizer()
    assert vectorizer is not None
    assert vectorizer.vector_size == 64
    assert vectorizer.COLLECTION_NAME == "mimic_iv_ecg_800k"
    
    # Valida estrutura dos protótipos de referência
    assert len(vectorizer.PROTOTYPE_RECORDS) >= 5
    first_record = vectorizer.PROTOTYPE_RECORDS[0]
    assert "ecg_id" in first_record
    assert "diagnostic_superclass" in first_record
    assert "coronary_occlusion" in first_record
    assert "clinical_outcome" in first_record


def test_mimic_vector_encoding_and_normalization():
    vectorizer = MIMICIVVectorizer()
    
    sample_record = {
        "age": 65,
        "gender": "M",
        "heart_rate": 110,
        "st_deviation_mm": 3.5,
        "sdnn_ms": 25.0,
        "in_icu": True,
        "troponin_i_ng_ml": 15.0,
        "diagnostic_superclass": "MI",
        "diagnostic_statements": ["Acute anterior myocardial infarction", "ST elevation V1-V4"],
        "keywords": ["stemi", "infarto", "anterior"]
    }
    
    vec = vectorizer.encode_record(sample_record)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (64,)
    
    # Verifica que o vetor é normalizado (norma L2 = 1.0)
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_generate_synthetic_mimic_batch():
    vectorizer = MIMICIVVectorizer()
    batch = vectorizer.generate_synthetic_mimic_batch(batch_size=50, start_id=200000)
    
    assert len(batch) == 50
    for record in batch:
        assert record["ecg_id"].startswith("MIMIC-")
        assert "age" in record and 20 <= record["age"] <= 90
        assert "heart_rate" in record and 40 <= record["heart_rate"] <= 190
        assert record["diagnostic_superclass"] in ["NORM", "MI", "STTC", "CD", "HYP", "AFIB"]
        assert isinstance(record["in_icu"], bool)
        assert "clinical_outcome" in record


def test_ingest_mimic_records_and_progress_callback():
    vectorizer = MIMICIVVectorizer()
    
    progress_calls = []
    def callback(current, total, rate):
        progress_calls.append((current, total, rate))
        
    result = vectorizer.ingest_mimic_records(
        total_records=100,
        batch_size=25,
        recreate_collection=False,
        progress_callback=callback
    )
    
    assert result["success"] is True
    assert result["records_indexed"] == 100
    assert result["vector_dimension"] == 64
    assert result["elapsed_seconds"] >= 0.0
    assert result["throughput_records_per_sec"] > 0.0
    assert len(progress_calls) >= 4


def test_search_similar_cases_string_and_dict():
    vectorizer = MIMICIVVectorizer()
    
    # Busca por string textual
    results_stemi = vectorizer.search_similar_cases("Acute anterior STEMI supradesnivelamento V1-V4", top_k=2)
    assert len(results_stemi) == 2
    assert any(r["diagnostic_superclass"] in ["MI", "STTC"] for r in results_stemi)
    assert "similarity_percentage" in results_stemi[0]
    
    # Busca por exame normal
    results_norm = vectorizer.search_similar_cases("Normal sinus rhythm ECG fisiologico normal", top_k=2)
    assert len(results_norm) == 2
    assert any(r["diagnostic_superclass"] == "NORM" for r in results_norm)
    
    # Busca por dicionário estruturado
    results_dict = vectorizer.search_similar_cases({
        "diagnostic_superclass": "AFIB",
        "heart_rate": 140,
        "st_deviation_mm": -0.8
    }, top_k=2)
    assert len(results_dict) == 2


def test_rag_knowledge_integration_with_mimic():
    rag_engine = ECGKnowledgeVectorDB()
    mimic_cases = rag_engine.search_mimic_cases("STEMI anterior oclusao ADA", top_k=2)
    
    assert isinstance(mimic_cases, list)
    assert len(mimic_cases) <= 2
    if mimic_cases:
        assert "ecg_id" in mimic_cases[0]
        assert "clinical_outcome" in mimic_cases[0]
