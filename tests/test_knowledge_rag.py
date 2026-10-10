"""
Testes automatizados para o módulo de Base Vetorial e RAG Cardiológico.
"""

import pytest
from src.data.ecg_knowledge_rag import ECGKnowledgeVectorDB


def test_vector_db_initialization_and_search():
    vector_db = ECGKnowledgeVectorDB()
    
    # Busca por critérios de STEMI / Infarto
    stemi_results = vector_db.search_guidelines("supradesnivelamento de ST infarto agudo", top_k=2)
    assert len(stemi_results) > 0
    assert "relevance_score" in stemi_results[0]
    assert stemi_results[0]["relevance_score"] > 0
    assert "STEMI" in stemi_results[0]["title"] or "Infarto" in stemi_results[0]["title"]


def test_vector_db_clinical_context_retrieval():
    vector_db = ECGKnowledgeVectorDB()

    # Cenário de Paciente com elevação de ST e taquicardia
    ecg_metrics = {
        "is_st_abnormal": True,
        "st_condition": "Elevação Anormal do Segmento ST (Possível Isquemia / Infarto Agudo)",
        "heart_rate_bpm": 125,
        "rhythm_severity": "Grave",
        "sdnn_ms": 32.0
    }
    
    retrieved = vector_db.retrieve_clinical_context(ecg_metrics=ecg_metrics)
    assert len(retrieved) > 0
    
    formatted_citations = vector_db.format_rag_citations(retrieved)
    assert len(formatted_citations) > 0
    assert "Fonte:" in formatted_citations
    assert "Evidência:" in formatted_citations


def test_vector_db_normal_patient_retrieval():
    vector_db = ECGKnowledgeVectorDB()

    # Paciente normal
    ecg_metrics = {
        "is_st_abnormal": False,
        "st_condition": "Morfologia Normal de Segmento ST",
        "heart_rate_bpm": 72,
        "rhythm_severity": "Normal",
        "sdnn_ms": 85.0
    }

    retrieved = vector_db.retrieve_clinical_context(ecg_metrics=ecg_metrics)
    assert len(retrieved) > 0


def test_reference_cases_search():
    vector_db = ECGKnowledgeVectorDB()
    cases = vector_db.search_reference_cases("ritmo sinusal normal giovanny", top_k=1)
    assert len(cases) > 0
    assert "NORM" in cases[0]["ground_truth_diagnosis"] or "Normal" in cases[0]["title"]


def test_qdrant_connectivity_check():
    vector_db = ECGKnowledgeVectorDB()
    is_online, msg = vector_db.check_qdrant_connection()
    assert isinstance(is_online, bool)
    assert isinstance(msg, str)
