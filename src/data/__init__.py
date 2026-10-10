"""
Módulo de carregamento e pré-processamento de dados clínicos e cardiológicos.
"""

from .loader import (
    load_heart_dataset,
    split_heart_data,
    build_preprocessor,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN
)
from .ecg_knowledge_rag import ECGKnowledgeVectorDB
from .hf_dataset_loader import HuggingFaceECGHub
from .mimic_vectorizer import MIMICIVVectorizer

__all__ = [
    "load_heart_dataset",
    "split_heart_data",
    "build_preprocessor",
    "CATEGORICAL_FEATURES",
    "NUMERICAL_FEATURES",
    "TARGET_COLUMN",
    "ECGKnowledgeVectorDB",
    "HuggingFaceECGHub",
    "MIMICIVVectorizer"
]
