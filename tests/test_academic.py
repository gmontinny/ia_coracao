"""
Testes unitários para utilitários acadêmicos e exportação de artigos.
"""

import pytest
import pandas as pd
from src.utils.academic_metrics import generate_latex_table, generate_markdown_table
from src.models.llm_report import ECGClinicalReportGenerator


def test_academic_latex_and_markdown():
    dummy_df = pd.DataFrame([{
        'Modelo': 'XGBoost',
        'Acurácia (Teste)': 0.885,
        'Acurácia CV (Média ± DP)': '0.872 ± 0.021',
        'Sensibilidade / Recall': 0.912,
        'Especificidade': 0.854,
        'Precisão': 0.880,
        'F1-Score': 0.895,
        'ROC-AUC': 0.942,
        'PR-AUC': 0.938,
        'Brier Score': 0.098
    }])

    latex = generate_latex_table(dummy_df)
    assert "\\begin{table*}" in latex
    assert "XGBoost" in latex

    md = generate_markdown_table(dummy_df)
    assert "XGBoost" in md


def test_llm_academic_report():
    gen = ECGClinicalReportGenerator()
    article_sec = gen.generate_academic_article_section()
    assert "Methodology" in article_sec
    assert "Gradient Boosting" in article_sec
