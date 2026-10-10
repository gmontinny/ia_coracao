"""
Testes automatizados para o Hugging Face ECG Hub e Catálogo de Biossinais.
"""

import pytest
import pandas as pd
from src.data.hf_dataset_loader import HuggingFaceECGHub


def test_hf_hub_catalog_and_filtering():
    hub = HuggingFaceECGHub()
    all_datasets = hub.get_all_datasets()
    assert len(all_datasets) >= 10, "Deveria haver múltiplos datasets catalogados."

    # Filtro por 12 derivações
    leads_12 = hub.filter_datasets(leads=12)
    assert len(leads_12) > 0
    assert all(d["leads"] == 12 for d in leads_12)

    # Filtro por 1 derivação
    leads_1 = hub.filter_datasets(leads=1)
    assert len(leads_1) > 0
    assert all(d["leads"] == 1 for d in leads_1)

    # Filtro por texto
    ptb_xl = hub.filter_datasets(search_query="ptb-xl")
    assert len(ptb_xl) == 1
    assert "PTB-XL" in ptb_xl[0]["name"]


def test_hf_hub_statistics_and_scp_ontology():
    hub = HuggingFaceECGHub()
    stats = hub.get_summary_statistics()

    assert stats["total_datasets_cataloged"] >= 10
    assert stats["total_international_samples"] > 1000000
    assert stats["total_countries"] >= 5
    assert 12 in stats["leads_breakdown"]

    # Mapeamento SCP-ECG
    norm_info = hub.map_scp_ecg_diagnostic("NORM")
    assert norm_info["category"] == "Fisiológico"
    assert norm_info["risk_weight"] < 0.20

    mi_info = hub.map_scp_ecg_diagnostic("MI")
    assert mi_info["category"] == "Isquêmico / Coronário"
    assert mi_info["risk_weight"] >= 0.85


def test_hf_hub_cross_dataset_benchmark_table_and_latex():
    hub = HuggingFaceECGHub()
    bench_df = hub.generate_cross_dataset_benchmark_table()
    assert isinstance(bench_df, pd.DataFrame)
    assert len(bench_df) >= 3
    assert "AUC-ROC" in bench_df.columns
    assert "Sensibilidade (%)" in bench_df.columns

    latex_code = hub.generate_benchmark_latex()
    assert "\\begin{table*}" in latex_code
    assert "PTB-XL" in latex_code
    assert "\\end{table*}" in latex_code


def test_hf_hub_token_handling_and_validation():
    hub = HuggingFaceECGHub(token="hf_test_custom_token_123456789")
    assert hub.get_token() == "hf_test_custom_token_123456789"
    assert hub.mask_token().startswith("hf_test")
    assert hub.mask_token().endswith("6789")

    # Atualização dinâmica de token
    hub.set_token("hf_new_token_abcdefghijk")
    assert hub.get_token() == "hf_new_token_abcdefghijk"

    # Validação de token: testa apenas a estrutura do retorno sem fazer requisição HTTP real
    # (evita falha em CI/CD sem conexão com internet)
    from unittest.mock import patch, MagicMock
    mock_response = MagicMock()
    mock_response.status = 401
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", side_effect=Exception("Sem conexão")):
        token_status = hub.validate_hf_token()
        assert isinstance(token_status, dict)
        assert "valid" in token_status
        assert "message" in token_status
        assert token_status["valid"] is False
