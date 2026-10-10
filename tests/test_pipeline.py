"""
Testes de integração para o pipeline unificado, detecções YOLO e gerador de laudos.
"""

import os
from pathlib import Path
import pytest
import pandas as pd

from src.pipeline.ecg_pipeline import UnifiedECGPipeline
from src.data.loader import load_heart_dataset


def test_unified_pipeline_image():
    pipeline = UnifiedECGPipeline(auto_train_tabular=True)
    
    img_dir = Path("img")
    images = list(img_dir.glob("*.*"))
    assert len(images) > 0

    patient_meta = {
        'Age': 58,
        'Sex': 'M',
        'ChestPainType': 'ASY',
        'RestingBP': 150,
        'Cholesterol': 240,
        'FastingBS': 0,
        'ExerciseAngina': 'Y'
    }

    res = pipeline.analyze_image_ecg(str(images[0]), patient_metadata=patient_meta)
    assert 'signal_processing' in res
    assert 'yolo_detections' in res
    assert 'vision_deep_learning' in res
    assert 'multimodal_prediction' in res
    assert 'clinical_report' in res
    assert 'full_report_text' in res['clinical_report']
    
    assert 'leads' in res['yolo_detections']
    assert 'annotated_image' in res['yolo_detections']
    assert 'confidence_interval_95' in res['multimodal_prediction']


def test_unified_pipeline_normal_ecgs():
    pipeline = UnifiedECGPipeline(auto_train_tabular=True)
    
    # Giovanny 01 e 02 devem ser classificados como baixo risco
    for name in ['giovanny_01.png', 'giovanny_02.png']:
        p = Path("img") / name
        if p.exists():
            res = pipeline.analyze_image_ecg(str(p))
            assert res['multimodal_prediction']['risk_score'] < 0.35
            assert res['multimodal_prediction']['risk_tier'] == 'Baixo Risco'
            assert res['signal_processing']['metrics']['is_st_abnormal'] is False


def test_unified_pipeline_batch_df():
    pipeline = UnifiedECGPipeline(auto_train_tabular=True)
    df = load_heart_dataset(treated=True).head(20)

    df_out = pipeline.analyze_batch_dataframe(df)
    assert 'Pred_HeartDisease' in df_out.columns
    assert 'Prob_Risco_Cardiovascular' in df_out.columns
    assert 'Classificacao_Risco' in df_out.columns
    assert len(df_out) == 20
