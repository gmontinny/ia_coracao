"""
Testes unitários para o processamento de imagens, sinais de ECG e detecção YOLO de derivações.
"""

import os
from pathlib import Path
import pytest
import numpy as np

from src.models.ecg_signal_processing import ECGSignalProcessor
from src.models.ecg_vision import ECGDeepVisionNet
from src.models.ecg_yolo import ECGYoloDetector


def test_ecg_signal_processor_with_images():
    img_dir = Path("img")
    images = list(img_dir.glob("*.*"))
    assert len(images) > 0

    processor = ECGSignalProcessor(sampling_rate_hz=500)
    sample_img_path = str(images[0])

    res = processor.process_ecg_image(sample_img_path)
    assert 'signal_1d' in res
    assert 'peaks' in res
    assert 'metrics' in res
    assert 'grid_removed_gray' in res

    metrics = res['metrics']
    assert 'heart_rate_bpm' in metrics
    assert 30 <= metrics['heart_rate_bpm'] <= 220
    assert 'st_slope' in metrics
    assert 'rhythm_classification' in metrics
    assert 'risk_score' in metrics


def test_ecg_deep_vision_net():
    img_dir = Path("img")
    images = list(img_dir.glob("*.*"))
    assert len(images) > 0

    vision_net = ECGDeepVisionNet()
    sample_img_path = str(images[0])

    processor = ECGSignalProcessor()
    loaded_img = processor.load_image(sample_img_path)

    res = vision_net.predict(loaded_img)
    assert 'prediction' in res
    assert 'prob_disease' in res
    assert 'gradcam_overlay_rgb' in res
    assert 'confidence_interval_95' in res
    assert 0.0 <= res['prob_disease'] <= 1.0


def test_ecg_yolo_detector():
    img_dir = Path("img")
    images = list(img_dir.glob("*.*"))
    assert len(images) > 0

    processor = ECGSignalProcessor()
    loaded_img = processor.load_image(str(images[0]))

    yolo = ECGYoloDetector()
    leads = yolo.detect_leads(loaded_img)
    assert len(leads) > 0
    assert 'bbox' in leads[0]
    assert 'lead' in leads[0]

    waves = yolo.detect_wave_peaks_bboxes(loaded_img, r_peaks_x=[50, 150, 250])
    assert len(waves) > 0

    annotated = yolo.draw_detections(loaded_img, leads, waves)
    assert annotated.shape == loaded_img.shape

    crops = yolo.crop_leads(loaded_img, leads)
    assert len(crops) > 0


def test_giovanny_01_heart_rate_and_header_extraction():
    """Valida a calibração de FC e extração de metadados em giovanny_01.png (FC ~84 bpm)."""
    img_path = Path("img/giovanny_01.png")
    if not img_path.exists():
        pytest.skip("img/giovanny_01.png não encontrado")

    processor = ECGSignalProcessor(sampling_rate_hz=500)
    res = processor.process_ecg_image(str(img_path))
    metrics = res['metrics']
    meta = res['header_metadata']
    
    # FC deve ser 84 bpm (ou muito próxima) e não mais 160 bpm
    assert abs(metrics['heart_rate_bpm'] - 84.0) <= 2.0
    assert metrics['rhythm_classification'] == 'Ritmo Sinusal Regular'
    assert metrics['st_condition'] == 'Normal / Isoelétrico Fisiológico'
    assert metrics['risk_score'] < 0.35
    assert meta.get('patient_name') is not None


def test_giovanny_03_telemetry_and_fc_extraction():
    """Valida a extração de telemetria rica (FC 95 bpm, P, PR, QRS, QT, QTc, SAP, SAQRS) em giovanny_03.jpg."""
    img_path = Path("img/giovanny_03.jpg")
    if not img_path.exists():
        pytest.skip("img/giovanny_03.jpg não encontrado")

    processor = ECGSignalProcessor(sampling_rate_hz=500)
    res = processor.process_ecg_image(str(img_path))
    metrics = res['metrics']
    meta = res['header_metadata']
    
    # Validações estritas de telemetria
    assert abs(metrics['heart_rate_bpm'] - 95.0) <= 1.0
    assert meta.get('header_heart_rate_bpm') == 95.0
    assert meta.get('patient_name') is not None
    assert meta.get('patient_age') == 50
    assert meta.get('qt_ms') is not None
    assert meta.get('qtc_ms') is not None
    assert meta.get('electrical_axis_deg') is not None or meta.get('saqrs_axis_deg') is not None
    assert meta.get('doctor_name') is not None or meta.get('doctor_rqe') == '2518'
    assert metrics['st_condition'] == 'Normal / Isoelétrico Fisiológico'
    assert metrics['risk_score'] < 0.35
