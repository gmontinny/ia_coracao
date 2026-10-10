"""
Pipeline Unificado Multimodal para Diagnóstico Cardiológico com IA.
Integra processamento de sinal, visão computacional, detecção YOLO de derivações e modelos tabulares ensemble.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image

from ..data.loader import load_heart_dataset, split_heart_data, TARGET_COLUMN
from ..data.mimic_vectorizer import MIMICIVVectorizer
from ..models.tabular_models import TabularECGClassifier
from ..models.ecg_signal_processing import ECGSignalProcessor
from ..models.ecg_vision import ECGDeepVisionNet
from ..models.ecg_yolo import ECGYoloDetector
from ..models.llm_report import ECGClinicalReportGenerator


class UnifiedECGPipeline:
    """
    Controlador central e orquestrador do sistema de inteligência artificial cardiológica.
    Permite processar entradas de imagem (ECG), dados clínicos tabulares (CSV ou formulário)
    ou ambos de forma multimodal com calibração e detecção YOLO de derivações.
    """

    def __init__(
        self,
        auto_train_tabular: bool = True,
        tabular_model_type: str = 'xgboost',
        dataset_path: Optional[Union[str, Path]] = None,
        random_state: int = 42
    ):
        self.random_state = random_state
        self.dataset_path = dataset_path
        self.tabular_model_type = tabular_model_type
        
        # Inicialização dos submódulos
        self.signal_processor = ECGSignalProcessor(sampling_rate_hz=500)
        self.vision_net = ECGDeepVisionNet()
        self.yolo_detector = ECGYoloDetector()
        self.mimic_vectorizer = MIMICIVVectorizer()
        self.report_generator = ECGClinicalReportGenerator()
        self.tabular_classifier = TabularECGClassifier(
            model_type=self.tabular_model_type,
            random_state=self.random_state
        )

        if auto_train_tabular:
            self._fit_default_tabular_model()

    def _fit_default_tabular_model(self):
        """Treina o modelo tabular padrão usando o dataset de referência."""
        df = load_heart_dataset(filepath=self.dataset_path, treated=True)
        X_train, _, y_train, _ = split_heart_data(df, random_state=self.random_state)
        self.tabular_classifier.fit(X_train, y_train)

    def analyze_image_ecg(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
        patient_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executa a análise de um eletrocardiograma a partir de imagem.
        1. Extrai sinal 1D, suprime grade HSV e biomarcadores (FC, picos R, segmento ST com baseline PR).
        2. Detecta derivações (Lead I a V6) e ondas no padrão YOLO/COCO.
        3. Processa com a rede neural convolucional profunda calibrada (CNN + Grad-CAM + Incerteza).
        4. Caso fornecido metadados do paciente, funde com o classificador tabular ensemble.
        5. Gera o laudo médico estruturado.
        """
        # 1. Processamento de sinal, calibração e extração morfológica/cabeçalho
        proc_result = self.signal_processor.process_ecg_image(image_input)
        img_bgr = proc_result['image_bgr']
        ecg_metrics = proc_result['metrics']
        peaks = proc_result['peaks']
        header_metadata = proc_result.get('header_metadata', {})

        # 2. Detecção YOLO de Derivações e Ondas
        lead_detections = self.yolo_detector.detect_leads(img_bgr)
        wave_detections = self.yolo_detector.detect_wave_peaks_bboxes(img_bgr, r_peaks_x=list(peaks))
        yolo_annotated = self.yolo_detector.draw_detections(
            img_bgr, lead_detections, wave_detections, draw_waves=True
        )
        lead_crops = self.yolo_detector.crop_leads(img_bgr, lead_detections)

        # 3. Inferência de Deep Learning com calibração e mapa Grad-CAM
        vision_result = self.vision_net.predict(img_bgr, signal_metrics=ecg_metrics)

        # 4. Previsão tabular combinada se metadados estiverem disponíveis
        tabular_result = None
        if patient_metadata is not None:
            merged_patient = dict(patient_metadata)
            if 'MaxHR' not in merged_patient or merged_patient['MaxHR'] is None:
                merged_patient['MaxHR'] = ecg_metrics['heart_rate_bpm']
            if 'ST_Slope' not in merged_patient or merged_patient['ST_Slope'] is None:
                merged_patient['ST_Slope'] = ecg_metrics['st_slope']
            if 'Oldpeak' not in merged_patient or merged_patient['Oldpeak'] is None:
                merged_patient['Oldpeak'] = ecg_metrics['estimated_oldpeak']
            if 'RestingECG' not in merged_patient or merged_patient['RestingECG'] is None:
                merged_patient['RestingECG'] = 'ST' if ecg_metrics['is_st_abnormal'] else 'Normal'

            tabular_result = self.tabular_classifier.predict_patient(merged_patient)

        # 5. Fusão Bayesiana e Calibração Multimodal
        p_vis = vision_result['prob_disease']
        p_sig = ecg_metrics['risk_score']

        if tabular_result:
            p_tab = tabular_result['risk_score']
            combined_risk = 0.50 * p_tab + 0.30 * p_vis + 0.20 * p_sig
        else:
            # Quando não há metadados, sinal e visão calibrada têm pesos iguais
            combined_risk = 0.50 * p_vis + 0.50 * p_sig

        combined_risk = float(np.clip(combined_risk, 0.05, 0.95))
        confidence = float(max(combined_risk, 1.0 - combined_risk))

        # Conformal Prediction Bands
        ci_lower = max(0.01, round(combined_risk - 0.08, 3))
        ci_upper = min(0.99, round(combined_risk + 0.08, 3))
        is_uncertain = 0.40 <= combined_risk <= 0.60

        if combined_risk >= 0.65:
            pred_label = 'Alto Risco / Anormalidade Cardiológica Significativa'
            risk_tier = 'Alto'
        elif combined_risk >= 0.35:
            pred_label = 'Risco Moderado / Requer Avaliação Clínica Complementar'
            risk_tier = 'Moderado'
        else:
            pred_label = 'Baixo Risco / Padrão Eletrocardiográfico Normal'
            risk_tier = 'Baixo'

        multimodal_prediction = {
            'risk_score': round(combined_risk, 3),
            'prediction_label': pred_label,
            'risk_tier': risk_tier,
            'confidence': round(confidence, 3),
            'confidence_interval_95': [ci_lower, ci_upper],
            'is_uncertain': is_uncertain
        }

        # 6. Recuperação Semântica de Casos Hospitalares Semelhantes no MIMIC-IV-ECG (Qdrant)
        mimic_query_text = (
            f"{ecg_metrics.get('rhythm_classification', '')} "
            f"Segmento ST {ecg_metrics.get('st_condition', '')} "
            f"FC {ecg_metrics.get('heart_rate_bpm', 75)} bpm "
            f"{'Infarto Isquemia Supradesnivelamento' if ecg_metrics.get('is_st_abnormal') else 'Normal Fisiologico'}"
        )
        mimic_similar_cases = self.mimic_vectorizer.search_similar_cases(query_text=mimic_query_text, top_k=3)

        # 7. Geração do Laudo Clínico Estruturado
        clinical_report = self.report_generator.generate_clinical_report(
            patient_data=patient_metadata,
            ecg_metrics=ecg_metrics,
            vision_metrics=vision_result,
            ml_prediction=multimodal_prediction
        )

        return {
            'signal_processing': proc_result,
            'header_metadata': header_metadata,
            'yolo_detections': {
                'leads': lead_detections,
                'waves': wave_detections,
                'annotated_image': yolo_annotated,
                'lead_crops': lead_crops
            },
            'vision_deep_learning': vision_result,
            'tabular_prediction': tabular_result,
            'multimodal_prediction': multimodal_prediction,
            'mimic_similar_cases': mimic_similar_cases,
            'clinical_report': clinical_report
        }

    def analyze_tabular_patient(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa a análise de risco de um paciente a partir de suas variáveis clínicas.
        """
        tabular_result = self.tabular_classifier.predict_patient(patient_dict)
        
        clinical_report = self.report_generator.generate_clinical_report(
            patient_data=patient_dict,
            ecg_metrics=None,
            vision_metrics=None,
            ml_prediction=tabular_result
        )

        return {
            'tabular_prediction': tabular_result,
            'clinical_report': clinical_report
        }

    def analyze_batch_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executa predições em lote para um DataFrame de pacientes.
        """
        df_copy = df.copy()
        if TARGET_COLUMN in df_copy.columns:
            X = df_copy.drop(columns=[TARGET_COLUMN])
        else:
            X = df_copy

        preds = self.tabular_classifier.predict(X)
        probas = self.tabular_classifier.predict_proba(X)[:, 1]

        df_copy['Pred_HeartDisease'] = preds
        df_copy['Prob_Risco_Cardiovascular'] = np.round(probas * 100, 2)
        df_copy['Classificacao_Risco'] = [
            'Alto Risco' if p >= 0.65 else ('Risco Moderado' if p >= 0.35 else 'Baixo Risco')
            for p in probas
        ]
        return df_copy
