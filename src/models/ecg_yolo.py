"""
Módulo de Detecção de Objetos Eletrocardiográficos, Derivações e Ondas (YOLO / COCO Standard).
Permite segmentar derivações (Lead I a V6 + D2 longo), complexos QRS, ondas P e T em imagens de ECG.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image


class ECGYoloDetector:
    """
    Detector de derivações e estruturas morfológicas de ECG baseado nos padrões YOLO / COCO.
    Detecta caixas delimitadoras (bounding boxes) de:
    - 12 Derivações Padrão: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    - DII Longo (Rhythm Strip)
    - Complexos QRS, Ondas P e Ondas T
    - Grade Milimetrada e Escala de Calibração (10mm/mV, 25mm/s)
    """

    LEAD_NAMES_12 = [
        "I", "II", "III",
        "aVR", "aVL", "aVF",
        "V1", "V2", "V3",
        "V4", "V5", "V6"
    ]

    def __init__(self, confidence_threshold: float = 0.60):
        self.confidence_threshold = confidence_threshold

    # Layout padrão: ordem das derivações por coluna (4 colunas × 3 linhas)
    # Coluna 0: I, aVR, V1, V4 | Coluna 1: II, aVL, V2, V5 | etc.
    _LEAD_ORDER_4COL = [
        ["I",   "II",  "III"],
        ["aVR", "aVL", "aVF"],
        ["V1",  "V2",  "V3"],
        ["V4",  "V5",  "V6"],
    ]

    def _estimate_signal_density(self, roi: np.ndarray) -> float:
        """Estima a densidade de traçado em uma região (0.0 a 1.0) via pixels escuros."""
        if roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        dark = np.sum(gray < 100)
        return float(dark) / roi.size

    def detect_leads(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Segmenta as derivações do ECG por geometria de layout padrão (3×4 + tira de ritmo).
        A confiança é estimada pela densidade de traçado em cada região detectada,
        não por valores aleatórios.

        Nota: Esta é uma segmentação geométrica baseada em layout padrão IEC 60601-2-51.
        Para detecção por rede neural real, é necessário fine-tune de YOLOv8 em dataset anotado.

        Retorna lista de caixas no formato YOLO/COCO:
        [{'lead': 'I', 'bbox': [x1, y1, x2, y2], 'confidence': 0.94, 'method': 'geometric'}, ...]
        """
        h, w = img_bgr.shape[:2]
        detections = []

        is_12lead_format = h > 400 and w > 400

        if is_12lead_format:
            main_h = int(h * 0.75)
            rhythm_h = h - main_h
            n_rows, n_cols = 3, 4
            row_step = main_h // n_rows
            col_step = w // n_cols

            for col_idx, col_leads in enumerate(self._LEAD_ORDER_4COL):
                for row_idx, lead_name in enumerate(col_leads):
                    x1 = col_idx * col_step
                    y1 = row_idx * row_step
                    x2 = (col_idx + 1) * col_step
                    y2 = (row_idx + 1) * row_step

                    pad_x = int(col_step * 0.03)
                    pad_y = int(row_step * 0.05)
                    bx1 = max(0, x1 + pad_x)
                    by1 = max(0, y1 + pad_y)
                    bx2 = min(w, x2 - pad_x)
                    by2 = min(main_h, y2 - pad_y)

                    roi = img_bgr[by1:by2, bx1:bx2]
                    density = self._estimate_signal_density(roi)
                    # Confiança baseada em densidade: traçado típico tem 2-8% de pixels escuros
                    confidence = float(np.clip(0.70 + density * 4.0, 0.70, 0.98))

                    class_id = col_idx * n_rows + row_idx
                    detections.append({
                        'lead': lead_name,
                        'class_id': class_id,
                        'bbox': [bx1, by1, bx2, by2],
                        'confidence': round(confidence, 3),
                        'category': 'lead',
                        'method': 'geometric_layout'
                    })

            # Tira de Ritmo (Lead II longo)
            rs_x1 = int(w * 0.02)
            rs_y1 = main_h + int(rhythm_h * 0.05)
            rs_x2 = int(w * 0.98)
            rs_y2 = h - int(rhythm_h * 0.05)
            rs_roi = img_bgr[rs_y1:rs_y2, rs_x1:rs_x2]
            rs_density = self._estimate_signal_density(rs_roi)
            rs_confidence = float(np.clip(0.75 + rs_density * 4.0, 0.75, 0.99))

            detections.append({
                'lead': 'II (Longo / Ritmo)',
                'class_id': 12,
                'bbox': [rs_x1, rs_y1, rs_x2, rs_y2],
                'confidence': round(rs_confidence, 3),
                'category': 'rhythm_strip',
                'method': 'geometric_layout'
            })
        else:
            # Traçado de tira única / monitor
            density = self._estimate_signal_density(img_bgr)
            confidence = float(np.clip(0.75 + density * 4.0, 0.75, 0.98))
            detections.append({
                'lead': 'Lead II / Tira Única',
                'class_id': 0,
                'bbox': [0, 0, w, h],
                'confidence': round(confidence, 3),
                'category': 'single_lead',
                'method': 'geometric_layout'
            })

        return detections

    def detect_wave_peaks_bboxes(
        self,
        img_bgr: np.ndarray,
        r_peaks_x: List[int],
        sampling_rate: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Gera caixas delimitadoras estilo YOLO para os complexos QRS, ondas P e ondas T
        a partir das coordenadas detectadas.
        """
        h, w = img_bgr.shape[:2]
        boxes = []

        for idx, peak_x in enumerate(r_peaks_x):
            # QRS Complex Box: ~80-120ms de largura centrado no pico R
            qrs_w = max(16, int(w * 0.03))
            qrs_x1 = max(0, peak_x - qrs_w // 2)
            qrs_x2 = min(w, peak_x + qrs_w // 2)
            qrs_roi = img_bgr[int(h * 0.15):int(h * 0.85), qrs_x1:qrs_x2]
            qrs_conf = float(np.clip(0.80 + self._estimate_signal_density(qrs_roi) * 3.0, 0.80, 0.99))

            boxes.append({
                'label': f'QRS_{idx+1}',
                'class_name': 'QRS_Complex',
                'class_id': 100,
                'bbox': [qrs_x1, int(h * 0.15), qrs_x2, int(h * 0.85)],
                'confidence': round(qrs_conf, 3)
            })

            # T Wave Box: ~100-250ms após pico R
            t_offset = int(w * 0.06)
            t_w = max(20, int(w * 0.04))
            t_x1 = min(w - 1, peak_x + t_offset - t_w // 2)
            t_x2 = min(w, peak_x + t_offset + t_w // 2)
            if t_x2 > t_x1 + 5:
                t_roi = img_bgr[int(h * 0.35):int(h * 0.75), t_x1:t_x2]
                t_conf = float(np.clip(0.70 + self._estimate_signal_density(t_roi) * 3.5, 0.70, 0.97))
                boxes.append({
                    'label': f'T_{idx+1}',
                    'class_name': 'T_Wave',
                    'class_id': 101,
                    'bbox': [t_x1, int(h * 0.35), t_x2, int(h * 0.75)],
                    'confidence': round(t_conf, 3)
                })

            # P Wave Box: ~120-200ms antes do pico R
            p_offset = int(w * 0.06)
            p_w = max(16, int(w * 0.03))
            p_x1 = max(0, peak_x - p_offset - p_w // 2)
            p_x2 = max(0, peak_x - p_offset + p_w // 2)
            if p_x2 > p_x1 + 5:
                p_roi = img_bgr[int(h * 0.40):int(h * 0.70), p_x1:p_x2]
                p_conf = float(np.clip(0.65 + self._estimate_signal_density(p_roi) * 3.5, 0.65, 0.95))
                boxes.append({
                    'label': f'P_{idx+1}',
                    'class_name': 'P_Wave',
                    'class_id': 102,
                    'bbox': [p_x1, int(h * 0.40), p_x2, int(h * 0.70)],
                    'confidence': round(p_conf, 3)
                })

        return boxes

    def draw_detections(
        self,
        img_bgr: np.ndarray,
        lead_detections: List[Dict[str, Any]],
        wave_detections: Optional[List[Dict[str, Any]]] = None,
        draw_waves: bool = True
    ) -> np.ndarray:
        """
        Desenha as caixas delimitadoras estilo YOLO / COCO na imagem de ECG.
        """
        annotated = img_bgr.copy()

        # Cores padrão para classes (BGR)
        COLOR_LEAD = (255, 120, 0)      # Azul / Ciano
        COLOR_RHYTHM = (0, 200, 255)    # Amarelo
        COLOR_QRS = (0, 0, 255)         # Vermelho
        COLOR_T = (0, 180, 0)           # Verde
        COLOR_P = (255, 0, 200)         # Magenta

        # Desenha as derivações
        for det in lead_detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['lead']} ({det['confidence']*100:.0f}%)"
            color = COLOR_RHYTHM if det.get('category') == 'rhythm_strip' else COLOR_LEAD

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Tag de texto com fundo preenchido
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Desenha as ondas morfológicas (QRS, P, T) se solicitado
        if draw_waves and wave_detections:
            for wdet in wave_detections:
                x1, y1, x2, y2 = wdet['bbox']
                cname = wdet['class_name']
                if cname == 'QRS_Complex':
                    color = COLOR_QRS
                elif cname == 'T_Wave':
                    color = COLOR_T
                else:
                    color = COLOR_P

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
                cv2.putText(
                    annotated,
                    wdet['label'],
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA
                )

        return annotated

    def crop_leads(self, img_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Recorta cada derivação detectada individualmente para análise morfológica dedicada.
        """
        crops = {}
        h, w = img_bgr.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))

            if x2 > x1 + 10 and y2 > y1 + 10:
                crop = img_bgr[y1:y2, x1:x2]
                crops[det['lead']] = crop

        return crops
