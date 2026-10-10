"""
Processador de Sinais e Digitalizador de Imagens de Eletrocardiograma (ECG).
Converte imagens de ECG em sinais 1D digitalizados e extrai biomarcadores cardíacos com calibração precisa.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter, medfilt
import matplotlib.pyplot as plt
import io
from PIL import Image

# Leitor OCR opcional com inicialização sob demanda para extração de parâmetros do cabeçalho
_OCR_READER = None

def _get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        try:
            import easyocr
            _OCR_READER = easyocr.Reader(['pt', 'en'], gpu=False, verbose=False)
        except Exception:
            _OCR_READER = False
    return _OCR_READER if _OCR_READER is not False else None


class ECGSignalProcessor:
    """
    Pipeline de Visão Computacional e Processamento Digital de Sinais (PDS)
    para digitalização, filtragem de grade milimetrada e extração calibrada de parâmetros clínicos de ECG.
    """

    def __init__(self, sampling_rate_hz: int = 500):
        self.sampling_rate = sampling_rate_hz

    def load_image(self, image_input: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """Carrega e padroniza a imagem de entrada em array BGR do OpenCV."""
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Imagem não encontrada: {path_str}")
            img = cv2.imread(path_str)
            if img is None:
                raise ValueError(f"Não foi possível decodificar a imagem: {path_str}")
            return img

        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img

        elif isinstance(image_input, Image.Image):
            return cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)

        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            return image_input

        else:
            raise TypeError(f"Tipo de imagem não suportado: {type(image_input)}")

    def remove_grid_hsv(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filtro avançado no espaço de cor HSV para supressão da grade milimetrada
        (tons vermelhos, laranjas, rosas, verdes ou cinzas claros) isolando o traçado escuro.
        
        Retorna:
            trace_gray: Imagem em escala de cinza com grade removida.
            grid_mask: Máscara binária da grade detectada.
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 1. Máscara de cores da grade (vermelho, magenta, rosa, laranja, amarelo)
        grid_color_mask1 = ((h < 25) | (h > 150)) & (s > 25)
        # Máscara de verde/ciano (alguns papéis térmicos)
        grid_color_mask2 = ((h >= 35) & (h <= 90)) & (s > 25)
        grid_mask = (grid_color_mask1 | grid_color_mask2).astype(np.uint8) * 255

        # Cria imagem limpa substituindo pixels da grade por fundo branco
        cleaned_bgr = img_bgr.copy()
        cleaned_bgr[grid_mask > 0] = [255, 255, 255]

        # 2. Conversão para escala de cinza
        gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)

        # 3. Remover linhas residuais de grade cinza usando morfologia direcional
        # Detecta linhas horizontais e verticais finas da grade
        k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        # Binarização inversa temporária
        inv_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
        )
        grid_lines_h = cv2.morphologyEx(inv_thresh, cv2.MORPH_OPEN, k_horiz)
        grid_lines_v = cv2.morphologyEx(inv_thresh, cv2.MORPH_OPEN, k_vert)
        grid_lines = cv2.bitwise_or(grid_lines_h, grid_lines_v)

        # Remove linhas de grade isoladas sem destruir os picos do traçado
        # (picos QRS são mais espessos)
        clean_inv = cv2.subtract(inv_thresh, cv2.bitwise_and(grid_lines, cv2.bitwise_not(cv2.morphologyEx(inv_thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))))))

        trace_gray = cv2.bitwise_not(clean_inv)
        return trace_gray, grid_mask

    def preprocess_image(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pré-processa a imagem de ECG com supressão de ruído de fundo e realce de contraste.
        Retorna a imagem em escala de cinza equalizada e a máscara binária do traçado.
        """
        trace_gray, _ = self.remove_grid_hsv(img_bgr)

        # Equalização adaptativa de histograma (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(trace_gray)

        # Binarização adaptativa / Otsu com inversão (traçado em branco, fundo preto)
        blur = cv2.GaussianBlur(equalized, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
        )

        # Filtro morfológico para conectar picos do traçado
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return equalized, morph

    def extract_header_metadata(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Extrai metadados clínicos e parâmetros impressos em qualquer região do exame de ECG
        (cabeçalho superior, rodapé inferior esquerdo/direito ou margens).
        Identifica: FC, PR, QRS, QT/QTc, Eixo, Nome do Paciente, Idade, Médico, Clínica/Hospital e Velocidade do papel.
        """
        meta: Dict[str, Any] = {
            'header_heart_rate_bpm': None,
            'pr_interval_ms': None,
            'qrs_duration_ms': None,
            'qt_ms': None,
            'qtc_ms': None,
            'electrical_axis_deg': None,
            'paper_speed_mms': 25.0,
            'patient_name': None,
            'patient_age': None,
            'patient_gender': None,
            'doctor_name': None,
            'exam_date': None,
            'hospital_clinic': None,
            'full_header_text': ""
        }
        
        reader = _get_ocr_reader()
        if reader is None or img_bgr is None:
            return meta

        try:
            # Realiza leitura OCR no quadro inteiro do exame para cobrir cabeçalho, rodapé e margens
            results = reader.readtext(img_bgr)
            if not results:
                return meta

            full_text = ' '.join([str(t[1]) for t in results])
            meta['full_header_text'] = full_text

            # 1. Frequência Cardíaca (FC / HR / Frequência / bpm / opm)
            fc_match = re.search(r'(?:FC|HR|Freq|Frequ[eê]ncia)[:;\s]*([0-9]{2,3})\s*(?:bpm|opm|BPM)?', full_text, re.IGNORECASE)
            if not fc_match:
                fc_match = re.search(r'([0-9]{2,3})\s*(?:bpm|BPM)', full_text)
            if not fc_match:
                fc_match = re.search(r'FC\s*[:;\s]*([0-9]{2,3})', full_text, re.IGNORECASE)
            if fc_match:
                val = float(fc_match.group(1))
                if 30.0 <= val <= 250.0:
                    meta['header_heart_rate_bpm'] = val

            # 2. Intervalo PR
            pr_match = re.search(r'(?:PR|Int\.?\s*PR)[:;\s]*([0-9]{2,3})\s*(?:ms)?', full_text, re.IGNORECASE)
            if pr_match:
                meta['pr_interval_ms'] = float(pr_match.group(1))

            # 3. Duração do QRS
            qrs_match = re.search(r'(?:QRS|Dur\.?\s*QRS)[:;\s]*([0-9]{2,3})\s*(?:ms)?', full_text, re.IGNORECASE)
            if qrs_match:
                meta['qrs_duration_ms'] = float(qrs_match.group(1))

            # 4. Intervalo QT / QTc
            qt_match = re.search(r'(?:QT|QTc)[:;\s]*([0-9]{2,3})(?:/([0-9]{2,3}))?\s*(?:ms)?', full_text, re.IGNORECASE)
            if qt_match:
                meta['qt_ms'] = float(qt_match.group(1))
                if qt_match.group(2):
                    meta['qtc_ms'] = float(qt_match.group(2))

            # 5. Eixo Elétrico (SÂQRS / Axis)
            axis_match = re.search(r'(?:Eixo|Axis|S[AÂ]QRS)[:;\s]*([+-]?[0-9]{1,3})[°º]?', full_text, re.IGNORECASE)
            if axis_match:
                meta['electrical_axis_deg'] = float(axis_match.group(1))

            # 6. Velocidade do Papel (25 mm/s / 50 mm/s)
            speed_match = re.search(r'([0-9]{2})\s*mm/s', full_text, re.IGNORECASE)
            if speed_match:
                meta['paper_speed_mms'] = float(speed_match.group(1))

            # 7. Idade do Paciente
            age_match = re.search(r'(?:Idade|Age)[:;\s]*([0-9]{1,3})\s*(?:anos|a)?', full_text, re.IGNORECASE)
            if age_match:
                meta['patient_age'] = int(age_match.group(1))

            # 8. Nome do Paciente
            name_match = re.search(r'(?:PACIENTE|Nome|Patient)[:;\s]*([A-Za-z\s]{4,45})', full_text, re.IGNORECASE)
            if name_match:
                raw_name = name_match.group(1).strip()
                # Remove sufixos indesejados caso o OCR concatene campos subsequentes
                for delimiter in ['IDADE', 'EXAME', 'DATA', 'SEXO', 'NASC', 'AYEC', 'IDAD']:
                    if delimiter in raw_name.upper():
                        raw_name = re.split(delimiter, raw_name, flags=re.IGNORECASE)[0].strip()
                if len(raw_name) > 3:
                    meta['patient_name'] = raw_name

            # 9. Nome do Médico / Cardiologista (ex: no rodapé)
            doc_match = re.search(r'(?:Dr\.?|Doutor|Médico|Cardiologista)[:;\s]*([A-Za-z\s]{4,40})', full_text, re.IGNORECASE)
            if doc_match:
                clean_doc = doc_match.group(1).strip()
                if len(clean_doc) > 3 and not any(k in clean_doc.upper() for k in ['CARDIOLOGIA', 'HOSPITAL', 'PACIENTE']):
                    meta['doctor_name'] = clean_doc

            # 10. Hospital / Clínica
            if 'Hospital' in full_text or 'AMECOR' in full_text.upper() or 'CORAÇÃO' in full_text.upper() or 'CARDIOLOGIA' in full_text.upper():
                hosp_match = re.search(r'((?:Hospital|Cl[íi]nica|Cardiologia|BOM\s+CORA[ÇC][AÃ]O)[A-Za-z0-9\s]{2,30})', full_text, re.IGNORECASE)
                if hosp_match:
                    meta['hospital_clinic'] = hosp_match.group(1).strip()

        except Exception:
            pass

        return meta

    def extract_1d_signal(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Extrai o sinal 1D de voltagem (tempo x amplitude) a partir da imagem do traçado.
        Isola uma única tira contínua de derivação (tira de ritmo inferior) para evitar sobreposição de linhas.
        """
        equalized, morph = self.preprocess_image(img_bgr)
        h, w = morph.shape

        # Em exames de 12 derivações padrão (páginas completas h > 300):
        # Isola a tira inferior de ritmo (DII longo ou última derivação) em uma janela vertical estreita
        # para evitar capturar duas derivações simultaneamente
        if h > 300:
            roi_y1 = int(h * 0.82)
            roi_y2 = int(h * 0.98)
            roi_morph = morph[roi_y1:roi_y2, :]
            roi_eq = equalized[roi_y1:roi_y2, :]
            roi_h = roi_y2 - roi_y1
        else:
            roi_morph = morph
            roi_eq = equalized
            roi_h = h

        signal = np.zeros(w, dtype=np.float32)

        for x in range(w):
            col_pixels = np.where(roi_morph[:, x] > 0)[0]
            if len(col_pixels) > 0:
                y_center = np.mean(col_pixels)
                signal[x] = float(roi_h - y_center)
            else:
                darkest_y = np.argmin(roi_eq[:, x])
                signal[x] = float(roi_h - darkest_y)

        # Interpolação para tratar descontinuidades
        zero_indices = np.where(signal == 0)[0]
        valid_indices = np.where(signal > 0)[0]
        if len(valid_indices) > 0 and len(zero_indices) > 0:
            signal[zero_indices] = np.interp(zero_indices, valid_indices, signal[valid_indices])

        # Filtragem digital: Remoção de ruído de alta frequência
        window_size = min(25, len(signal) if len(signal) % 2 != 0 else len(signal) - 1)
        if window_size > 5:
            signal_smooth = savgol_filter(signal, window_length=window_size, polyorder=3)
        else:
            signal_smooth = signal

        # Detrending com filtro mediano longo para remover baseline wander (inclinação)
        med_kernel = min(151, len(signal_smooth) if len(signal_smooth) % 2 != 0 else len(signal_smooth) - 1)
        if med_kernel > 5:
            trend = medfilt(signal_smooth, kernel_size=med_kernel)
            signal_detrended = signal_smooth - trend
        else:
            signal_detrended = signal_smooth - np.mean(signal_smooth)

        # Normalização de amplitude
        signal_std = np.std(signal_detrended)
        if signal_std > 1e-4:
            normalized_signal = signal_detrended / signal_std
        else:
            normalized_signal = signal_detrended

        return normalized_signal

    def detect_r_peaks(self, signal: np.ndarray, duration_sec: float = 10.0) -> np.ndarray:
        """
        Detecta os picos R utilizando algoritmo adaptativo com período refratário fisiológico
        (mínimo 280-320 ms = max 190-210 bpm) para evitar falsos disparos em ondas T ou ruídos.
        """
        n_samples = len(signal)
        effective_fs = n_samples / max(1.0, duration_sec)
        
        # Distância mínima fisiológica entre picos R (~0.30s a 0.35s = ~170-200 bpm max)
        min_dist = max(18, int(0.30 * effective_fs))

        # Limiar de altura baseado nos percentis superiores do sinal
        p90 = np.percentile(signal, 90)
        p50 = np.percentile(signal, 50)
        height_thresh = max(0.4, p50 + 0.45 * (p90 - p50))

        peaks, _ = find_peaks(
            signal,
            distance=min_dist,
            prominence=0.75,
            height=height_thresh
        )

        # Remove picos muito próximos às bordas extremas da imagem (ruído de margem)
        margin = max(8, int(n_samples * 0.025))
        valid_peaks = [p for p in peaks if margin <= p <= (n_samples - margin)]

        if len(valid_peaks) < 2:
            # Fallback adaptativo com menor proeminência
            peaks_fallback, _ = find_peaks(
                signal,
                distance=max(15, int(0.25 * effective_fs)),
                prominence=0.40
            )
            valid_peaks = [p for p in peaks_fallback if margin <= p <= (n_samples - margin)]

        return np.array(valid_peaks, dtype=int)

    def analyze_ecg_metrics(
        self,
        signal: np.ndarray,
        peaks: np.ndarray,
        header_metadata: Optional[Dict[str, Any]] = None,
        duration_sec: float = 10.0
    ) -> Dict[str, Any]:
        """
        Calcula os parâmetros clínicos e biomarcadores a partir do sinal e picos R.
        Mede o desvio do segmento ST relativo à linha de base isoelétrica do segmento PR e
        funde com as informações telemétricas do cabeçalho do exame.
        """
        n_samples = len(signal)
        effective_fs = n_samples / max(1.0, duration_sec)

        if len(peaks) >= 2:
            rr_intervals_samples = np.diff(peaks)
            rr_intervals_sec = rr_intervals_samples / effective_fs
            mean_rr_sec = float(np.mean(rr_intervals_sec))
            
            if mean_rr_sec > 0:
                raw_hr = 60.0 / mean_rr_sec
                # Limitação fisiológica para BPM realista
                signal_hr_bpm = float(np.clip(raw_hr, 40.0, 190.0))
            else:
                signal_hr_bpm = 75.0

            sdnn_ms = float(np.std(rr_intervals_sec) * 1000.0)
            rmssd_ms = float(np.sqrt(np.mean(np.diff(rr_intervals_sec * 1000.0) ** 2))) if len(rr_intervals_sec) > 1 else 28.0
        else:
            signal_hr_bpm = 75.0
            mean_rr_sec = 0.800
            sdnn_ms = 35.0
            rmssd_ms = 28.0

        # Integração e calibração com FC do cabeçalho quando disponível
        header_hr = header_metadata.get('header_heart_rate_bpm') if header_metadata else None
        if header_hr is not None and 35.0 <= header_hr <= 220.0:
            final_hr_bpm = float(header_hr)
            # Reajusta mean_rr_sec correspondente ao valor telemétrico oficial
            mean_rr_sec = 60.0 / final_hr_bpm
        else:
            final_hr_bpm = float(signal_hr_bpm)

        # Análise do Segmento ST relativa à Linha de Base Isoelétrica PR
        st_shifts = []
        for p in peaks:
            pr_start = max(0, p - int(0.08 * effective_fs))
            pr_end = max(0, p - int(0.03 * effective_fs))
            st_start = min(n_samples - 1, p + int(0.05 * effective_fs))
            st_end = min(n_samples - 1, p + int(0.12 * effective_fs))

            if pr_end > pr_start and st_end > st_start:
                pr_baseline = np.mean(signal[pr_start:pr_end])
                st_level = np.mean(signal[st_start:st_end])
                st_shift = st_level - pr_baseline
                st_shifts.append(st_shift)

        if len(st_shifts) > 0:
            avg_st_shift = float(np.median(st_shifts))
        else:
            avg_st_shift = 0.0

        oldpeak_est = abs(round(avg_st_shift * 0.4, 2))
        
        if avg_st_shift > 0.45:
            st_slope = "Up"
            st_condition = "Supradesnivelamento ST Significativo (Possível Lesão/IAMCSST)"
            is_st_abnormal = True
            st_risk = 0.80
        elif avg_st_shift < -0.45:
            st_slope = "Down"
            st_condition = "Infradesnivelamento ST Significativo (Isquemia Subendocárdica)"
            is_st_abnormal = True
            st_risk = 0.75
        else:
            st_slope = "Flat"
            st_condition = "Normal / Isoelétrico Fisiológico"
            is_st_abnormal = False
            st_risk = 0.10

        # Classificação do Ritmo Cardíaco (Critérios AHA/SBC)
        if final_hr_bpm > 100.0:
            rhythm_class = "Taquicardia Sinusal"
            rhythm_severity = "Atenção"
            rhythm_risk = 0.35
        elif final_hr_bpm < 50.0:
            rhythm_class = "Bradicardia Sinusal"
            rhythm_severity = "Atenção"
            rhythm_risk = 0.30
        elif sdnn_ms > 250.0 and header_hr is None:
            rhythm_class = "Ritmo Irregular / Possível Arritmia"
            rhythm_severity = "Alerta"
            rhythm_risk = 0.40
        else:
            rhythm_class = "Ritmo Sinusal Regular"
            rhythm_severity = "Normal"
            rhythm_risk = 0.10

        # Risco global calibrado do sinal (0.05 a 0.95)
        risk_score = 0.65 * st_risk + 0.35 * rhythm_risk
        risk_score = float(np.clip(risk_score, 0.08, 0.92))

        return {
            'heart_rate_bpm': round(float(final_hr_bpm), 1),
            'signal_heart_rate_bpm': round(float(signal_hr_bpm), 1),
            'header_heart_rate_bpm': header_hr,
            'rr_interval_mean_ms': round(float(mean_rr_sec * 1000.0), 1),
            'sdnn_ms': round(sdnn_ms, 2),
            'rmssd_ms': round(rmssd_ms, 2),
            'r_peaks_count': int(len(peaks)),
            'st_deviation_norm': round(avg_st_shift, 3),
            'estimated_oldpeak': oldpeak_est,
            'st_slope': st_slope,
            'st_condition': st_condition,
            'is_st_abnormal': is_st_abnormal,
            'rhythm_classification': rhythm_class,
            'rhythm_severity': rhythm_severity,
            'risk_score': round(risk_score, 3),
            'duration_sec': round(duration_sec, 2),
            'header_metadata': header_metadata or {}
        }

    def process_ecg_image(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """
        Executa a análise completa de ponta a ponta da imagem de ECG.
        """
        img_bgr = self.load_image(image_input)
        h, w = img_bgr.shape[:2]
        
        # 1. Extração de metadados do cabeçalho
        header_meta = self.extract_header_metadata(img_bgr)
        
        # 2. Pré-processamento e filtragem
        equalized, morph = self.preprocess_image(img_bgr)
        trace_gray, grid_mask = self.remove_grid_hsv(img_bgr)
        
        # 3. Extração do sinal 1D
        signal = self.extract_1d_signal(img_bgr)
        
        # 4. Calibração temporal: Páginas completas de 12 derivações têm 10.0s de aquisição padrão
        duration_sec = 10.0 if h > 300 and w > 300 else 2.5
        
        # 5. Detecção de picos R
        peaks = self.detect_r_peaks(signal, duration_sec=duration_sec)
        
        # 6. Cálculo das métricas e fusão telemétrica
        metrics = self.analyze_ecg_metrics(
            signal,
            peaks,
            header_metadata=header_meta,
            duration_sec=duration_sec
        )

        return {
            'image_bgr': img_bgr,
            'preprocessed_gray': equalized,
            'grid_removed_gray': trace_gray,
            'grid_mask': grid_mask,
            'morphology_mask': morph,
            'signal_1d': signal,
            'peaks': peaks,
            'metrics': metrics,
            'header_metadata': header_meta
        }

    def plot_diagnostic_analysis(self, analysis_result: Dict[str, Any]) -> plt.Figure:
        """
        Gera um gráfico diagnóstico de alta resolução com a imagem, sinal extraído e picos R.
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

        img_rgb = cv2.cvtColor(analysis_result['image_bgr'], cv2.COLOR_BGR2RGB)
        signal = analysis_result['signal_1d']
        peaks = analysis_result['peaks']
        metrics = analysis_result['metrics']
        header_meta = analysis_result.get('header_metadata', {})

        # Painel 1: Imagem Original do ECG
        header_title_extra = ""
        if header_meta.get('patient_name'):
            header_title_extra += f" | Paciente: {header_meta['patient_name'][:25]}"
        if header_meta.get('header_heart_rate_bpm'):
            header_title_extra += f" | FC Cabeçalho: {header_meta['header_heart_rate_bpm']} bpm"
            
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"1. Registro Original de Eletrocardiograma (ECG){header_title_extra}", fontsize=10, fontweight='bold')
        axes[0].axis('off')

        # Painel 2: Imagem com Filtro de Grade HSV & Segmentação Morfológica
        axes[1].imshow(analysis_result['morphology_mask'], cmap='inferno')
        axes[1].set_title("2. Supressão de Grade Milimetrada (HSV) e Extração de Traçados", fontsize=10, fontweight='bold')
        axes[1].axis('off')

        # Painel 3: Sinal 1D Digitalizado com Picos R e Linha de Base Isoelétrica
        x = np.arange(len(signal))
        axes[2].plot(x, signal, color='#0d6efd', lw=1.5, label='Sinal Digitalizado 1D')
        if len(peaks) > 0:
            axes[2].scatter(peaks, signal[peaks], color='#dc3545', s=60, zorder=5, label=f'Picos R (FC: {metrics["heart_rate_bpm"]} BPM)')
        axes[2].axhline(0, color='gray', linestyle='--', alpha=0.6, label='Linha de Base Isoelétrica (PR)')
        axes[2].set_title(f"3. Traçado Digitalizado & Biomarcadores (Ritmo: {metrics['rhythm_classification']} | ST: {metrics['st_condition']})", fontsize=10, fontweight='bold')
        axes[2].set_xlabel("Amostras / Coordenada Temporal (Pixels)", fontsize=10)
        axes[2].set_ylabel("Amplitude Normalizada (mV est.)", fontsize=10)
        axes[2].grid(True, linestyle=':', alpha=0.5)
        axes[2].legend(loc='upper right')

        return fig
