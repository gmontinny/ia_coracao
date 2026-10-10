"""
Gerador de Laudos Clínicos e Sumários Acadêmicos com Inteligência Artificial e LLM.
Estruturado sob princípios de utilidade clínica, estratificação de risco e rigor científico.
"""

import os
from typing import Dict, Any, List, Optional, Union
import json
import datetime

from src.data.ecg_knowledge_rag import ECGKnowledgeVectorDB


class ECGClinicalReportGenerator:
    """
    Gerador de Laudos Clínicos e Seções Acadêmicas para Artigos Científicos.
    Sintetiza sinais de ECG, visão computacional e modelos preditivos com recuperação
    aumentada por inteligência (RAG) em diretrizes internacionais (AHA/ACC/ESC/SBC).
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.rag_engine = ECGKnowledgeVectorDB()

    def generate_clinical_report(
        self,
        patient_data: Optional[Dict[str, Any]] = None,
        ecg_metrics: Optional[Dict[str, Any]] = None,
        vision_metrics: Optional[Dict[str, Any]] = None,
        ml_prediction: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Gera um laudo médico estruturado com estratificação de risco utilitária.
        """
        timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        # Consolidação de variáveis de risco
        risk_score = 0.0
        risk_components = []

        if ml_prediction:
            risk_score = max(risk_score, ml_prediction.get('risk_score', 0.0))
            prob_pct = f"{ml_prediction.get('risk_score', 0.0) * 100:.1f}%"
            risk_components.append(f"Probabilidade pelo Modelo Ensemble: {prob_pct}")

        if ecg_metrics:
            ecg_risk = ecg_metrics.get('risk_score', 0.0)
            risk_score = max(risk_score, ecg_risk)
            if ecg_metrics.get('is_st_abnormal'):
                risk_components.append(f"Anormalidade de Segmento ST: {ecg_metrics.get('st_condition')} (Oldpeak est.: {ecg_metrics.get('estimated_oldpeak')} mV)")
            if ecg_metrics.get('rhythm_severity') != "Normal":
                risk_components.append(f"Alteração de Ritmo: {ecg_metrics.get('rhythm_classification')} ({ecg_metrics.get('heart_rate_bpm')} BPM)")

        if vision_metrics:
            v_risk = vision_metrics.get('prob_disease', 0.0)
            if v_risk >= 0.5:
                risk_components.append(f"Detecção por Rede Neural Convolucional (Grad-CAM): Padrão Morfológico Isquêmico ({v_risk * 100:.1f}%)")

        # Estratificação utilitária de conduta clínica
        if risk_score >= 0.65:
            urgency_level = "ALTA PRIORIDADE / URGÊNCIA CARDIOLÓGICA"
            triage_action = "Encaminhamento imediato para avaliação cardiológica especializada, monitorização contínua e estratificação invasiva/não invasiva."
            recommended_tests = [
                "Cineangiocoronariografia (Cateterismo Cardíaco) se quadro instável",
                "Ecocardiograma Transtorácico com Doppler para avaliação de fração de ejeção e motilidade segmentar",
                "Dosagem seriada de Troponina Ultrassensível e CK-MB",
                "Eletrocardiograma de 12 derivações seriado"
            ]
            prognosis = "Elevado risco de eventos cardiovasculares adversos maiores (MACE). A intervenção precoce maximiza anos de vida ajustados por qualidade (QALYs) e previne morbimortalidade."
        elif risk_score >= 0.35:
            urgency_level = "PRIORIDADE MODERADA / ACOMPANHAMENTO AMBULATORIAL PRIORITÁRIO"
            triage_action = "Agendamento de consulta cardiológica em curto prazo para complementação diagnóstica funcional."
            recommended_tests = [
                "Teste Ergométrico Computadorizado ou Ecocardiograma de Estresse",
                "Holter 24 horas para pesquisa de arritmias paroxísticas",
                "Angiotomografia de Artérias Coronárias (Escore de Cálcio)",
                "Perfil Lipídico completo e Glicemia de Jejum / HbA1c"
            ]
            prognosis = "Risco intermediário. O controle rigoroso de fatores de risco modificáveis (pressão, lipídios, estilo de vida) previne progressão aterosclerótica."
        else:
            urgency_level = "BAIXO RISCO / MANUTENÇÃO PREVENTIVA"
            triage_action = "Manutenção de hábitos saudáveis, reavaliação de rotina e controle preventivo dos fatores de risco."
            recommended_tests = [
                "Check-up cardiológico anual de rotina",
                "Acompanhamento da pressão arterial e exames laboratoriais periódicos"
            ]
            prognosis = "Excelente prognóstico cardiovascular global a curto e médio prazo sob manutenção de hábitos cardioprotetores."

        # Recuperação Semântica de Diretrizes (RAG)
        retrieved_guidelines = self.rag_engine.retrieve_clinical_context(
            ecg_metrics=ecg_metrics,
            ml_prediction=ml_prediction,
            patient_data=patient_data
        )
        rag_citations_text = self.rag_engine.format_rag_citations(retrieved_guidelines)

        # Metadados do cabeçalho
        header_meta = ecg_metrics.get('header_metadata', {}) if ecg_metrics else {}
        header_hr_str = f" (Cabeçalho: {header_meta.get('header_heart_rate_bpm')} BPM)" if header_meta.get('header_heart_rate_bpm') else ""
        patient_str = f"- Identificação / Paciente: {header_meta.get('patient_name')}\n" if header_meta.get('patient_name') else ""
        if patient_data and patient_data.get('Age'):
            patient_str += f"- Idade do Paciente: {patient_data.get('Age')} anos\n"
        elif header_meta.get('patient_age'):
            patient_str += f"- Idade do Paciente: {header_meta.get('patient_age')} anos\n"

        intervals_str = ""
        if header_meta.get('pr_interval_ms'):
            intervals_str += f"- Intervalo PR: {header_meta.get('pr_interval_ms')} ms\n"
        if header_meta.get('qrs_duration_ms'):
            intervals_str += f"- Duração QRS: {header_meta.get('qrs_duration_ms')} ms\n"
        if header_meta.get('qt_ms'):
            intervals_str += f"- Intervalo QT/QTc: {header_meta.get('qt_ms')}{' / ' + str(header_meta.get('qtc_ms')) if header_meta.get('qtc_ms') else ''} ms\n"

        # Montagem do relatório formatado
        report_text = f"""
================================================================================
                    RELATÓRIO CLÍNICO DE ANÁLISE CARDIOLÓGICA
                     SISTEMA DE INTELIGÊNCIA ARTIFICIAL ECG-AI
================================================================================
Data/Hora da Emissão: {timestamp}
Nível de Triagem / Urgência: {urgency_level}
Índice Global de Risco Cardiológico: {risk_score * 100:.1f}%

1. ACHADOS MULTIMODAIS E BIOMARCADORES DETECTADOS:
--------------------------------------------------------------------------------
{patient_str}{chr(10).join(['- ' + c for c in risk_components]) if risk_components else '- Nenhum biomarcador de alto risco detectado nos exames analisados.'}

2. DADOS ELETROCARDIOGRÁFICOS COMPUTADOS E TELEMETRIA:
--------------------------------------------------------------------------------
- Frequência Cardíaca Calibrada: {ecg_metrics.get('heart_rate_bpm', 'N/A') if ecg_metrics else 'N/A'} BPM{header_hr_str}
- Ritmo Cardíaco: {ecg_metrics.get('rhythm_classification', 'N/A') if ecg_metrics else 'N/A'}
- Intervalo RR Médio: {ecg_metrics.get('rr_interval_mean_ms', 'N/A') if ecg_metrics else 'N/A'} ms
- Variabilidade da FC (SDNN): {ecg_metrics.get('sdnn_ms', 'N/A') if ecg_metrics else 'N/A'} ms
{intervals_str}- Morfologia do Segmento ST: {ecg_metrics.get('st_condition', 'N/A') if ecg_metrics else 'N/A'}
- Depressão/Elevação ST (Oldpeak est.): {ecg_metrics.get('estimated_oldpeak', 'N/A') if ecg_metrics else 'N/A'} mV

3. CONDUTA CLÍNICA BASEADA EM UTILIDADE E MAXIMIZAÇÃO DE BENEFÍCIO:
--------------------------------------------------------------------------------
{triage_action}

Exames Complementares de Maior Rendimento Diagnóstico:
{chr(10).join(['  • ' + t for t in recommended_tests])}

4. DIRETRIZES CLÍNICAS E EVIDÊNCIAS RECUPERADAS (RAG / BASE VETORIAL):
--------------------------------------------------------------------------------
{rag_citations_text if rag_citations_text else 'Diretrizes gerais de prevenção cardiovascular aplicadas.'}

5. PERSPECTIVA PROGNÓSTICA E IMPACTO EM SAÚDE:
--------------------------------------------------------------------------------
{prognosis}
================================================================================
* Nota de Responsabilidade: Este laudo foi gerado por sistema de suporte à decisão clínica por IA. 
Deve ser interpretado por médico devidamente habilitado conjuntamente com o quadro clínico do paciente.
"""
        return {
            'timestamp': timestamp,
            'urgency_level': urgency_level,
            'risk_score': risk_score,
            'risk_components': risk_components,
            'triage_action': triage_action,
            'recommended_tests': recommended_tests,
            'prognosis': prognosis,
            'retrieved_guidelines': retrieved_guidelines,
            'full_report_text': report_text.strip()
        }

    def generate_academic_article_section(
        self,
        benchmark_results: Optional[List[Dict[str, Any]]] = None,
        dataset_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Gera uma seção metodológica e de resultados pronta para submissão em artigos científicos.
        """
        section = """
### 3. Methodology & Experimental Framework (Para Publicação Científica)

#### 3.1 Dataset Characterization and Preprocessing
The study evaluates the multi-modal integration of clinical tabular records and digitalized electrocardiographic (ECG) waveforms. The preprocessing pipeline applies robust standard scaling for numerical features and one-hot encoding for categorical descriptors, with imputation strategies executed within cross-validation folds to prevent data leakage.

#### 3.2 Machine Learning and Deep Learning Architectures
The analytical framework incorporates an ensemble of supervised learning models:
1. **Gradient Boosting Models (XGBoost & LightGBM)**: Configured with shallow tree depth ($max\\_depth=3$) and learning rates $\\eta=0.05$ with $L_2$ regularization to minimize overfitting.
2. **Deep Convolutional Neural Network (CNN)**: Multi-layer residual convolutional architecture extracting spatial waveform anomalies, coupled with Gradient-weighted Class Activation Mapping (Grad-CAM) to provide transparent feature attribution for clinicians.
3. **Digital Signal Processing Pipeline**: Converts raw ECG images to 1D voltage traces via adaptive CLAHE filtering, morphological opening, centroid trace reconstruction, and Savitzky-Golay noise filtering.

#### 3.3 Clinical Utility and Evaluation Metrics
Models were validated using 5-fold Stratified Cross-Validation. Given the asymmetric cost of diagnostic errors in cardiology—where false negatives lead to critical patient mortality—the evaluation framework emphasizes **Sensitivity (Recall)**, **Area Under the ROC Curve (ROC-AUC)**, **Precision-Recall AUC (PR-AUC)**, and **Brier Score Calibration**.
"""
        return section.strip()
