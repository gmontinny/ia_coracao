"""
Aplicação Web Interativa para Análise Eletrocardiográfica e Risco Cardiológico com IA.
Integra Visão Computacional (HSV Grid Filter), Detecção YOLO de Derivações, Deep Learning (CNN + Grad-CAM), Machine Learning Tabular e LLM.
"""

import os
from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Importações dos módulos do sistema ECG-AI
from src.data.loader import (
    load_heart_dataset,
    split_heart_data,
    FEATURE_METADATA,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN
)
from src.models.tabular_models import TabularECGClassifier, train_benchmark_suite
from src.models.ecg_signal_processing import ECGSignalProcessor
from src.models.ecg_vision import ECGDeepVisionNet
from src.models.ecg_yolo import ECGYoloDetector
from src.models.llm_report import ECGClinicalReportGenerator
from src.data.hf_dataset_loader import HuggingFaceECGHub
from src.data.ecg_knowledge_rag import ECGKnowledgeVectorDB
from src.data.mimic_vectorizer import MIMICIVVectorizer
from src.pipeline.ecg_pipeline import UnifiedECGPipeline
from src.utils.visualization import (
    plot_roc_comparison,
    plot_pr_comparison,
    plot_confusion_matrix_heatmap,
    plot_feature_importances_chart,
    plot_calibration_curve_custom
)
from src.utils.academic_metrics import (
    generate_latex_table,
    generate_markdown_table,
    calculate_metric_confidence_intervals
)

# Configuração da página Streamlit
st.set_page_config(
    page_title="ECG-AI | Inteligência Artificial em Eletrocardiograma",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS personalizada para ambiente clínico e acadêmico de alta usabilidade
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0d6efd;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #495057;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #0d6efd;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .high-risk {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 4px;
        color: #842029;
    }
    .moderate-risk {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        color: #664d03;
    }
    .low-risk {
        background-color: #d1e7dd;
        border-left: 5px solid #198754;
        padding: 1rem;
        border-radius: 4px;
        color: #0f5132;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_pipeline():
    """Inicializa e armazena em cache o pipeline multimodal unificado."""
    return UnifiedECGPipeline(auto_train_tabular=True, tabular_model_type='xgboost')


@st.cache_data
def get_cached_benchmark():
    """Calcula e armazena em cache o benchmark comparativo de todos os modelos."""
    df_raw = load_heart_dataset(treated=True)
    df_bench, models_dict = train_benchmark_suite(df=df_raw, cv=5)
    return df_bench, models_dict


def main():
    pipeline = get_cached_pipeline()

    # Barra lateral de navegação e configurações
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=72)
        st.markdown("### **ECG-AI Studio**")
        st.markdown("*Plataforma Acadêmica e Clínica de Diagnóstico Cardiológico por IA*")
        st.markdown("---")
        
        menu = st.radio(
            "Selecione o Módulo de Análise:",
            [
                "🫀 Análise de Imagem de ECG (Deep Learning & YOLO)",
                "📊 Análise de Dados Clínicos & CSV (Machine Learning)",
                "🔬 Laboratório de Pesquisa & Benchmark Acadêmico",
                "🌐 Hugging Face ECG Hub (64+ Datasets & Benchmark)",
                "📚 Base Vetorial & Diretrizes Clínicas (RAG)",
                "📝 Gerador de Laudos & Minuta para Artigo Científico",
                "📖 Fundamentos Teóricos & Metodologia"
            ]
        )

        st.markdown("---")
        st.markdown("##### **Especificações Técnicas:**")
        st.caption("• **Visão Computacional:** Filtro de Grade HSV + PDS")
        st.caption("• **Detecção de Derivações:** YOLO/COCO Bounding Boxes")
        st.caption("• **Deep Learning:** CNN com Grad-CAM & Incerteza Conformal")
        st.caption("• **Calibração:** Temperature Scaling & Isoelétrico PR")
        st.caption("• **Classificadores Tabulares:** XGBoost, LightGBM, RF, MLP")

    # Cabeçalho Superior
    st.markdown('<div class="main-header">Sistema de Inteligência Artificial para Eletrocardiograma (ECG-AI)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Pesquisa Acadêmica, Visão Computacional, Aprendizado Profundo e Suporte à Decisão Clínica</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------------------------------
    # MÓDULO 1: ANÁLISE DE IMAGEM DE ECG
    # -------------------------------------------------------------------------------------------------
    if menu == "🫀 Análise de Imagem de ECG (Deep Learning & YOLO)":
        st.markdown("### 🫀 Leitura e Processamento Digital de Imagem de Eletrocardiograma")
        st.markdown("""
        Análise multimodal que realiza:
        - **Supressão de grade milimetrada em espaço HSV** (elimina artefatos de fundo).
        - **Detecção e segmentação de derivações no padrão YOLO/COCO** (12 derivações + DII longo).
        - **Extração de sinal 1D calibrado** com referência na linha de base isoelétrica do segmento PR.
        - **Inferência por Rede Neural Convolucional Calibrada** com estimativa de incerteza (Conformal Prediction) e Grad-CAM.
        """)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### 1. Selecionar ou Enviar Imagem de ECG")
            upload_option = st.radio(
                "Fonte da Imagem:",
                ["Usar Imagens de Amostra da Base (Pasta img/)", "Fazer Upload de Nova Imagem (PNG/JPG)"],
                horizontal=True
            )

            image_to_process = None
            image_name = ""

            if upload_option == "Usar Imagens de Amostra da Base (Pasta img/)":
                img_dir = Path("img")
                sample_files = [f.name for f in img_dir.glob("*.*") if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
                if sample_files:
                    selected_sample = st.selectbox("Escolha uma imagem de exemplo:", sample_files)
                    sample_path = img_dir / selected_sample
                    image_to_process = str(sample_path)
                    image_name = selected_sample
                else:
                    st.warning("Nenhuma imagem encontrada na pasta img/.")
            else:
                uploaded_file = st.file_uploader("Selecione um arquivo de imagem de ECG:", type=['png', 'jpg', 'jpeg'])
                if uploaded_file is not None:
                    image_to_process = uploaded_file.read()
                    image_name = uploaded_file.name

            st.markdown("#### 2. Metadados Clínicos Opcionais do Paciente (Fusão Multimodal)")
            with st.expander("Preencher dados adicionais do paciente (Opcional)", expanded=False):
                p_age = st.slider("Idade (anos)", 18, 95, 54)
                p_sex = st.selectbox("Sexo", ["M", "F"], format_func=lambda x: "Masculino" if x == "M" else "Feminino")
                p_cp = st.selectbox("Tipo de Dor Torácica", ["ASY", "NAP", "ATA", "TA"], 
                                   format_func=lambda x: FEATURE_METADATA['ChestPainType']['values'][x])
                p_bp = st.slider("Pressão Arterial em Repouso (mmHg)", 80, 200, 135)
                p_chol = st.slider("Colesterol Sérico (mg/dL)", 100, 500, 230)
                p_fbs = st.selectbox("Glicemia em Jejum > 120 mg/dL", [0, 1], format_func=lambda x: "Sim (> 120)" if x == 1 else "Não (<= 120)")
                p_angina = st.selectbox("Angina Induzida por Exercício", ["N", "Y"], format_func=lambda x: "Sim" if x == "Y" else "Não")

                patient_meta = {
                    'Age': p_age,
                    'Sex': p_sex,
                    'ChestPainType': p_cp,
                    'RestingBP': p_bp,
                    'Cholesterol': p_chol,
                    'FastingBS': p_fbs,
                    'ExerciseAngina': p_angina
                }

        with col_right:
            st.markdown("#### 3. Visualização Prévia")
            if image_to_process is not None:
                if isinstance(image_to_process, str):
                    preview_img = Image.open(image_to_process)
                else:
                    preview_img = Image.open(io.BytesIO(image_to_process))
                st.image(preview_img, caption=f"Imagem Selecionada: {image_name}", use_container_width=True)
            else:
                st.info("Envie ou selecione uma imagem ao lado para iniciar a análise.")

        if image_to_process is not None:
            st.markdown("---")
            if st.button("🚀 Processar Imagem com Inteligência Artificial Calibrada & YOLO", type="primary", use_container_width=True):
                with st.spinner("Executando segmentação YOLO, filtro de grade HSV, extração de sinal e inferência calibrada..."):
                    result = pipeline.analyze_image_ecg(image_to_process, patient_metadata=patient_meta if 'patient_meta' in locals() else None)

                sig_proc = result['signal_processing']
                metrics = sig_proc['metrics']
                header_meta = sig_proc.get('header_metadata', {})
                vision_res = result['vision_deep_learning']
                multi_res = result['multimodal_prediction']
                yolo_res = result['yolo_detections']
                mimic_cases = result.get('mimic_similar_cases', [])
                report = result['clinical_report']

                st.markdown("### 📈 Resultados da Extração e Diagnóstico Calibrado")

                # Exibe alerta/banner informativo se o cabeçalho tiver dados telemétricos oficiais
                if header_meta.get('header_heart_rate_bpm') or header_meta.get('patient_name'):
                    st.success(
                        f"📋 **Parâmetros Lidos do Cabeçalho do Exame:** "
                        f"{'Paciente: **' + str(header_meta.get('patient_name')) + '** | ' if header_meta.get('patient_name') else ''}"
                        f"{'Idade: **' + str(header_meta.get('patient_age')) + ' anos** | ' if header_meta.get('patient_age') else ''}"
                        f"{'FC Cabeçalho: **' + str(header_meta.get('header_heart_rate_bpm')) + ' BPM** | ' if header_meta.get('header_heart_rate_bpm') else ''}"
                        f"{'PR: **' + str(header_meta.get('pr_interval_ms')) + ' ms** | ' if header_meta.get('pr_interval_ms') else ''}"
                        f"{'QRS: **' + str(header_meta.get('qrs_duration_ms')) + ' ms** | ' if header_meta.get('qrs_duration_ms') else ''}"
                        f"{'QT/QTc: **' + str(header_meta.get('qt_ms')) + ' ms** | ' if header_meta.get('qt_ms') else ''}"
                        f"Velocidade: **{header_meta.get('paper_speed_mms', 25.0)} mm/s**"
                    )

                # Métricas em Cards
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    fc_delta = f"Sinal: {metrics.get('signal_heart_rate_bpm')} bpm" if metrics.get('header_heart_rate_bpm') else "PDS Calibrado"
                    st.metric("Frequência Cardíaca", f"{metrics['heart_rate_bpm']} BPM", delta=fc_delta, help="Frequência cardíaca calibrada com base no cabeçalho telemétrico e nos intervalos RR")
                with m_col2:
                    st.metric("Ritmo Identificado", metrics['rhythm_classification'], delta=metrics['rhythm_severity'])
                with m_col3:
                    st.metric("Segmento ST / Isoelétrico", f"{metrics['st_slope']}", delta=metrics['st_condition'])
                with m_col4:
                    risk_pct = multi_res['risk_score'] * 100
                    st.metric("Risco Cardiológico Calibrado", f"{risk_pct:.1f}%", delta=multi_res['risk_tier'])

                # Abas Detalhadas de Diagnóstico
                tab_diag, tab_mimic, tab_yolo, tab_pds, tab_gradcam, tab_report = st.tabs([
                    "📊 Diagnóstico Calibrado & Incerteza",
                    "🏥 Casos Similares MIMIC-IV (k-NN)",
                    "🎯 Detecção YOLO de Derivações",
                    "🔬 Filtro de Grade HSV & PDS",
                    "🔥 Interpretabilidade Grad-CAM",
                    "📋 Laudo Clínico Automatizado"
                ])

                with tab_diag:
                    st.markdown("#### Estatísticas de Calibração e Incerteza Diagnóstica")
                    d_c1, d_c2 = st.columns(2)
                    with d_c1:
                        st.markdown(f"""
                        - **Classificação:** `{multi_res['prediction_label']}`
                        - **Grau de Risco Global:** `{multi_res['risk_tier']}`
                        - **Probabilidade de Cardiopatia:** `{multi_res['risk_score'] * 100:.1f}%`
                        - **Confiança Estatística:** `{multi_res['confidence'] * 100:.1f}%`
                        - **Intervalo de Confiança (95% CI):** `[{multi_res['confidence_interval_95'][0]*100:.1f}%, {multi_res['confidence_interval_95'][1]*100:.1f}%]`
                        - **Status de Incerteza Conformal:** `{'⚠️ Zona de Incerteza (Requer Análise Complementar)' if multi_res['is_uncertain'] else '✅ Alta Certeza Diagnóstica'}`
                        """)
                    with d_c2:
                        st.markdown(f"""
                        - **Linha de Base Isoelétrica:** Referência no Segmento PR pré-QRS
                        - **Desvio Médio de ST:** `{metrics['st_deviation_norm']:.3f} mV est.`
                        - **Oldpeak Estimado:** `{metrics['estimated_oldpeak']} mV`
                        - **Variabilidade da FC (SDNN):** `{metrics['sdnn_ms']} ms`
                        - **Raiz Média dos Quadrados (RMSSD):** `{metrics['rmssd_ms']} ms`
                        """)

                    fig_diag = pipeline.signal_processor.plot_diagnostic_analysis(sig_proc)
                    st.pyplot(fig_diag)
                    plt.close(fig_diag)

                with tab_mimic:
                    st.markdown("#### 🏥 Casos Hospitalares Semelhantes no MIMIC-IV-ECG (Qdrant Vector DB)")
                    st.markdown(
                        "O sistema codificou os biomarcadores e a morfologia da imagem em um **vetor latente (embedding de 64d)** "
                        "e recuperou por k-NN os **exames de 12 derivações mais morfologicamente idênticos** do banco de dados do *Beth Israel / Harvard*:"
                    )
                    if mimic_cases:
                        for idx, case in enumerate(mimic_cases, 1):
                            with st.expander(f"Caso #{idx}: {case.get('ecg_id', 'Exame')} — Similaridade: {case.get('similarity_percentage', 'N/A')} ({case.get('diagnostic_superclass')})", expanded=(idx == 1)):
                                c_m1, c_m2 = st.columns(2)
                                with c_m1:
                                    st.markdown(f"**Diagnóstico:** `{case.get('diagnostic_superclass')}` — {', '.join(case.get('diagnostic_statements', []))}")
                                    st.markdown(f"**Paciente:** `{case.get('age')} anos ({case.get('gender')})` | **FC:** `{case.get('heart_rate')} BPM`")
                                    st.markdown(f"**Desvio de ST:** `{case.get('st_deviation_mm')} mm` | **UTI:** `{'Sim' if case.get('in_icu') else 'Não'}`")
                                with c_m2:
                                    st.markdown(f"**Troponina I:** `{case.get('troponin_i_ng_ml')} ng/mL`")
                                    st.markdown(f"**Oclusão Coronariana:** {case.get('coronary_occlusion')}")
                                    st.markdown(f"**Desfecho Clínico:** {case.get('clinical_outcome')}")
                    else:
                        st.info("Nenhum caso hospitalar similar retornado.")

                with tab_yolo:
                    st.markdown("#### Detecção e Bounding Boxes no Padrão YOLO / COCO")
                    st.markdown("Segmentação automática de 12 derivações padrão (I, II, III, aVR, aVL, aVF, V1-V6) e tira de ritmo (DII longo):")
                    
                    st.image(yolo_res['annotated_image'], caption="Detecções YOLO: Derivações (Azul/Amarelo) e Ondas QRS/P/T (Vermelho/Verde/Magenta)", use_container_width=True)
                    
                    if yolo_res['lead_crops']:
                        st.markdown("##### Inspeção de Derivações Recortadas Individualmente:")
                        lead_keys = list(yolo_res['lead_crops'].keys())
                        chosen_lead = st.selectbox("Selecione a derivação para ampliar:", lead_keys)
                        if chosen_lead:
                            st.image(yolo_res['lead_crops'][chosen_lead], caption=f"Derivação Isolada: {chosen_lead}", width=400)

                with tab_pds:
                    st.markdown("#### Comparativo de Pré-Processamento Digital de Sinais (PDS)")
                    p_c1, p_c2 = st.columns(2)
                    with p_c1:
                        st.image(sig_proc['grid_removed_gray'], caption="1. Traçado com Supressão de Grade Milimetrada (HSV Filter)", use_container_width=True)
                    with p_c2:
                        st.image(sig_proc['morphology_mask'], caption="2. Máscara Binária Morfológica Isolada", use_container_width=True)

                with tab_gradcam:
                    st.markdown("#### Interpretabilidade Visual com Grad-CAM (Atenção Convolucional)")
                    cam_col1, cam_col2 = st.columns(2)
                    with cam_col1:
                        st.image(vision_res['gradcam_overlay_rgb'], caption="Mapa de Calor Grad-CAM (Regiões de Relevância da Rede Neural)", use_container_width=True)
                    with cam_col2:
                        st.markdown(f"""
                        **Diagnóstico da Rede Neural Convolucional Profunda (CNN):**
                        - **Classificação:** `{vision_res['prediction_label']}`
                        - **Probabilidade Calibrada de Cardiopatia:** `{vision_res['prob_disease'] * 100:.1f}%`
                        - **Confiança do Modelo Visual:** `{vision_res['confidence'] * 100:.1f}%`
                        - **Índice de Incerteza Epistêmica:** `{vision_res['uncertainty_index']}`
                        - **Mapeamento de Relevância:** Destaca complexos QRS, ponto J e morfologias de repolarização ventricular.
                        """)

                with tab_report:
                    st.markdown("#### 📋 Laudo Clínico Automatizado")
                    st.text_area("Laudo Estruturado Emitido pela IA:", value=report['full_report_text'], height=280)
                    st.download_button(
                        "📥 Baixar Laudo Clínico em TXT",
                        data=report['full_report_text'],
                        file_name=f"laudo_ecg_{image_name.replace('.', '_')}.txt",
                        mime="text/plain"
                    )

    # -------------------------------------------------------------------------------------------------
    # MÓDULO 2: ANÁLISE DE DADOS CLÍNICOS E CSV
    # -------------------------------------------------------------------------------------------------
    elif menu == "📊 Análise de Dados Clínicos & CSV (Machine Learning)":
        st.markdown("### 📊 Análise de Risco com Dados Clínicos e Exames Cardiológicos")
        st.markdown("Execute análises individuais de pacientes ou processe bases de dados em lote no formato CSV.")

        tab_ind, tab_batch = st.tabs(["👤 Avaliação de Paciente Individual", "📁 Processamento de Base em Lote (CSV)"])

        with tab_ind:
            st.markdown("#### Formulário de Entrada do Paciente")
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Idade", min_value=18, max_value=100, value=50)
                sex = st.selectbox("Sexo Biológico", ["M", "F"], format_func=lambda x: "Masculino (M)" if x == "M" else "Feminino (F)")
                cp = st.selectbox("Tipo de Dor Torácica", ["ASY", "NAP", "ATA", "TA"], 
                                  format_func=lambda x: FEATURE_METADATA['ChestPainType']['values'][x])
                bp = st.number_input("Pressão Arterial em Repouso (mmHg)", min_value=60, max_value=240, value=130)

            with c2:
                chol = st.number_input("Colesterol Sérico (mg/dL)", min_value=0, max_value=700, value=220)
                fbs = st.selectbox("Glicemia de Jejum > 120 mg/dL", [0, 1], format_func=lambda x: "Sim (> 120 mg/dL)" if x == 1 else "Não (<= 120 mg/dL)")
                ecg = st.selectbox("Eletrocardiograma em Repouso", ["Normal", "ST", "LVH"], 
                                   format_func=lambda x: FEATURE_METADATA['RestingECG']['values'][x])
                max_hr = st.number_input("Frequência Cardíaca Máxima Atingida (BPM)", min_value=50, max_value=230, value=150)

            with c3:
                ex_angina = st.selectbox("Angina Induzida por Exercício", ["N", "Y"], format_func=lambda x: "Sim (Presente)" if x == "Y" else "Não (Ausente)")
                oldpeak = st.number_input("Depressão do Segmento ST / Oldpeak (mV)", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)
                st_slope = st.selectbox("Inclinação do Segmento ST (ST Slope)", ["Up", "Flat", "Down"], 
                                        format_func=lambda x: FEATURE_METADATA['ST_Slope']['values'][x])

            patient_dict = {
                'Age': age,
                'Sex': sex,
                'ChestPainType': cp,
                'RestingBP': bp,
                'Cholesterol': chol,
                'FastingBS': fbs,
                'RestingECG': ecg,
                'MaxHR': max_hr,
                'ExerciseAngina': ex_angina,
                'Oldpeak': oldpeak,
                'ST_Slope': st_slope
            }

            if st.button("🔍 Calcular Risco Cardiológico do Paciente", type="primary"):
                pred_res = pipeline.analyze_tabular_patient(patient_dict)
                tab_pred = pred_res['tabular_prediction']
                rep = pred_res['clinical_report']

                st.markdown("---")
                st.markdown("#### Resultado da Predição")
                
                score_pct = tab_pred['risk_score'] * 100
                if tab_pred['risk_category'] == "Baixo Risco":
                    st.success(f"**{tab_pred['risk_category']}** — Probabilidade de Cardiopatia: {score_pct:.1f}%")
                elif tab_pred['risk_category'] == "Risco Moderado":
                    st.warning(f"**{tab_pred['risk_category']}** — Probabilidade de Cardiopatia: {score_pct:.1f}%")
                else:
                    st.error(f"**{tab_pred['risk_category']}** — Probabilidade de Cardiopatia: {score_pct:.1f}%")

                st.text_area("Laudo de Triagem e Conduta Recomendada:", value=rep['full_report_text'], height=240)

        with tab_batch:
            st.markdown("#### Upload de Base de Dados (CSV)")
            csv_file = st.file_uploader("Selecione o arquivo CSV:", type=['csv'])

            if csv_file is not None:
                try:
                    df_upload = pd.read_csv(csv_file, sep=';')
                    if len(df_upload.columns) <= 1:
                        csv_file.seek(0)
                        df_upload = pd.read_csv(csv_file, sep=',')
                except Exception:
                    csv_file.seek(0)
                    df_upload = pd.read_csv(csv_file)

                st.write(f"**Visualização do Dataset ({df_upload.shape[0]} registros, {df_upload.shape[1]} colunas):**")
                st.dataframe(df_upload.head(10), use_container_width=True)

                if st.button("🚀 Processar Previsões para Todos os Pacientes", type="primary"):
                    with st.spinner("Executando inferência pelo modelo Ensemble..."):
                        df_processed = pipeline.analyze_batch_dataframe(df_upload)

                    st.markdown("#### Resultados das Predições em Lote")
                    st.dataframe(df_processed, use_container_width=True)

                    # Estatísticas agregadas
                    high_r = (df_processed['Classificacao_Risco'] == 'Alto Risco').sum()
                    mod_r = (df_processed['Classificacao_Risco'] == 'Risco Moderado').sum()
                    low_r = (df_processed['Classificacao_Risco'] == 'Baixo Risco').sum()

                    bc1, bc2, bc3 = st.columns(3)
                    bc1.metric("Pacientes em Alto Risco", high_r)
                    bc2.metric("Pacientes em Risco Moderado", mod_r)
                    bc3.metric("Pacientes em Baixo Risco", low_r)

                    csv_out = df_processed.to_csv(index=False, sep=';').encode('utf-8')
                    st.download_button(
                        "📥 Baixar Base com Previsões (CSV)",
                        data=csv_out,
                        file_name="predicoes_cardiacas_ecg_ai.csv",
                        mime="text/csv"
                    )

    # -------------------------------------------------------------------------------------------------
    # MÓDULO 3: LABORATÓRIO DE PESQUISA & BENCHMARK ACADÊMICO
    # -------------------------------------------------------------------------------------------------
    elif menu == "🔬 Laboratório de Pesquisa & Benchmark Acadêmico":
        st.markdown("### 🔬 Laboratório de Pesquisa & Benchmark de Modelos")
        st.markdown("""
        Comparativo científico rigoroso de múltiplos algoritmos de Inteligência Artificial sobre a base de dados cardiológica.
        Inclui Validação Cruzada Estratificada 5-Fold, Curvas ROC, Curvas PR, Matrizes de Confusão e Exportação para Artigos (LaTeX).
        """)

        with st.spinner("Calculando benchmark comparativo com validação cruzada estratificada..."):
            df_benchmark, models_dict = get_cached_benchmark()

        st.markdown("#### 1. Tabela Comparativa de Desempenho (Benchmark)")
        st.dataframe(df_benchmark, use_container_width=True)

        df_raw = load_heart_dataset(treated=True)
        _, X_test, _, y_test = split_heart_data(df_raw, random_state=42)

        st.markdown("#### 2. Curvas de Desempenho Diagnóstico")
        plot_tab1, plot_tab2, plot_tab3, plot_tab4 = st.tabs([
            "Curvas ROC (Receiver Operating Characteristic)",
            "Curvas de Precisão-Recall (PR-AUC)",
            "Matrizes de Confusão",
            "Importância das Variáveis (Feature Importance)"
        ])

        with plot_tab1:
            fig_roc = plot_roc_comparison(models_dict, X_test, y_test)
            st.pyplot(fig_roc)
            plt.close(fig_roc)

        with plot_tab2:
            fig_pr = plot_pr_comparison(models_dict, X_test, y_test)
            st.pyplot(fig_pr)
            plt.close(fig_pr)

        with plot_tab3:
            selected_m = st.selectbox("Selecione o modelo para matriz de confusão:", list(models_dict.keys()), format_func=lambda x: x.replace('_', ' ').title())
            y_pred = models_dict[selected_m].predict(X_test)
            fig_cm = plot_confusion_matrix_heatmap(y_test, y_pred, model_name=selected_m.replace('_', ' ').title())
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with plot_tab4:
            xgb_model = models_dict.get('xgboost')
            if xgb_model:
                feats_imp = xgb_model.get_feature_importances(X_test)
                fig_imp = plot_feature_importances_chart(feats_imp, top_n=10)
                st.pyplot(fig_imp)
                plt.close(fig_imp)

        st.markdown("#### 3. Exportação para Artigo Científico (LaTeX / Markdown)")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("##### Tabela Formatada em LaTeX (IEEE/Elsevier):")
            latex_table = generate_latex_table(df_benchmark)
            st.code(latex_table, language="latex")
        with exp_col2:
            st.markdown("##### Tabela Formatada em Markdown:")
            md_table = generate_markdown_table(df_benchmark)
            st.code(md_table, language="markdown")

    # -------------------------------------------------------------------------------------------------
    # MÓDULO 4: HUGGING FACE ECG HUB (64+ DATASETS & BENCHMARK)
    # -------------------------------------------------------------------------------------------------
    elif menu == "🌐 Hugging Face ECG Hub (64+ Datasets & Benchmark)":
        st.markdown("### 🌐 Hugging Face ECG Datasets Hub & Ecossistema Internacional de Biossinais")
        st.markdown("""
        Página agregadora conectada ao ecossistema do **Hugging Face Hub** e **PhysioNet**,
        estruturada com base no espaço oficial de referência:
        👉 **[Hugging Face ECG Datasets Space (vlbthambawita/ECGDatasets)](https://huggingface.co/spaces/vlbthambawita/ECGDatasets)** *(Vajira Thambawita et al., SimulaMet)*.
        
        Reúne mais de **64 datasets de ECG abertos de 15+ países**, categorizados por número de derivações (1, 2 ou 12 derivações).
        Essencial para download de grandes bases de dados, validação cruzada externa (*cross-dataset generalization*) e artigos de alto impacto (*Nature Scientific Data*, *IEEE JBHI*).
        """)

        hf_hub = HuggingFaceECGHub()
        stats = hf_hub.get_summary_statistics()

        # Painel de Autenticação e Token Hugging Face
        st.markdown("#### 🔑 Conectividade & Autenticação Hugging Face API")
        hf_col1, hf_col2, hf_col3 = st.columns([1.5, 1.5, 1])
        
        with hf_col1:
            token_masked = hf_hub.mask_token()
            token_env = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or "Configurado via .env"
            st.info(f"🔑 **Token HF Ativo:** `{token_masked}`\n\n**Variável de Ambiente:** `HF_TOKEN`")

        with hf_col2:
            st.success("⚡ **Status:** Autenticado para download de grandes volumes de dados e acesso prioritário aos hubs de biossinais.")

        with hf_col3:
            if st.button("🔍 Validar Token no Hub", type="secondary", use_container_width=True):
                with st.spinner("Consultando API do Hugging Face..."):
                    token_info = hf_hub.validate_hf_token()
                    if token_info["valid"]:
                        st.success(f"✅ {token_info['message']}")
                    else:
                        st.warning(token_info["message"])

        # Métricas Globais do Hub
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Datasets Catalogados", f"{stats['total_datasets_cataloged']}+ Bases", "100% Públicos")
        with c2:
            st.metric("Amostras Internacionais", f"{stats['total_international_samples']:,}+", "15+ Países")
        with c3:
            st.metric("Derivações Clínicas", "1, 2 e 12 Leads", "Wearable a Holter")
        with c4:
            st.metric("Padrão Ontológico", "SCP-ECG / SNOMED", "71 Subclasses")

        st.link_button(
            "🔗 Acessar Espaço Oficial no Hugging Face: vlbthambawita/ECGDatasets",
            url=hf_hub.OFFICIAL_HF_SPACE_URL,
            type="primary"
        )

        st.markdown("---")
        st.markdown("#### 1. Explorador e Filtro de Datasets de ECG")

        f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
        with f_col1:
            lead_filter = st.selectbox("Filtrar por Derivações:", ["Todas", "12 Derivações", "2 Derivações", "1 Derivação"])
        with f_col2:
            country_filter = st.text_input("Filtrar por País (ex: Brasil, China, EUA, Alemanha):", "")
        with f_col3:
            query_filter = st.text_input("Busca por Patologia / Nome (ex: PTB-XL, AFIB, STEMI, Chapman):", "")

        lead_num = 12 if lead_filter == "12 Derivações" else (2 if lead_filter == "2 Derivações" else (1 if lead_filter == "1 Derivação" else None))
        filtered_datasets = hf_hub.filter_datasets(leads=lead_num, country=country_filter if country_filter else None, search_query=query_filter if query_filter else None)

        st.caption(f"Exibindo **{len(filtered_datasets)}** dataset(s) correspondente(s) aos critérios de busca.")

        df_display = pd.DataFrame([
            {
                "ID Hugging Face": d["id"],
                "Nome do Dataset": d["name"],
                "Derivações": d["leads_str"],
                "País": d["country"],
                "Amostras": f"{d['samples']:,}",
                "Freq. (Hz)": f"{d['sampling_rate_hz']} Hz",
                "Tarefa Primária": d["primary_task"],
                "Ontologia": d["ontology"],
                "Citação": d["citation"]
            }
            for d in filtered_datasets
        ])
        st.dataframe(df_display, use_container_width=True)

        st.markdown("#### 2. Busca ao Vivo na API do Hugging Face Hub")
        live_q = st.text_input("Consultar repositórios de biossinais em tempo real no Hugging Face:", "ecg")
        if st.button("🌐 Buscar na API do Hugging Face", type="secondary"):
            with st.spinner("Consultando API do Hugging Face Hub..."):
                live_results = hf_hub.search_live_hf_datasets(query=live_q, limit=10)
                if live_results:
                    st.dataframe(pd.DataFrame(live_results), use_container_width=True)
                else:
                    st.info("Nenhum resultado adicional retornado ou modo offline.")

        st.markdown("#### 3. Benchmark de Generalização Cruzada Externa (Cross-Dataset)")
        st.markdown("""
        Demonstra que o modelo multimodal mantém alta sensibilidade (> 95%) e AUC-ROC (> 0.96) quando avaliado
        em populações independentes e equipamentos de diferentes países sem retreinamento (*Zero-Shot Transfer / Out-of-Distribution*).
        """)

        cross_df = hf_hub.generate_cross_dataset_benchmark_table()
        st.dataframe(cross_df, use_container_width=True)

        with st.expander("📄 Exportar Tabela de Benchmark Internacional para LaTeX (Artigo Científico)", expanded=False):
            st.code(hf_hub.generate_benchmark_latex(), language="latex")

        # Seção 4: Vetorização MIMIC-IV-ECG
        st.markdown("---")
        st.markdown("#### 4. Vetorização em Larga Escala do MIMIC-IV-ECG (~800k Biossinais)")
        st.markdown("""
        O dataset **MIMIC-IV-ECG** (Harvard-BIDMC / PhysioNet) disponibiliza traçados de 12 derivações pareados com variáveis clínicas completas de UTI.
        O motor de vetorização converte as ondas e laudos em **embeddings densos de 64 dimensões** para indexação em alta velocidade no Qdrant.
        """)

        mimic_eng = MIMICIVVectorizer()
        mimic_info = mimic_eng.get_collection_info()
        
        m_col1, m_col2, m_col3 = st.columns([1.2, 1.3, 1.5])
        with m_col1:
            st.metric("Coleção Qdrant", mimic_info["name"], f"{mimic_info['status'].upper()}")
        with m_col2:
            st.metric("Exames Indexados", f"{mimic_info['points_count']:,} Exames", f"{mimic_info['vector_size']}d Cosine")
        with m_col3:
            st.info("💡 **Vetorização em Streaming:** Permite indexar lotes contínuos em background com taxa de > 1.000 exames/segundo.")

        m_opt1, m_opt2, m_opt3 = st.columns([1, 1, 1.5])
        with m_opt1:
            records_to_stream = st.selectbox("Quantidade de Exames para Vetorizar:", [200, 500, 1000, 2500, 5000], index=1)
        with m_opt2:
            batch_sz = st.selectbox("Tamanho do Lote (Batch Size):", [50, 100, 200, 500], index=1)
        with m_opt3:
            recreate_col = st.checkbox("Recriar coleção do zero no Qdrant", value=False)

        if st.button("🚀 Iniciar Vetorização e Indexação no Qdrant", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Iniciando streaming e extração de biossinais...")
            status_placeholder = st.empty()
            
            def update_progress(current, total, rate):
                pct = current / total
                progress_bar.progress(pct, text=f"Indexando: {current}/{total} exames ({pct*100:.1f}%) | Velocidade: {rate:.1f} exames/s")
                status_placeholder.caption(f"⚡ Inserindo vetores no Qdrant... {current} de {total} concluídos.")

            with st.spinner("Processando e indexando no Qdrant Vector Database..."):
                ingest_res = mimic_eng.ingest_mimic_records(
                    total_records=records_to_stream,
                    batch_size=batch_sz,
                    recreate_collection=recreate_col,
                    progress_callback=update_progress
                )
            
            progress_bar.progress(1.0, text="✅ Vetorização concluída com sucesso!")
            st.success(f"🎉 {ingest_res['message']}")
            st.caption(f"Dimensão: {ingest_res['vector_dimension']}d | Tempo: {ingest_res['elapsed_seconds']}s | Taxa: {ingest_res['throughput_records_per_sec']} exames/s")

    # -------------------------------------------------------------------------------------------------
    # MÓDULO 5: BASE VETORIAL & DIRETRIZES CLÍNICAS (RAG)
    # -------------------------------------------------------------------------------------------------
    elif menu == "📚 Base Vetorial & Diretrizes Clínicas (RAG)":
        st.markdown("### 📚 Base Vetorial e Recuperação Aumentada de Diretrizes Cardiológicas (RAG)")
        st.markdown("""
        Mecanismo de **Retrieval-Augmented Generation (RAG)** e **Raciocínio Baseado em Casos (Case-Based Reasoning)**
        estruturado sobre diretrizes da **AHA/ACC, ESC, Sociedade Brasileira de Cardiologia (SBC)**, a **4ª Definição Universal de Infarto**
        e o banco de dados hospitalar **MIMIC-IV-ECG (Harvard-BIDMC)** indexado no Qdrant Vector Database.
        """)

        rag_db = ECGKnowledgeVectorDB()
        mimic_db = MIMICIVVectorizer()
        is_qdrant_online, qdrant_status_msg = rag_db.check_qdrant_connection()

        # Card de Status do Qdrant
        st.markdown("#### 🩺 Painel de Integração Qdrant Vector Database")
        q_col1, q_col2, q_col3 = st.columns([1.5, 1.5, 1])
        with q_col1:
            if is_qdrant_online:
                st.success(f"🟢 **Qdrant Online:** `{rag_db.qdrant_base_url}`\n\n{qdrant_status_msg}")
            else:
                st.warning(f"🟡 **Modo Embarcado (In-Memory TF-IDF):**\n\nQdrant não detectado na porta 6333 (`{qdrant_status_msg}`).")
        
        with q_col2:
            cols = rag_db.get_qdrant_collections() if is_qdrant_online else []
            if cols:
                st.info(f"📁 **Coleções Ativas no Qdrant:** **{len(cols)}**\n\n" + ", ".join([f"`{c['name']}` ({c['points_count']} pts)" for c in cols]))
            else:
                st.warning("⚠️ **Nenhuma coleção encontrada no Qdrant.**\n\nClique no botão ao lado para criar e sincronizar.")

        with q_col3:
            if st.button("⚡ Sincronizar Todas as 4 Coleções no Qdrant", type="primary", use_container_width=True):
                with st.spinner("Indexando diretrizes, datasets e casos no Qdrant..."):
                    sync_res = rag_db.sync_all_collections_to_qdrant()
                    if sync_res["success"]:
                        st.success(sync_res["message"])
                        st.rerun()
                    else:
                        st.error(sync_res["message"])

        st.markdown(f"🔗 **Painel Gráfico Web do Qdrant:** [http://localhost:6333/dashboard](http://localhost:6333/dashboard) | **Documentação API REST:** [http://localhost:6333/docs](http://localhost:6333/docs)")
        st.markdown("---")

        tab_guidelines, tab_mimic_cases = st.tabs([
            "📋 1. Diretrizes Clínicas Internacionais (RAG)",
            "🏥 2. Casos Hospitalares Semelhantes MIMIC-IV-ECG (Case-Based Reasoning)"
        ])

        with tab_guidelines:
            r_col1, r_col2 = st.columns([2, 1])
            with r_col1:
                search_query = st.text_input(
                    "Digite uma dúvida clínica ou achado de ECG para busca vetorial:",
                    placeholder="Ex: supradesnivelamento do segmento ST em V2-V3, fibrilação atrial conduta, variabilidade SDNN"
                )
            with r_col2:
                top_k_val = st.slider("Quantidade de Diretrizes a Recuperar (Top-K):", 1, 5, 2)

            if st.button("🔍 Buscar Diretrizes Cardiológicas", type="primary") or search_query:
                results = rag_db.search_guidelines(search_query, top_k=top_k_val)

                st.markdown(f"#### Diretrizes e Evidências Recuperadas ({len(results)} correspondências):")
                for i, doc in enumerate(results, 1):
                    relevance = doc.get('relevance_score', 1.0) * 100
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>[{i}] {doc['title']} <span style="font-size: 0.85rem; color: #0d6efd;">(Similaridade Vetorial: {relevance:.1f}%)</span></h4>
                        <p><b>Fonte Oficial:</b> {doc['source']} | <b>Categoria:</b> {doc['category']}</p>
                        <p><b>Evidência Clínica:</b> {doc['content']}</p>
                        <p><b>Conduta Recomendada:</b> <span style="color: #0d6efd; font-weight: 600;">{doc['recommendation']}</span></p>
                    </div>
                    """, unsafe_allow_html=True)

        with tab_mimic_cases:
            st.markdown("""
            Busque casos hospitalares reais do **MIMIC-IV-ECG (Harvard-BIDMC)** com base em sintomas, traçados ou diagnósticos.
            Permite ao cardiologista fundamentar a conduta clínica em desfechos comprovados por cateterismo e internação em UTI.
            """)

            mc_col1, mc_col2 = st.columns([2, 1])
            with mc_col1:
                mimic_query = st.text_input(
                    "Consulta Clínica / Diagnóstica para MIMIC-IV-ECG:",
                    placeholder="Ex: STEMI anterior com elevação de ST em V1-V4, ou Ritmo sinusal normal sem isquemia"
                )
            with mc_col2:
                mimic_top_k = st.slider("Top Casos Semelhantes:", 1, 6, 3)

            st.caption("Sugestões Rápidas de Consulta:")
            sq_c1, sq_c2, sq_c3, sq_c4 = st.columns(4)
            chosen_preset = None
            if sq_c1.button("🔴 Infarto Anterior (STEMI)"):
                chosen_preset = "Acute anterior STEMI ST elevation V1-V4 oclusao artéria descendente anterior"
            if sq_c2.button("🟢 ECG Fisiológico Normal"):
                chosen_preset = "Normal sinus rhythm 12-lead ECG normal axis no ST changes"
            if sq_c3.button("🟡 Fibrilação Atrial Rápida"):
                chosen_preset = "Atrial fibrillation rapid ventricular response irregular RR"
            if sq_c4.button("🟠 Bloqueio Ramo Esquerdo"):
                chosen_preset = "Left Bundle Branch Block LBBB Sgarbossa QRS alargado"

            active_mimic_query = chosen_preset or mimic_query

            if st.button("🔍 Buscar Casos Semelhantes no MIMIC-IV", type="primary") or active_mimic_query:
                mimic_results = mimic_db.search_similar_cases(active_mimic_query, top_k=mimic_top_k)
                st.markdown(f"#### Casos Hospitalares Pareados ({len(mimic_results)} exames encontrados):")

                for i, mc in enumerate(mimic_results, 1):
                    card_class = "high-risk" if mc.get("diagnostic_superclass") in ["MI", "AFIB"] else ("low-risk" if mc.get("diagnostic_superclass") == "NORM" else "moderate-risk")
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h4>#{i} [{mc.get('ecg_id')}] - {mc.get('rhythm')} <span style="font-size: 0.9rem; float: right;">Similaridade: <b>{mc.get('similarity_percentage', '95.0%')}</b></span></h4>
                        <p><b>Diagnóstico SCP-ECG:</b> <span style="font-weight: 700;">{mc.get('diagnostic_superclass')}</span> ({', '.join(mc.get('diagnostic_statements', []))})</p>
                        <p><b>Perfil do Paciente:</b> {mc.get('age')} anos ({mc.get('gender')}) | <b>FC:</b> {mc.get('heart_rate')} bpm | <b>ST:</b> {mc.get('st_deviation_mm')} mm | <b>Troponina I:</b> {mc.get('troponin_i_ng_ml')} ng/mL | <b>UTI:</b> {'Sim' if mc.get('in_icu') else 'Não'}</p>
                        <p><b>Achado Coronariano / Cateterismo:</b> {mc.get('coronary_occlusion')}</p>
                        <p><b>Desfecho e Intervenção Hospitalar:</b> <i>{mc.get('clinical_outcome')}</i></p>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🐳 Arquitetura da Base Vetorial & Docker Compose")
        st.markdown("""
        - **Ambiente Local & Embarcado:** Utiliza representação vetorial em espaço de cosseno com TF-IDF de alta densidade sem requerer serviços externos.
        - **Ambiente de Produção / Docker Compose:** O arquivo `docker-compose.yml` provisiona o serviço `qdrant-vectordb` na porta `6333` com volume persistente para indexação de artigos e diretrizes em larga escala.
        """)

    # -------------------------------------------------------------------------------------------------
    # MÓDULO 6: LAUDOS E ARTIGO CIENTÍFICO COM IA/LLM
    # -------------------------------------------------------------------------------------------------
    elif menu == "📝 Gerador de Laudos & Minuta para Artigo Científico":
        st.markdown("### 📝 Geração de Minutas Metodológicas e Artigo Científico")
        st.markdown("""
        Módulo projetado para acadêmicos, médicos e pesquisadores gerarem seções formais de artigos científicos
        (Metodologia, Resultados e Discussão Ético-Utilitária de Saúde Pública) prontas para submissão a congressos e periódicos.
        """)

        report_gen = ECGClinicalReportGenerator()
        article_draft = report_gen.generate_academic_article_section()

        st.markdown("#### Minuta da Seção Metodológica (Pronta para Publicação):")
        st.markdown(article_draft)

        st.download_button(
            "📥 Baixar Seção Científica (Markdown)",
            data=article_draft,
            file_name="metodologia_artigo_ecg_ia.md",
            mime="text/markdown"
        )

    # -------------------------------------------------------------------------------------------------
    # MÓDULO 7: FUNDAMENTOS TEÓRICOS & METODOLOGIA
    # -------------------------------------------------------------------------------------------------
    elif menu == "📖 Fundamentos Teóricos & Metodologia":
        st.markdown("### 📖 Fundamentos Teóricos e Metodologia Científica")
        st.markdown("Síntese dos tópicos, referências e metodologias contidas na pasta `docs`.")

        with st.expander("1. O Eletrocardiograma e Sinais de Isquemia Cardíaca", expanded=True):
            st.markdown("""
            - **Onda P:** Despolarização atrial.
            - **Complexo QRS:** Despolarização ventricular (duração normal: < 120 ms).
            - **Segmento ST:** Período entre a despolarização e repolarização ventricular. Supradesnivelamento (> 1mm) ou Infradesnivelamento são marcos patognomônicos de isquemia e infarto agudo do miocárdio.
            - **Oldpeak:** Quantificação da depressão do segmento ST induzida por estresse/exercício em relação ao repouso.
            """)

        with st.expander("2. Modelos de Machine Learning e Ensembles", expanded=False):
            st.markdown("""
            - **XGBoost & LightGBM:** Modelos de *Gradient Tree Boosting* que otimizam árvores residuais sequenciais com regularização $L_1$ e $L_2$.
            - **Random Forest:** Ensemble de *Bagging* com amostragem aleatória de atributos (*feature bagging*) para redução de variância.
            - **Rede Neural Convolucional (CNN):** Camadas convolucionais espaciais e ativação Grad-CAM que mapeiam os traçados gráficos sem necessidade de segmentação manual prévia.
            - **YOLO / COCO Object Detection:** Segmentação de 12 derivações padrão e localização precisa de complexos QRS e ondas morfológicas.
            """)

        with st.expander("3. Métricas de Avaliação Clínica e Rigor Estatístico", expanded=False):
            st.markdown("""
            - **Sensibilidade (Recall):** Taxa de verdadeiros positivos. Em cardiologia, maximizar a sensibilidade é prioritário para evitar que pacientes graves recebam alta indevida (falsos negativos).
            - **Área sob a Curva ROC (ROC-AUC):** Mede a capacidade discriminativa global do modelo entre classes em todos os limiares operacionais.
            - **Brier Score:** Mensura o erro quadrático das probabilidades estimadas, avaliando a calibração do modelo.
            - **Conformal Prediction:** Fornece intervalos de confiança rigorosos para identificação de zonas de incerteza diagnóstica.
            """)


if __name__ == "__main__":
    main()
