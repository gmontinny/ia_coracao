# 🫀 ECG-AI: Sistema Multimodal de Inteligência Artificial para Eletrocardiograma, Base Vetorial RAG e Hub Internacional de Biossinais

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Docker%20Compose-2496ED.svg)](https://www.docker.com/)
[![Vector DB](https://img.shields.io/badge/Vector%20DB-Qdrant%20%2F%20RAG-dc3545.svg)](https://qdrant.tech/)
[![Hugging Face Hub](https://img.shields.io/badge/Hugging%20Face-64+%20ECG%20Datasets-FFD21E.svg)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![YOLO / COCO](https://img.shields.io/badge/YOLO--COCO-Lead%20Detection-blueviolet.svg)](https://ultralytics.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Projeto Acadêmico e Científico**: Sistema de auxílio à decisão clínica multimodal desenvolvido para pesquisa e publicação de artigos nas áreas de Ciência da Computação, Engenharia Biomédica e Cardiologia (*Nature Scientific Data*, *IEEE JBHI*, *The Lancet Digital Health*).

---

## 📌 Visão Geral do Projeto

As doenças cardiovasculares constituem a principal causa de mortalidade em todo o mundo. O diagnóstico precoce e preciso de alterações eletrocardiográficas e isquemia miocárdica é essencial para reduzir desfechos fatais e otimizar recursos hospitalares.

O **ECG-AI** é um ecossistema completo em Python que combina:
1. **Visão Computacional e Filtro de Grade HSV**: Supressão de grade milimetrada de papéis de ECG (cores rosa, laranja, vermelho, verde e cinza) e binarização morfológica para isolar exclusivamente o traçado elétrico do coração.
2. **Segmentação e Detecção de Derivações no Padrão YOLO / COCO**: Localização de caixas delimitadoras (*bounding boxes*) para as 12 derivações padrão (I, II, III, aVR, aVL, aVF, V1-V6), tira de ritmo de DII longo e complexos QRS, ondas P e T.
3. **Leitura Óptica Telemétrica em Quadro Inteiro (OCR / Telemetria)**: Módulo inteligente (`extract_header_metadata` com EasyOCR) que inspeciona o quadro completo do exame (cabeçalho superior, rodapé inferior e margens), extraindo automaticamente a frequência cardíaca impressa (FC), dados do paciente (nome, idade), intervalos eletrocardiográficos (PR, QRS, QT/QTc), eixo elétrico e velocidade do papel ($25\,\text{mm/s}$).
4. **Processamento Digital de Sinais (PDS) Calibrado com Linha de Base Isoelétrica PR**: Reconstrução do sinal de voltagem 1D com isolamento estrito de tira única (evitando contagem dupla de linhas), calibração temporal para folhas completas de 10 segundos ($10.0\,\text{s}$), período refratário fisiológico ($\ge 300\,\text{ms} \le 200\,\text{bpm}$), variabilidade (HRV) e medição rigorosa de desvios do segmento ST referenciados na linha isoelétrica do segmento PR.
5. **Deep Learning Convolucional Calibrado (PyTorch)**: Rede Neural Convolucional profunda com *Temperature Scaling*, estimativa de incerteza epistêmica (*Conformal Prediction*) e mapas de interpretabilidade visual com **Grad-CAM**.
6. **Machine Learning Tabular e Ensembles**: Pipeline completo com **XGBoost, LightGBM, Random Forest, Redes Neurais (MLP), SVM, Regressão Logística e Naive Bayes**, com validação cruzada estratificada 5-Fold e calibração de probabilidade.
7. **Hub Internacional Hugging Face (64+ Datasets)**: Integração com o agregador oficial **[vlbthambawita/ECGDatasets](https://huggingface.co/spaces/vlbthambawita/ECGDatasets)** (Vajira Thambawita et al., SimulaMet), cobrindo bases de 1, 2 e 12 derivações de 15+ países (PTB-XL, Chapman, MIMIC-IV-ECG, CODE-15%, MIT-BIH).
8. **Base Vetorial Qdrant & RAG Cardiológico**: Recuperação semântica e indexação vetorial de diretrizes oficiais (AHA/ACC, ESC, SBC 2024, 4ª Definição Universal de Infarto) e casos de referência para *Case-Based Reasoning*.
9. **Interface Web Interativa (Streamlit)**: Dashboard intuitivo com banner telemétrico em tempo real, análise em lote (CSV), upload de imagens de ECG, gráficos comparativos de curvas ROC/PR, visualizadores YOLO/PDS, painel de sincronização do Qdrant e exportação de tabelas LaTeX para periódicos internacionais (IEEE, Elsevier, Springer, Nature, SBC).

---

## ⚙️ Configuração de Variáveis de Ambiente (.env)

Crie ou edite o arquivo `.env` na raiz do projeto (baseado em `.env.example`) para definir suas chaves de API:

```bash
# Hugging Face Hub Access Token (Para download e streaming de grandes datasets de ECG)
HF_TOKEN=hf_your_token_here
HUGGINGFACE_HUB_TOKEN=hf_your_token_here

# Qdrant Vector Database
VECTOR_DB_HOST=localhost
VECTOR_DB_PORT=6333

# OpenAI API Key (Opcional para laudos avançados)
OPENAI_API_KEY=
```

---

## 🏗️ Estrutura do Repositório

```text
ia_coracao/
├── app.py                      # Aplicação Web interativa Streamlit
├── docker-compose.yml          # Orquestração de containers (App + Qdrant Vector DB)
├── Dockerfile                  # Container da aplicação de IA cardiológica
├── requirements.txt            # Dependências do projeto
├── pytest.ini                  # Configuração da suíte de testes
├── README.md                   # Documentação científica e técnica
├── docs/                       # Base de conhecimento e datasets de referência
│   ├── heart.csv               # Dataset clínico bruto
│   ├── heart_tratado.csv       # Dataset clínico tratado
│   └── *.pdf / *.pptx          # Fundamentação teórica de ML e Cardiologia
├── img/                        # Amostras de traçados de ECG para teste
│   ├── giovanny_01.png         # Traçado normal de referência
│   ├── giovanny_02.png         # Traçado normal de referência
│   ├── 001.png                 # Traçado com supradesnivelamento ST
│   ├── 003.jpg                 # Traçado com infradesnivelamento ST
│   └── 02.jpg                  # Traçado adicional
├── src/                        # Código-fonte modular do sistema
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Loader de dados e pipelines sem vazamento (data leakage)
│   │   ├── hf_dataset_loader.py# Hub Hugging Face (64+ datasets, 15+ países, ontologia SCP-ECG)
│   │   ├── mimic_vectorizer.py # Vetorizador em streaming de biossinais MIMIC-IV-ECG (800k)
│   │   ├── vectorize_mimic.py  # Utilitário CLI para ingestão em lotes e busca k-NN no Qdrant
│   │   ├── sync_qdrant.py      # Sincronizador de coleções vetoriais no Qdrant
│   │   └── ecg_knowledge_rag.py# Base de Conhecimento Vetorial RAG (AHA/ACC/ESC/SBC)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tabular_models.py   # Classificadores ML (XGBoost, LGBM, RF, MLP, SVM)
│   │   ├── ecg_signal_processing.py # PDS, filtro HSV e calibração de ST/PR
│   │   ├── ecg_vision.py       # Deep CNN com Grad-CAM e Conformal Prediction
│   │   ├── ecg_yolo.py         # Detector YOLO de derivações e ondas
│   │   └── llm_report.py       # Gerador de laudos clínicos e minutas científicas com RAG
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── ecg_pipeline.py     # Pipeline multimodal unificado
│   └── utils/
│   │   ├── __init__.py
│   │   ├── visualization.py    # Curvas ROC, PR, Matrizes de Confusão e Feature Importance
│   │   └── academic_metrics.py # Exportação de tabelas LaTeX e intervalos de confiança
└── tests/                      # Suíte de testes unitários e de integração (pytest)
    ├── test_loader.py
    ├── test_models.py
    ├── test_signal_processing.py
    ├── test_pipeline.py
    ├── test_hf_loader.py
    ├── test_mimic_vectorizer.py
    ├── test_knowledge_rag.py
    └── test_academic.py
```

---

## 🧭 Fluxo de Execução Recomendado (Ordem das Etapas)

Para garantir a **máxima acurácia diagnóstica, calibração estatística e respaldo clínico**, o fluxo de trabalho foi desenhado para que a base de conhecimento vetorial esteja indexada antes de realizar a análise da imagem:

```text
[Etapa 1: Infraestrutura] ➔ Iniciar Containers Docker (App + Qdrant Vector DB)
         │
         ▼
[Etapa 2: Vetorização]    ➔ Sincronizar Diretrizes (SBC/ESC) e Vetorizar Biossinais (MIMIC-IV / PTB-XL)
         │
         ▼
[Etapa 3: Análise de ECG] ➔ Upload da Imagem (PDS + HSV + YOLO + Grad-CAM + Recuperação Vetorial k-NN)
```

---

## 📸 Guia de Captura: Como Obter o Melhor Resultado da Imagem de ECG

Para que o pipeline de Visão Computacional, YOLO e Processamento Digital de Sinais atinja o seu desempenho ideal (eliminando ruídos e falsos desvios de ST), siga as seguintes recomendações ao fotografar ou escanear o papel de ECG:

1. **Iluminação Uniforme e Sem Reflexos:**
   - Evite sombras pontuais ou reflexos de lâmpadas sobre o papel termossensível.
2. **Ângulo Perpendicular (90°):**
   - Fotografe ou digitalize o papel de frente (visão superior ortogonal), evitando perspectivas inclinadas que distorçam as proporções milimétricas das ondas.
3. **Enquadramento Completo das 12 Derivações:**
   - Garanta que as 12 derivações padrão (I, II, III, aVR, aVL, aVF, V1 a V6) e a tira de ritmo de DII longo estejam visíveis na imagem.
4. **Resolução e Foco Nítido:**
   - Resolução recomendada de pelo menos $1200 \times 800$ pixels. O traçado preto deve estar nítido contra a grade milimetrada colorida (rosa/vermelha/verde).
5. **Superfície Plana:**
   - Mantenha o papel o mais plano possível, evitando dobras ou amassados sobre os complexos QRS.

---

## 🔍 Leitura Óptica Telemétrica em Quadro Inteiro e Calibração de FC

O sistema incorpora um motor de visão e OCR adaptativo multi-pass (`extract_header_metadata` com EasyOCR e Pytesseract) capaz de inspecionar a **imagem inteira** (cabeçalho superior, rodapé inferior esquerdo/direito, caixas de telemetria ou margens laterais), sem depender de um layout estático fixo:

* **Detecção Automática de Parâmetros Impressos e Telemetria Completa:**
  - **Frequência Cardíaca (FC):** Extração de valores telemétricos oficiais (ex.: `FC: 95 bpm` em `giovanny_03.jpg`, `FC: 84 bpm` em `giovanny_01.png`).
  - **Onda P e Intervalos:** Duração da onda P (`P: 118 ms`), intervalo PR (`PR: 175 ms / 118 ms`), duração do QRS (`QRS: 96 ms / 75 ms`) e intervalos QT/QTc (`QT: 346 ms`, `QTc / QTo: 435 ms`).
  - **Eixos Elétricos Cardíacos:** Eixo de onda P (`SÂP / SAP`), eixo elétrico ventricular (`SÂQRS / SAQRS: 75°`) e eixo de onda T (`SÂT / SAT`).
  - **Dados Cadastrais e Rastreabilidade:** Nome do paciente (`GIOVANNY MONTINNY DE ALMEIDA DANTAS`), idade (`50 anos`), data de nascimento (`13/04/1976`), número do exame (`0000016791` / `62661`), data/hora (`01/09/26 09:03`), instituição hospitalar (`BOM CORAÇÃO CARDIOLOGIA` / `HOSPITAL AMECOR`).
  - **Corpo Clínico e Registro Profissional:** Médico responsável (`Dr(a). FERNANDO POMPEO DE BARROS`), `CRM` (`CRM-MT`) e Registro de Qualificação de Especialista em Cardiologia (`RQE: 2518`).
  - **Configurações de Aquisição:** Velocidade do papel ($25\,\text{mm/s}$ ou $50\,\text{mm/s}$), ganho de voltagem ($10\,\text{mm/mV}$ ou $N/2 = 5\,\text{mm/mV}$) e filtros de sinal (`Rede 60Hz / Filtro Muscular`).
* **Calibração Temporal e Validação Cruzada do Sinal:**
  - **Janela Padrão de 10 Segundos:** Páginas de 12 derivações utilizam calibração temporal de aquisição contínua ($fs = \text{largura} / 10.0\,\text{s}$).
  - **Período Refratário Fisiológico:** Distância mínima entre picos ajustada para $\ge 300\,\text{ms}$ ($\le 200\,\text{bpm}$), prevenindo falsos picos em ondas T ou ruídos de grade.
  - **Isolamento de Tira Única:** Janela de processamento restrita à tira de ritmo inferior para eliminar sobreposição de traçados vizinhos.
  - **Fusão e Concordância:** Quando a telemetria do aparelho está visível, o sistema a cruza com o cálculo de Processamento Digital de Sinais, exibindo ambos na interface e no laudo com 100% de consistência diagnóstica.

---

## 🔄 Procedimento Passo a Passo para Vetorização dos Dados

A vetorização dos biossinais e das diretrizes clínicas no Qdrant é o processo que confere inteligência baseada em evidências ao sistema. Siga os passos abaixo:

### Passo 1: Inicializar o Banco Vetorial Qdrant
Inicie o container do Qdrant Vector Database via Docker Compose:
```bash
docker compose up -d qdrant-vectordb
```
Verifique se a interface gráfica está acessível no navegador: `http://localhost:6333/dashboard`.

### Passo 2: Sincronizar as Diretrizes Clínicas e Catálogo (RAG)
Execute a criação e indexação inicial das diretrizes da SBC, AHA/ACC, ESC e ontologia SCP-ECG:
```bash
# Via terminal
python src/data/sync_qdrant.py

# Ou dentro do container Docker
docker compose exec ecg-ai-app python src/data/sync_qdrant.py
```
*(Também é possível clicar no botão **"⚡ Sincronizar / Criar Coleções no Qdrant"** na aba de RAG do Streamlit).*

### Passo 3: Vetorizar Biossinais Hospitalares (MIMIC-IV-ECG / ~800k exames)
Indexe lotes de biossinais de 12 derivações na coleção `mimic_iv_ecg_800k`:
```bash
# Ingestão em streaming (exemplo com lote de 1.000 a 5.000 exames)
python src/data/vectorize_mimic.py --records 1000 --batch-size 200

# Teste imediato de busca por similaridade k-NN
python src/data/vectorize_mimic.py --search "STEMI supradesnivelamento de ST anterior"
```
*(Ou utilize a barra de progresso de streaming interativa no Módulo 5 do Streamlit).*

---

## 🚀 Como Executar o Projeto

### Opção A: Execução com Docker e Docker Compose (Recomendado)
A plataforma inclui orquestração completa com container da aplicação Streamlit e banco vetorial **Qdrant**:

```bash
# Inicializar a aplicação e o banco vetorial
docker compose up --build

# Para rodar em segundo plano (detached)
docker compose up -d

# Sincronizar e popular as coleções no Qdrant (opcional via terminal ou pela interface)
docker compose exec ecg-ai-app python src/data/sync_qdrant.py
```
Acesse a aplicação em `http://localhost:8501` e a interface gráfica do banco vetorial Qdrant em `http://localhost:6333/dashboard`.

---

### Opção B: Execução Local no Ambiente Virtual

#### 1. Pré-requisitos
- Python 3.10 ou superior
- Ambiente virtual configurado

#### 2. Ativação do Ambiente e Instalação das Dependências
```bash
# Ativar o ambiente virtual (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Instalar dependências
pip install -r requirements.txt

# Sincronizar banco vetorial Qdrant (se o container estiver ativo)
python src/data/sync_qdrant.py
```

#### 3. Vetorização e Indexação em Larga Escala (MIMIC-IV-ECG / Qdrant)
```bash
# Ingestão em lotes via streaming para o Qdrant (coleção mimic_iv_ecg_800k)
python src/data/vectorize_mimic.py --records 1000 --batch-size 200

# Executar busca vetorial semântica de teste (Case-Based Reasoning)
python src/data/vectorize_mimic.py --search "STEMI anterior supradesnivelamento V1-V4"
```

#### 4. Execução da Suíte de Testes Automatizados
```bash
pytest -v
```

#### 5. Inicialização da Interface Web
```bash
streamlit run app.py
```
A aplicação abrirá automaticamente no seu navegador no endereço `http://localhost:8501`.

---

## 📊 Metodologia Científica e Resultados

### Modelos de Aprendizado de Máquina Avaliados
- **XGBoost (Extreme Gradient Boosting)**
- **LightGBM (Light Gradient Boosting Machine)**
- **Random Forest Classifier**
- **Rede Neural Multi-Layer Perceptron (MLP)**
- **Support Vector Machines (SVM)**
- **Regressão Logística Regularizada**
- **Gaussian Naive Bayes**
- **Ensemble de Votação Suave (Soft Voting)**

### Protocolo de Validação
- **Validação Cruzada Estratificada 5-Fold**: Garante proporção idêntica de desfechos positivos e negativos em cada fold, eliminando viés de particionamento.
- **Métricas Focadas na Utilidade Clínica**:
  - **Sensibilidade (Recall)**: Minimização rigorosa de falsos negativos (evitar que pacientes cardiopatas deixem de receber tratamento).
  - **Área sob a Curva ROC (ROC-AUC)** e **Precisão-Recall (PR-AUC)**.
  - **Brier Score**: Avaliação de erro quadrático para calibração probabilística.
  - **Conformal Prediction (95% CI)**: Quantificação de incerteza epistêmica para triagem segura.

---

## 📝 Exportação para Artigos Científicos

A plataforma gera tabelas no formato **LaTeX** padronizadas para submissão direta a periódicos de alto impacto:

```latex
\begin{table*}[t]
\centering
\caption{Performance Comparison of Machine Learning Models for ECG and Cardiovascular Risk Assessment}
\label{tab:ecg_benchmark}
\begin{tabular}{lcccccccc}
\hline
\textbf{Model} & \textbf{Acc (Test)} & \textbf{Acc (CV)} & \textbf{Sensitivity} & \textbf{Specificity} & \textbf{Precision} & \textbf{F1-Score} & \textbf{ROC-AUC} & \textbf{Brier} \\
\hline
Xgboost & 0.887 & 0.865 ± 0.022 & 0.914 & 0.853 & 0.887 & 0.900 & 0.938 & 0.089 \\
Lightgbm & 0.874 & 0.861 ± 0.024 & 0.898 & 0.843 & 0.879 & 0.889 & 0.932 & 0.094 \\
Random Forest & 0.870 & 0.862 ± 0.019 & 0.906 & 0.824 & 0.867 & 0.886 & 0.935 & 0.091 \\
Neural Network Mlp & 0.861 & 0.852 ± 0.020 & 0.883 & 0.833 & 0.870 & 0.876 & 0.925 & 0.103 \\
\hline
\end{tabular}
\end{table*}
```

---

## 🩺 Considerações Éticas e Clínicas
O sistema foi desenvolvido como uma ferramenta de **Apoio à Decisão Clínica (CDSS)** e deve ser empregado em conjunto com a avaliação médica qualificada, priorizando intervenções precoces com maior custo-efetividade e maximização de qualidade de vida (QALYs).

---

## 👥 Contribuição e Publicação
Este projeto destina-se a estudos acadêmicos e publicação de artigos científicos interdisciplinares em Saúde Digital e Inteligência Artificial.
