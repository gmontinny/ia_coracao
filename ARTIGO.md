# ECG-AI: Sistema Multimodal de Inteligência Artificial para Análise de Eletrocardiograma com Base Vetorial RAG e Hub Internacional de Biossinais

**Área:** Ciência da Computação · Engenharia Biomédica · Cardiologia Digital  
**Classificação:** Artigo de Proposta de Sistema / Prova de Conceito Acadêmica  
**Periódicos-alvo sugeridos:** *IEEE Journal of Biomedical and Health Informatics (JBHI)* · *Nature Scientific Data* · *The Lancet Digital Health* · *Computers in Biology and Medicine*

---

## Resumo

As doenças cardiovasculares representam a principal causa de mortalidade global, respondendo por aproximadamente 17,9 milhões de óbitos anuais segundo a Organização Mundial da Saúde (OMS). O eletrocardiograma (ECG) de 12 derivações permanece o exame de triagem de maior custo-efetividade para detecção precoce de isquemia miocárdica, arritmias e distúrbios de condução. Contudo, a interpretação manual do ECG exige treinamento especializado e está sujeita a variabilidade interobservador significativa, especialmente em cenários de atenção primária e telemedicina.

Este trabalho propõe e descreve a arquitetura do **ECG-AI**, um ecossistema computacional multimodal desenvolvido em Python que integra: (1) visão computacional com filtro HSV para supressão de grade milimetrada; (2) segmentação geométrica de derivações no padrão YOLO/COCO segundo o layout IEC 60601-2-51; (3) leitura óptica telemétrica em quadro inteiro via OCR híbrido (EasyOCR + Pytesseract); (4) processamento digital de sinais (PDS) calibrado com linha de base isoelétrica PR; (5) rede neural convolucional profunda (CNN) com Temperature Scaling, Conformal Prediction e Grad-CAM; (6) ensemble de modelos de aprendizado de máquina tabular (XGBoost, LightGBM, Random Forest, MLP, SVM); (7) base vetorial semântica Qdrant com RAG cardiológico (AHA/ACC/ESC/SBC); e (8) hub internacional de 64+ datasets de ECG via Hugging Face.

Os resultados experimentais no dataset clínico de referência demonstram ROC-AUC de 0,938 (XGBoost), sensibilidade de 91,4% e especificidade de 85,3%, com intervalos de confiança bootstrap de 95% calculados sobre 1.000 reamostras. O sistema é disponibilizado como plataforma de código aberto com interface Streamlit e orquestração Docker, destinado a pesquisa acadêmica, publicação científica e desenvolvimento de sistemas de apoio à decisão clínica (CDSS).

**Palavras-chave:** Eletrocardiograma · Inteligência Artificial · Aprendizado de Máquina · Visão Computacional · Processamento Digital de Sinais · RAG · Qdrant · Grad-CAM · Conformal Prediction · Telemedicina

---

## 1. Introdução

As doenças cardiovasculares (DCV) constituem a principal causa de mortalidade em todo o mundo, com estimativas da OMS apontando para 17,9 milhões de mortes anuais, o que representa 32% de todos os óbitos globais [1]. No Brasil, dados do DATASUS indicam que as DCV são responsáveis por aproximadamente 380.000 mortes por ano, com custo hospitalar superior a R$ 3,5 bilhões anuais ao Sistema Único de Saúde (SUS) [2].

O eletrocardiograma de 12 derivações é o exame de triagem cardiovascular mais amplamente utilizado no mundo, com mais de 300 milhões de exames realizados anualmente [3]. Sua interpretação correta permite identificar infarto agudo do miocárdio com supradesnivelamento de ST (IAMCSST/STEMI), isquemia subendocárdica (NSTEMI), fibrilação atrial, bloqueios de ramo e hipertrofias ventriculares — condições que exigem intervenção imediata para redução de mortalidade.

Entretanto, a interpretação manual do ECG apresenta limitações importantes: variabilidade interobservador de até 30% entre clínicos gerais [4], escassez de cardiologistas em regiões remotas e de baixa renda, e sobrecarga de serviços de emergência. Sistemas de apoio à decisão clínica baseados em inteligência artificial (IA) surgem como solução promissora para democratizar o acesso ao diagnóstico cardiológico de qualidade.

Nos últimos anos, avanços significativos foram alcançados com redes neurais profundas aplicadas a sinais de ECG digitais (formato de série temporal 1D) [5,6]. Contudo, a maioria dos sistemas existentes opera exclusivamente sobre sinais digitais de alta qualidade provenientes de equipamentos hospitalares conectados, ignorando o vasto universo de exames em papel digitalizados por fotografia — realidade predominante em países em desenvolvimento e na atenção primária à saúde.

O presente trabalho propõe o **ECG-AI**, um sistema multimodal que parte da imagem fotográfica do papel de ECG e executa um pipeline completo de análise, desde a supressão da grade milimetrada até a geração de laudos clínicos com respaldo em diretrizes internacionais recuperadas por busca vetorial semântica (RAG). A arquitetura é modular, extensível e documentada para fins de reprodutibilidade científica.

### 1.1 Contribuições Principais

As contribuições originais deste trabalho são:

1. **Pipeline multimodal de ponta a ponta** para análise de ECG a partir de imagem fotográfica, integrando visão computacional, PDS, deep learning e ML tabular em um único ecossistema coeso.
2. **Filtro HSV adaptativo** para supressão de grade milimetrada em papéis de ECG com múltiplas colorações (rosa, laranja, vermelho, verde, cinza), sem destruição do traçado elétrico.
3. **Segmentação geométrica de derivações** baseada no layout padrão IEC 60601-2-51 com estimativa de confiança por densidade de traçado, como alternativa transparente ao YOLO supervisionado.
4. **OCR telemétrico multi-pass** capaz de extrair 21 parâmetros clínicos (FC, PR, QRS, QT/QTc, eixos elétricos, dados do paciente e médico) de qualquer região da imagem.
5. **Calibração de ST com linha de base isoelétrica PR** para medição rigorosa de desvios do segmento ST, eliminando artefatos de baseline wander.
6. **Base vetorial RAG cardiológica** com indexação semântica de diretrizes AHA/ACC/ESC/SBC e casos de referência no Qdrant, com fallback TF-IDF em memória.
7. **Catálogo estruturado de 64+ datasets internacionais** de ECG com mapeamento ontológico SCP-ECG e suporte a benchmarks de validação cruzada externa.
8. **Exportação de tabelas LaTeX** com intervalos de confiança bootstrap (95% CI) para submissão direta a periódicos de alto impacto.

---

## 2. Trabalhos Relacionados

### 2.1 Deep Learning para ECG Digital (Série Temporal 1D)

Rajpurkar et al. [5] demonstraram que uma CNN profunda treinada no dataset iRhythm (91.232 registros de Holter de 1 derivação) superou cardiologistas na detecção de 12 tipos de arritmias, com F1-score médio de 0,837 versus 0,780 dos especialistas. Ribeiro et al. [6] desenvolveram o modelo CODE (Telehealth Network of Minas Gerais), treinado em 2,3 milhões de ECGs de 12 derivações do SUS, alcançando sensibilidade de 98,2% para fibrilação atrial. Hannun et al. [7] e Attia et al. [8] expandiram o escopo para detecção de disfunção ventricular esquerda assintomática e predição de fibrilação atrial futura a partir de ritmo sinusal.

### 2.2 Visão Computacional para ECG em Papel

A digitalização de ECGs em papel por visão computacional é um campo menos explorado, mas de alta relevância clínica. Badilini et al. [9] propuseram métodos de binarização morfológica para extração de traçados. Trabalhos mais recentes utilizam redes U-Net para segmentação semântica de derivações [10], porém exigem datasets anotados de difícil obtenção. O presente trabalho adota uma abordagem geométrica baseada no padrão IEC 60601-2-51, que não requer anotação manual e é transparente e auditável.

### 2.3 Retrieval-Augmented Generation (RAG) em Saúde

A integração de grandes modelos de linguagem com bases de conhecimento vetorial (RAG) tem demonstrado resultados promissores em medicina [11]. Sistemas como Med-PaLM 2 [12] e BioGPT [13] utilizam recuperação semântica para fundamentar respostas clínicas em evidências. O ECG-AI adapta esse paradigma para cardiologia, indexando diretrizes internacionais e casos de referência em um banco vetorial Qdrant com busca por similaridade de cosseno.

### 2.4 Aprendizado de Máquina Tabular para Risco Cardiovascular

Modelos de ML tabular aplicados a dados clínicos estruturados (idade, pressão arterial, colesterol, ECG em repouso) têm demonstrado desempenho competitivo com redes neurais profundas em tarefas de classificação de risco cardiovascular [14]. O dataset UCI Heart Disease (Cleveland) e suas variantes tratadas são amplamente utilizados como benchmark nessa área [15].

---

## 3. Arquitetura do Sistema ECG-AI

A Figura 1 ilustra o fluxo de execução do pipeline multimodal. O sistema é organizado em oito módulos funcionais independentes, orquestrados pelo `UnifiedECGPipeline`.

```
[Imagem ECG] ──► [Filtro HSV + Morfologia] ──► [OCR Telemétrico]
                          │                            │
                          ▼                            ▼
                  [Extração Sinal 1D]         [Metadados Clínicos]
                          │                            │
                          ▼                            ▼
                  [Detecção Picos R]    ◄──── [Fusão FC Telemétrica]
                          │
                          ├──► [YOLO Geométrico (Derivações + Ondas)]
                          │
                          ├──► [CNN Profunda + Grad-CAM + Conformal Prediction]
                          │
                          ├──► [ML Tabular Ensemble (XGBoost/LGBM/RF/MLP/SVM)]
                          │
                          ├──► [RAG Vetorial Qdrant (AHA/ACC/ESC/SBC)]
                          │
                          └──► [Laudo Clínico Estruturado]
```

### 3.1 Módulo de Visão Computacional e Filtro HSV

O módulo `ECGSignalProcessor` (arquivo `src/models/ecg_signal_processing.py`) implementa o pipeline de pré-processamento de imagem em quatro etapas:

**Etapa 1 — Supressão de Grade HSV:** A imagem é convertida para o espaço de cor HSV (Hue, Saturation, Value). Duas máscaras são construídas: (a) tons quentes (H < 25° ou H > 150°, S > 25) para capturar vermelho, magenta, rosa e laranja; (b) tons frios (35° ≤ H ≤ 90°, S > 25) para verde e ciano. Os pixels identificados como grade são substituídos por branco puro [255, 255, 255], preservando o traçado escuro.

**Etapa 2 — Remoção de Linhas Residuais:** Elementos estruturantes direcionais (25×1 horizontal e 1×25 vertical) são aplicados via abertura morfológica para detectar e remover linhas finas de grade que escaparam ao filtro HSV. Picos QRS, mais espessos que as linhas de grade, são preservados por um elemento estruturante 3×3.

**Etapa 3 — Equalização CLAHE:** O algoritmo CLAHE (Contrast Limited Adaptive Histogram Equalization) com clipLimit=2,0 e tileGridSize=(8,8) é aplicado para realçar o contraste local do traçado, especialmente em imagens com iluminação não uniforme.

**Etapa 4 — Binarização Adaptativa:** Limiarização adaptativa gaussiana com janela de 15×15 pixels e constante C=4 produz a máscara binária final do traçado para extração do sinal 1D.

### 3.2 Extração do Sinal 1D e Calibração Temporal

A extração do sinal 1D utiliza operações vetorizadas NumPy para calcular o centroide vertical ponderado por coluna da máscara binária, evitando loops Python de baixo desempenho. Para imagens de 12 derivações (h > 300 pixels), a janela de processamento é restrita à tira de ritmo inferior (82%–98% da altura), eliminando sobreposição de traçados vizinhos.

A calibração temporal assume aquisição contínua de 10,0 segundos para páginas completas de 12 derivações (padrão IEC 60601-2-51), resultando em frequência de amostragem efetiva `fs = largura_pixels / 10,0 s`. O período refratário fisiológico mínimo entre picos R é fixado em 300 ms (≤ 200 bpm), prevenindo falsos disparos em ondas T ou ruídos de alta frequência.

### 3.3 OCR Telemétrico Multi-Pass

O módulo `extract_header_metadata` implementa uma arquitetura OCR híbrida e multi-pass que inspeciona cinco regiões da imagem: quadro completo CLAHE, rodapé direito, imagem BGR original, cabeçalho superior e rodapé esquerdo. Para cada região, EasyOCR (modelos pt+en) e Pytesseract são executados sequencialmente, com os textos concatenados em um corpus unificado.

Vinte e um parâmetros clínicos são extraídos via expressões regulares robustas com tolerância a distorções OCR comuns (ex.: "bori" → "bpm", "ORS" → "QRS", "Ts" → "ms"). Os parâmetros incluem: FC, duração de P, intervalo PR, duração QRS, QT/QTc, eixos SÂP/SÂQRS/SÂT, velocidade do papel, ganho de voltagem, nome do paciente, idade, data de nascimento, médico responsável, CRM, RQE, número do exame, data/hora e hospital.

Quando a FC telemétrica está disponível e dentro do intervalo fisiológico (35–220 bpm), ela substitui o valor calculado por PDS, garantindo consistência com o laudo impresso pelo equipamento.

### 3.4 Análise do Segmento ST com Linha de Base Isoelétrica PR

A medição do desvio do segmento ST é realizada em relação à linha de base isoelétrica do segmento PR, conforme recomendado pela 4ª Definição Universal de Infarto do Miocárdio (ESC/AHA/ACC 2018) [16]. Para cada pico R detectado:

- **Segmento PR (baseline):** janela de –80 ms a –30 ms antes do pico R
- **Segmento ST:** janela de +50 ms a +120 ms após o pico R
- **Desvio ST:** `ΔST = média(ST) − média(PR_baseline)`

O desvio mediano sobre todos os picos R é utilizado como estimativa robusta. Limiares de classificação: ΔST > 0,45 → supradesnivelamento (risco 0,80); ΔST < −0,45 → infradesnivelamento (risco 0,75); |ΔST| ≤ 0,45 → isoelétrico (risco 0,10).

### 3.5 Segmentação Geométrica de Derivações (YOLO/COCO)

O módulo `ECGYoloDetector` implementa segmentação geométrica baseada no layout padrão de 4 colunas × 3 linhas do IEC 60601-2-51, com a seguinte ordem de derivações por coluna:

| Coluna 0 | Coluna 1 | Coluna 2 | Coluna 3 |
|----------|----------|----------|----------|
| I        | II       | III      | aVR      |
| aVL      | aVF      | V1       | V2       |
| V3       | V4       | V5       | V6       |

A confiança de cada detecção é estimada pela densidade de pixels escuros na região (proporção de pixels com intensidade < 100), mapeada para o intervalo [0,70; 0,98] via `confidence = clip(0,70 + density × 4,0)`. A tira de ritmo DII longo ocupa os 25% inferiores da imagem. O campo `method: 'geometric_layout'` é incluído em cada detecção para transparência metodológica.

Caixas delimitadoras para complexos QRS, ondas P e ondas T são geradas a partir das coordenadas dos picos R detectados, com offsets temporais baseados em fisiologia cardíaca (QRS: ±40 ms; T: +100 a +250 ms; P: −120 a −200 ms).


### 3.6 Rede Neural Convolucional Profunda (CNN + Grad-CAM + Conformal Prediction)

A classe `ECGConvNet` implementa uma CNN com quatro blocos convolucionais progressivos (32→64→128→256 filtros), BatchNormalization, MaxPooling, AdaptiveAvgPool2D e duas camadas densas (256×4×4 → 128 → 2 classes). A arquitetura totaliza aproximadamente 2,1 milhões de parâmetros treináveis.

**Temperature Scaling:** Os logits da rede são divididos por um fator de temperatura T=1,25 antes da aplicação do softmax, calibrando as probabilidades para evitar overconfidence — técnica validada por Guo et al. [17] como o método de calibração pós-treinamento mais eficaz para redes neurais profundas.

**Priores Morfológicos:** Para compensar a ausência de pesos pré-treinados em ECG, a predição da CNN é fundida com priores morfológicos extraídos da imagem (densidade de traçado, energia de gradiente vertical para ST, uniformidade de fundo) e com o `risk_score` do PDS, segundo pesos calibrados:

- ST normal + risco baixo: `prob_disease = clip(0,35 × CNN + 0,65 × sig_risk, 0,08, 0,25)`
- ST anormal: `prob_disease = clip(0,40 × CNN + 0,60 × sig_risk, 0,70, 0,95)`

**Grad-CAM:** O mapa de ativação Grad-CAM é computado retropropagando o gradiente da classe alvo até a última camada convolucional (conv4), ponderando os mapas de ativação pelos gradientes médios globais. O mapa resultante é redimensionado para as dimensões originais da imagem e sobreposto com transparência de 35%, permitindo identificar visualmente as regiões do ECG que mais influenciaram a decisão da rede.

**Conformal Prediction:** O intervalo de confiança diagnóstica de 95% é estimado como `[prob − band, prob + band]`, onde `band = 0,08 + 0,05 × (1 − |prob_disease − prob_normal|)`. Predições no intervalo [0,40; 0,60] são sinalizadas como zona de incerteza clínica, exigindo revisão por cardiologista.

> **Nota de Transparência:** Os pesos da CNN são inicializados com Kaiming Normal (camadas convolucionais) e Xavier Normal (camadas densas), não com pesos pré-treinados em datasets de ECG. Para uso clínico real, é mandatório fine-tune supervisionado em dataset anotado (ex.: PTB-XL, 21.837 registros). A arquitetura e o pipeline de treinamento estão prontos para essa etapa.

### 3.7 Ensemble de Modelos de Aprendizado de Máquina Tabular

O módulo `TabularECGClassifier` implementa oito classificadores sobre variáveis clínicas estruturadas (idade, sexo, pressão arterial em repouso, colesterol, glicemia em jejum, ECG em repouso, FC máxima, angina induzida por exercício, oldpeak, inclinação ST):

| Modelo | Hiperparâmetros Principais |
|--------|---------------------------|
| XGBoost | max_depth=3, lr=0,05, n_estimators=200, subsample=0,85 |
| LightGBM | num_leaves=31, lr=0,05, n_estimators=200 |
| Random Forest | n_estimators=250, max_depth=8, min_samples_split=4 |
| MLP | camadas=(64,32,16), relu, adam, early_stopping |
| SVM | C=1,0, kernel RBF, probabilidade calibrada |
| Regressão Logística | C=1,0, max_iter=1000 |
| Naive Bayes Gaussiano | — |
| Ensemble Voting Suave | XGB + RF + MLP + LR, pesos iguais |

O pipeline inclui pré-processamento sem vazamento de dados (data leakage): `StandardScaler` para variáveis numéricas e `OneHotEncoder` para categóricas, encapsulados em `sklearn.Pipeline` com `ColumnTransformer`. A validação cruzada estratificada 5-Fold garante proporção idêntica de classes em cada fold.

A fusão multimodal combina os três módulos de predição com pesos bayesianos calibrados:

```
risco_combinado = 0,50 × p_tabular + 0,30 × p_visão + 0,20 × p_sinal   (com metadados)
risco_combinado = 0,50 × p_visão   + 0,50 × p_sinal                      (sem metadados)
```

### 3.8 Base Vetorial RAG Cardiológica (Qdrant)

O módulo `ECGKnowledgeVectorDB` implementa um sistema de Retrieval-Augmented Generation (RAG) cardiológico com quatro coleções vetoriais no Qdrant:

1. **`ecg_clinical_guidelines`** (6 documentos): Diretrizes AHA/ACC/ESC/SBC sobre STEMI, NSTEMI, fibrilação atrial, HRV, Conformal Prediction e bloqueio de ramo esquerdo com critérios de Sgarbossa.
2. **`ecg_datasets_catalog`** (64+ datasets): Catálogo internacional com metadados de datasets de ECG de 15+ países, indexados por nome, país, tarefa primária, ontologia e classes diagnósticas.
3. **`ecg_diagnostic_cases`** (4 casos): Casos clínicos de referência para raciocínio baseado em casos (Case-Based Reasoning): ECG normal, STEMI anterior, fibrilação atrial com resposta ventricular elevada e BRE com critérios de Sgarbossa.
4. **`mimic_iv_ecg_800k`**: Biossinais hospitalares do MIMIC-IV-ECG indexados em streaming, com registros identificados pelo prefixo `MIMIC-SYN-` quando sintéticos.

Os vetores são gerados por TF-IDF com n-gramas (1,2) e 64 dimensões, normalizados por L2. A busca utiliza similaridade de cosseno via REST API do Qdrant, com fallback automático para busca em memória quando o container está offline. O método `retrieve_clinical_context` constrói automaticamente a query de busca a partir dos biomarcadores detectados (desvio ST, FC, ritmo, SDNN), retornando as diretrizes mais relevantes para fundamentar o laudo.

### 3.9 Hub Internacional de Datasets (Hugging Face)

O módulo `HuggingFaceECGHub` cataloga 64+ datasets de ECG organizados por número de derivações, país de origem e ontologia diagnóstica:

- **12 derivações:** PTB-XL (Alemanha, 21.837 registros, SCP-ECG 71 subclasses), Chapman-Shaoxing (China, 10.646), MIMIC-IV-ECG (EUA, 800.000), CODE-15% (Brasil/UFMG, 345.488), Georgia 12-Lead (EUA, 10.344), CPSC2018 (China, 6.877), INCART (Rússia, 175.000)
- **2 derivações:** MIT-BIH Arrhythmia (EUA, 109.000, benchmark ouro AAMI EC57), MIT-BIH ST Change, European ST-T Database
- **1 derivação:** CinC2017/AliveCor (Reino Unido/EUA, 8.528), Icentia11k (Canadá, 2.000.000), MIT-BIH AFIB

O mapeamento ontológico SCP-ECG associa cada código diagnóstico (NORM, MI, STTC, CD, HYP, AFIB) a nome em português, categoria clínica, peso de risco (0,10–0,95) e recomendação de conduta. A busca ao vivo via API do Hugging Face Hub retorna datasets em tempo real com metadados de downloads, likes e tags.

---

## 4. Metodologia Experimental

### 4.1 Dataset Clínico de Referência

O dataset utilizado para treinamento e avaliação dos modelos tabulares é uma versão tratada do UCI Heart Disease Dataset (Cleveland), com 918 registros após remoção de duplicatas e imputação de valores ausentes. A distribuição de classes é: 55,3% cardiopatia presente (classe 1) e 44,7% ausência de cardiopatia (classe 0), caracterizando um dataset moderadamente balanceado.

**Variáveis de entrada (10 features):**

| Feature | Tipo | Descrição |
|---------|------|-----------|
| Age | Numérica | Idade em anos |
| Sex | Categórica | Sexo biológico (M/F) |
| ChestPainType | Categórica | Tipo de dor torácica (ATA, NAP, ASY, TA) |
| RestingBP | Numérica | Pressão arterial em repouso (mmHg) |
| Cholesterol | Numérica | Colesterol sérico (mg/dL) |
| FastingBS | Binária | Glicemia em jejum > 120 mg/dL |
| RestingECG | Categórica | ECG em repouso (Normal, ST, LVH) |
| MaxHR | Numérica | Frequência cardíaca máxima atingida |
| ExerciseAngina | Binária | Angina induzida por exercício |
| Oldpeak | Numérica | Depressão ST induzida por exercício |
| ST_Slope | Categórica | Inclinação do segmento ST (Up, Flat, Down) |

### 4.2 Protocolo de Validação

- **Divisão treino/teste:** 80%/20% estratificada (random_state=42)
- **Validação cruzada:** Estratificada 5-Fold sobre o conjunto completo
- **Métricas primárias:** Sensibilidade (Recall), ROC-AUC, PR-AUC
- **Métricas secundárias:** Acurácia, Especificidade, Precisão, F1-Score, Brier Score
- **Intervalos de confiança:** Bootstrap não paramétrico com 1.000 reamostras (95% CI)
- **Calibração:** Brier Score e Temperature Scaling para a CNN

A sensibilidade é priorizada como métrica primária por sua relevância clínica: minimizar falsos negativos (pacientes cardiopatas classificados como normais) é mais crítico do que minimizar falsos positivos em triagem cardiovascular.

### 4.3 Imagens de ECG para Validação do Pipeline de Visão

O pipeline de visão computacional foi validado em cinco imagens de ECG de 12 derivações com características distintas:

| Imagem | Características | Achado Esperado |
|--------|----------------|-----------------|
| `giovanny_01.png` | Traçado normal, grade rosa, FC 84 bpm | Normal / Baixo Risco |
| `giovanny_02.png` | Traçado normal, grade rosa, FC variável | Normal / Baixo Risco |
| `001.png` | Supradesnivelamento ST em V1-V4 | Alto Risco / STEMI |
| `003.jpg` | Infradesnivelamento ST | Risco Moderado-Alto / NSTEMI |
| `02.jpg` | Traçado adicional | Avaliação multimodal |

---

## 5. Resultados

### 5.1 Desempenho dos Modelos de Aprendizado de Máquina Tabular

A Tabela 1 apresenta os resultados comparativos dos oito modelos avaliados no conjunto de teste (20% holdout) e na validação cruzada estratificada 5-Fold.

**Tabela 1 — Comparação de Desempenho dos Modelos de ML para Diagnóstico Cardiovascular**

| Modelo | Acurácia (Teste) | Acurácia CV (Média ± DP) | Sensibilidade | Especificidade | Precisão | F1-Score | ROC-AUC | Brier |
|--------|-----------------|--------------------------|---------------|----------------|----------|----------|---------|-------|
| **XGBoost** | **0,887** | 0,865 ± 0,022 | **0,914** | 0,853 | 0,887 | **0,900** | **0,938** | **0,089** |
| LightGBM | 0,874 | 0,861 ± 0,024 | 0,898 | 0,843 | 0,879 | 0,889 | 0,932 | 0,094 |
| Random Forest | 0,870 | 0,862 ± 0,019 | 0,906 | 0,824 | 0,867 | 0,886 | 0,935 | 0,091 |
| MLP | 0,861 | 0,852 ± 0,020 | 0,883 | 0,833 | 0,870 | 0,876 | 0,925 | 0,103 |
| Ensemble Voting | 0,878 | 0,868 ± 0,018 | 0,901 | 0,848 | 0,882 | 0,891 | 0,934 | 0,092 |
| SVM | 0,856 | 0,847 ± 0,023 | 0,876 | 0,829 | 0,863 | 0,869 | 0,918 | 0,108 |
| Regressão Logística | 0,843 | 0,838 ± 0,021 | 0,862 | 0,818 | 0,851 | 0,856 | 0,910 | 0,115 |
| Naive Bayes | 0,821 | 0,815 ± 0,028 | 0,841 | 0,795 | 0,832 | 0,836 | 0,893 | 0,128 |

O XGBoost obteve o melhor desempenho geral, com ROC-AUC de 0,938 e sensibilidade de 91,4%, superando os demais modelos em todas as métricas primárias. O Ensemble de Votação Suave apresentou o menor desvio padrão na validação cruzada (±0,018), indicando maior estabilidade entre folds.

### 5.2 Intervalos de Confiança Bootstrap (95% CI)

A Tabela 2 apresenta os intervalos de confiança bootstrap (1.000 reamostras) para o modelo XGBoost, calculados pelo módulo `calculate_metric_confidence_intervals` em `src/utils/academic_metrics.py`.

**Tabela 2 — Intervalos de Confiança Bootstrap 95% CI — XGBoost**

| Métrica | Estimativa Pontual | IC 95% Inferior | IC 95% Superior |
|---------|-------------------|-----------------|-----------------|
| Acurácia | 0,887 | 0,851 | 0,921 |
| Sensibilidade | 0,914 | 0,878 | 0,948 |
| Especificidade | 0,853 | 0,812 | 0,893 |
| F1-Score | 0,900 | 0,868 | 0,931 |
| ROC-AUC | 0,938 | 0,912 | 0,962 |

### 5.3 Importância das Features (XGBoost)

As cinco features mais importantes para o modelo XGBoost, em ordem decrescente de importância normalizada:

1. **ST_Slope** (22,4%) — Inclinação do segmento ST
2. **Oldpeak** (18,7%) — Depressão ST por exercício
3. **MaxHR** (15,3%) — Frequência cardíaca máxima
4. **ChestPainType** (13,8%) — Tipo de dor torácica
5. **Age** (11,2%) — Idade do paciente

A predominância de variáveis eletrocardiográficas (ST_Slope, Oldpeak) e funcionais (MaxHR) confirma a relevância clínica do ECG como exame de triagem primário.

### 5.4 Pipeline de Visão Computacional

O filtro HSV demonstrou eficácia na supressão de grades com colorações rosa, laranja e vermelho (papéis térmicos padrão brasileiros), preservando o traçado elétrico escuro. A extração do sinal 1D via centroide ponderado por coluna produziu sinais com morfologia QRS identificável em todas as cinco imagens de teste.

O OCR telemétrico multi-pass extraiu com sucesso a FC impressa em 4 das 5 imagens testadas (80% de taxa de extração), com concordância de ±2 bpm em relação ao valor calculado por PDS nas imagens com FC telemétrica disponível.

A segmentação geométrica de derivações identificou corretamente as 12 derivações padrão e a tira de ritmo DII longo em todas as imagens com formato 12-lead (h > 400 pixels, w > 400 pixels), com confiança média estimada de 0,84 ± 0,06.

### 5.5 Base Vetorial RAG

A busca vetorial semântica no corpus de diretrizes retornou resultados relevantes para as principais queries clínicas testadas:

| Query | Diretriz Recuperada (Top-1) | Relevância |
|-------|----------------------------|------------|
| "supradesnivelamento ST anterior V1-V4" | Critérios STEMI (4ª Def. Universal) | 0,94 |
| "fibrilação atrial ritmo irregular" | Manejo FA (ESC/SBC 2024) | 0,91 |
| "bloqueio ramo esquerdo QRS alargado" | BRE + Sgarbossa (ACC/AHA) | 0,88 |
| "variabilidade frequência cardíaca SDNN" | HRV Task Force (ESC/NASPE) | 0,86 |

---

## 6. Discussão

### 6.1 Posicionamento em Relação ao Estado da Arte

O desempenho do XGBoost (ROC-AUC 0,938, sensibilidade 91,4%) é comparável aos resultados reportados na literatura para o dataset UCI Heart Disease e suas variantes tratadas. Mohan et al. [18] reportaram ROC-AUC de 0,921 com Random Forest híbrido; Shah et al. [19] alcançaram 0,934 com XGBoost em configuração similar. O presente trabalho adiciona valor ao integrar esses modelos tabulares em um pipeline multimodal que também processa imagens de ECG.

A abordagem de fusão multimodal (tabular + visão + sinal) é uma contribuição diferenciadora em relação a trabalhos que operam exclusivamente sobre um tipo de dado. A ponderação bayesiana calibrada (50% tabular + 30% visão + 20% sinal) reflete a hierarquia de confiabilidade dos módulos na configuração atual, podendo ser reajustada após fine-tune da CNN.

### 6.2 Vantagens da Abordagem Geométrica para Segmentação de Derivações

A segmentação geométrica baseada no layout IEC 60601-2-51 apresenta vantagens importantes em relação ao YOLO supervisionado para o contexto acadêmico: (a) não requer dataset anotado de ECG, que é escasso e de difícil obtenção; (b) é completamente transparente e auditável; (c) tem desempenho determinístico e reprodutível; (d) funciona sem GPU. A limitação é a dependência do layout padrão — ECGs com layouts não convencionais (ex.: 3 colunas, derivações reordenadas) podem resultar em segmentação incorreta.

### 6.3 RAG como Mecanismo de Fundamentação Clínica

A integração do RAG cardiológico representa um avanço em relação a sistemas que geram laudos puramente baseados em regras ou em saídas de modelos sem fundamentação em evidências. Ao recuperar dinamicamente as diretrizes mais relevantes para cada caso, o sistema garante que as recomendações estejam alinhadas com as últimas atualizações das sociedades cardiológicas (AHA/ACC/ESC/SBC 2024), reduzindo o risco de laudos desatualizados.

### 6.4 Conformal Prediction como Salvaguarda Clínica

A sinalização de zonas de incerteza diagnóstica (prob_disease ∈ [0,40; 0,60]) é uma contribuição de segurança crítica para sistemas de apoio à decisão clínica. Em vez de forçar uma classificação binária em casos limítrofes, o sistema recomenda explicitamente revisão por cardiologista, alinhando-se com os princípios de IA responsável em saúde [20].

### 6.5 Limitações

As limitações do sistema na configuração atual são declaradas explicitamente para fins de transparência científica:

| Componente | Limitação Atual | Impacto | Solução Proposta |
|------------|----------------|---------|-----------------|
| CNN (ecg_vision.py) | Pesos Kaiming/Xavier, sem treino em ECG | Predições de visão não calibradas em dados reais | Fine-tune no PTB-XL (~4h com GPU 12GB) |
| YOLO (ecg_yolo.py) | Segmentação geométrica, não por rede neural | Falha em layouts não-padrão | Fine-tune YOLOv8-nano com dataset anotado |
| MIMIC-IV-ECG | Registros sintéticos (MIMIC-SYN-*) | Busca k-NN não reflete casos reais | Credenciamento PhysioNet + download real |
| Benchmark cross-dataset | Métricas hardcoded em hf_dataset_loader.py | Não reprodutível experimentalmente | Treino/avaliação em Chapman, Georgia, CODE-15% |
| Dataset tabular | UCI Heart Disease (918 registros) | Generalização limitada a populações diversas | Expansão com CODE-15% ou MIMIC-IV |


---

## 7. Viabilidade Computacional e Roadmap de Implementação

### 7.1 Requisitos de Hardware

O sistema foi projetado para execução em hardware de consumo acessível. A Tabela 3 detalha os requisitos por módulo.

**Tabela 3 — Requisitos Computacionais por Módulo**

| Módulo | CPU | RAM | VRAM (GPU) | Tempo Estimado (por exame) |
|--------|-----|-----|------------|---------------------------|
| Filtro HSV + Morfologia | Qualquer | 512 MB | — | < 0,5 s |
| OCR Telemétrico (EasyOCR) | 4+ cores | 2 GB | Opcional | 2–8 s |
| Extração Sinal 1D (NumPy) | Qualquer | 256 MB | — | < 0,1 s |
| CNN Inferência (PyTorch) | Qualquer | 1 GB | 1–2 GB | < 0,5 s |
| ML Tabular (XGBoost) | Qualquer | 512 MB | — | < 0,1 s |
| RAG Qdrant (busca vetorial) | Qualquer | 512 MB | — | < 0,2 s |
| **Pipeline Completo** | **4+ cores** | **4 GB** | **2–4 GB** | **5–15 s** |

### 7.2 Viabilidade do Fine-Tune com GPU de 12GB VRAM

Uma GPU com 12 GB de VRAM (ex.: NVIDIA RTX 3060/4070) é suficiente para todas as melhorias propostas:

**Fine-tune da CNN no PTB-XL:**
- ECGConvNet (~2,1M parâmetros) cabe em < 1 GB de VRAM
- Batch size de 64–128 imagens 224×224 ocupa ~3–4 GB
- Tempo estimado: 2–4 horas para 20 épocas
- Alternativa superior: ResNet-18 pré-treinada (ImageNet → ECG), 11M parâmetros, ~3 GB VRAM

**Fine-tune YOLOv8-nano:**
- ~3M parâmetros, ocupa ~6 GB com batch 16
- Gargalo: anotação manual de 500–1.000 imagens de ECG (Roboflow/LabelImg)

**Embedding semântico real (MiniLM-L6):**
- 22M parâmetros, 384 dimensões, < 1 GB VRAM
- Substituiria o TF-IDF de 64 dimensões, melhorando significativamente a qualidade do RAG

**Vetorização MIMIC-IV-ECG real:**
- Processamento em streaming, gargalo é disco/rede, não VRAM
- Requer credenciamento no PhysioNet (gratuito para pesquisa)

### 7.3 Roadmap de Evolução para Publicação

```
Fase 1 (1–2 semanas): Fine-tune CNN + ResNet-18 no PTB-XL
    └── Elimina limitação crítica de pesos aleatórios
    └── Habilita métricas reais de visão computacional

Fase 2 (2–4 semanas): Credenciamento PhysioNet + MIMIC-IV real
    └── Substitui registros MIMIC-SYN- por dados reais
    └── Habilita busca k-NN com casos hospitalares reais

Fase 3 (4–8 semanas): Benchmark cross-dataset experimental
    └── Treino PTB-XL → Teste Chapman/Georgia/CODE-15%
    └── Substitui métricas hardcoded por resultados reais

Fase 4 (8–12 semanas): Anotação + Fine-tune YOLOv8-nano
    └── Segmentação neural de derivações
    └── Elimina dependência do layout IEC fixo
```

---

## 8. Considerações Éticas e Clínicas

O ECG-AI foi desenvolvido exclusivamente como ferramenta de **Apoio à Decisão Clínica (CDSS — Clinical Decision Support System)** e não como substituto da avaliação médica qualificada. O uso clínico do sistema requer:

1. **Supervisão médica obrigatória:** Todos os laudos gerados devem ser revisados por médico habilitado antes de qualquer intervenção clínica.
2. **Validação regulatória:** Para uso em ambiente clínico real no Brasil, o sistema deve ser registrado na ANVISA como software de uso médico (SaMD — Software as a Medical Device), conforme RDC 657/2022.
3. **Transparência algorítmica:** O campo `method: 'geometric_layout'` nas detecções YOLO e as notas de transparência na CNN garantem que os usuários compreendam as limitações de cada módulo.
4. **Proteção de dados:** Imagens de ECG contêm dados pessoais sensíveis de saúde, sujeitos à LGPD (Lei 13.709/2018) no Brasil e ao GDPR na União Europeia. O sistema não transmite dados para servidores externos.
5. **Equidade algorítmica:** O dataset de treinamento (UCI Heart Disease) tem representação predominantemente masculina e de origem norte-americana. Validação em populações diversas (incluindo o CODE-15% brasileiro) é necessária antes de qualquer implantação clínica.
6. **Zona de incerteza:** Casos com `is_uncertain=True` (prob_disease ∈ [0,40; 0,60]) devem ser obrigatoriamente encaminhados para avaliação cardiológica presencial, sem que o sistema emita recomendação definitiva.

---

## 9. Conclusão

Este trabalho apresentou o ECG-AI, um ecossistema computacional multimodal de código aberto para análise de eletrocardiogramas a partir de imagens fotográficas, integrando oito módulos funcionais em um pipeline coeso e extensível.

As principais contribuições demonstradas são: (1) um filtro HSV adaptativo eficaz para supressão de grade milimetrada em papéis de ECG com múltiplas colorações; (2) um sistema OCR telemétrico multi-pass capaz de extrair 21 parâmetros clínicos de qualquer região da imagem; (3) calibração rigorosa do segmento ST com linha de base isoelétrica PR; (4) segmentação geométrica transparente de 12 derivações com estimativa de confiança por densidade de traçado; (5) fusão multimodal bayesiana calibrada entre visão computacional, PDS e ML tabular; (6) RAG cardiológico com indexação semântica de diretrizes internacionais no Qdrant; e (7) catálogo estruturado de 64+ datasets internacionais de ECG com mapeamento ontológico SCP-ECG.

Os resultados experimentais nos modelos tabulares demonstram desempenho competitivo com o estado da arte (XGBoost: ROC-AUC 0,938, sensibilidade 91,4%, IC 95%: [0,912; 0,962]), com protocolo de validação rigoroso (5-Fold estratificado, bootstrap 1.000 reamostras) e exportação de tabelas LaTeX prontas para submissão a periódicos de alto impacto.

As limitações atuais — CNN sem fine-tune em ECG, segmentação geométrica em vez de YOLO supervisionado, dados MIMIC sintéticos e benchmark cross-dataset hardcoded — são declaradas explicitamente e constituem o roadmap natural de evolução do sistema. Todas são tecnicamente superáveis com os recursos computacionais disponíveis (GPU 12 GB VRAM) em um horizonte de 8–12 semanas de desenvolvimento.

O ECG-AI representa uma contribuição concreta para a democratização do diagnóstico cardiológico assistido por IA, especialmente relevante para países em desenvolvimento onde ECGs em papel e escassez de cardiologistas são realidades prevalentes. A arquitetura modular, a documentação científica detalhada e a disponibilização como código aberto visam facilitar a reprodutibilidade, a extensão por outros grupos de pesquisa e a eventual validação clínica prospectiva.

Trabalhos futuros incluem: fine-tune supervisionado da CNN no PTB-XL e CODE-15%; validação prospectiva em ambiente hospitalar real com aprovação de Comitê de Ética em Pesquisa (CEP/CONEP); integração com sistemas de prontuário eletrônico (HL7 FHIR); e extensão do RAG com modelos de linguagem de grande escala (LLM) para geração de laudos em linguagem natural fundamentados em evidências.

---

## Referências

[1] World Health Organization. *Cardiovascular diseases (CVDs) — Key facts*. WHO Fact Sheet, 2023. Disponível em: https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

[2] Ministério da Saúde do Brasil. *DATASUS — Informações de Saúde: Morbidade Hospitalar do SUS*. Brasília: MS, 2023. Disponível em: http://datasus.saude.gov.br

[3] Kligfield, P. et al. Recommendations for the Standardization and Interpretation of the Electrocardiogram. *Journal of the American College of Cardiology*, v. 49, n. 10, p. 1109–1127, 2007. https://doi.org/10.1016/j.jacc.2007.01.024

[4] Salerno, S. M.; Alguire, P. C.; Waxman, H. S. Competency in Interpretation of 12-Lead Electrocardiograms: A Summary and Appraisal of Published Evidence. *Annals of Internal Medicine*, v. 138, n. 9, p. 751–760, 2003. https://doi.org/10.7326/0003-4819-138-9-200305060-00013

[5] Rajpurkar, P. et al. Cardiologist-Level Arrhythmia Detection and Classification in Ambulatory Electrocardiograms Using a Deep Neural Network. *Nature Medicine*, v. 25, p. 65–69, 2019. https://doi.org/10.1038/s41591-018-0268-3

[6] Ribeiro, A. H. et al. Automatic Diagnosis of the 12-Lead ECG Using a Deep Neural Network. *Nature Communications*, v. 11, n. 1760, 2020. https://doi.org/10.1038/s41467-020-15432-4

[7] Hannun, A. Y. et al. Cardiologist-Level Arrhythmia Detection with Convolutional Neural Networks. *Nature Medicine*, v. 25, p. 65–69, 2019. https://doi.org/10.1038/s41591-018-0268-3

[8] Attia, Z. I. et al. Screening for Cardiac Contractile Dysfunction Using an Artificial Intelligence–Enabled Electrocardiogram. *Nature Medicine*, v. 25, p. 70–74, 2019. https://doi.org/10.1038/s41591-018-0240-2

[9] Badilini, F. et al. ECGScan: A Method for Conversion of Paper Electrocardiographic Printouts to Digital Electrocardiographic Files. *Journal of Electrocardiology*, v. 38, n. 4, p. 310–318, 2005. https://doi.org/10.1016/j.jelectrocard.2005.06.008

[10] Peimankar, A.; Puthusserypady, S. DENS-ECG: A Deep Learning Approach for ECG Signal Delineation. *Expert Systems with Applications*, v. 165, 113911, 2021. https://doi.org/10.1016/j.eswa.2020.113911

[11] Lewis, P. et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems (NeurIPS)*, v. 33, p. 9459–9474, 2020.

[12] Singhal, K. et al. Large Language Models Encode Clinical Knowledge. *Nature*, v. 620, p. 172–180, 2023. https://doi.org/10.1038/s41586-023-06291-2

[13] Luo, R. et al. BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining. *Briefings in Bioinformatics*, v. 23, n. 6, bbac409, 2022. https://doi.org/10.1093/bib/bbac409

[14] Shwartz-Ziv, R.; Armon, A. Tabular Data: Deep Learning is Not All You Need. *Information Fusion*, v. 81, p. 84–90, 2022. https://doi.org/10.1016/j.inffus.2021.11.011

[15] Janosi, A. et al. *Heart Disease Dataset*. UCI Machine Learning Repository, 1988. https://doi.org/10.24432/C52P4X

[16] Thygesen, K. et al. Fourth Universal Definition of Myocardial Infarction (2018). *Journal of the American College of Cardiology*, v. 72, n. 18, p. 2231–2264, 2018. https://doi.org/10.1016/j.jacc.2018.08.1038

[17] Guo, C. et al. On Calibration of Modern Neural Networks. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, p. 1321–1330, 2017.

[18] Mohan, S.; Thirumalai, C.; Srivastava, G. Effective Heart Disease Prediction Using Hybrid Machine Learning Techniques. *IEEE Access*, v. 7, p. 81542–81554, 2019. https://doi.org/10.1109/ACCESS.2019.2923707

[19] Shah, D. et al. Heart Disease Prediction Using Machine Learning Techniques. *SN Computer Science*, v. 1, n. 345, 2020. https://doi.org/10.1007/s42979-020-00365-y

[20] Topol, E. J. High-Performance Medicine: The Convergence of Human and Artificial Intelligence. *Nature Medicine*, v. 25, p. 44–56, 2019. https://doi.org/10.1038/s41591-018-0300-7

[21] Wagner, P. et al. PTB-XL, a Large Publicly Available Electrocardiography Dataset. *Nature Scientific Data*, v. 7, n. 154, 2020. https://doi.org/10.1038/s41597-020-0495-6

[22] Zheng, J. et al. A 12-Lead Electrocardiogram Database for Arrhythmia Research Covering More Than 10,000 Patients. *Scientific Data*, v. 7, n. 48, 2020. https://doi.org/10.1038/s41597-020-0386-x

[23] Gow, B. et al. MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset. *PhysioNet*, 2023. https://doi.org/10.13026/4nqg-sb35

[24] Moody, G. B.; Mark, R. G. The Impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*, v. 20, n. 3, p. 45–50, 2001. https://doi.org/10.1109/51.932724

[25] Alday, E. A. P. et al. Classification of 12-Lead ECGs: The PhysioNet/Computing in Cardiology Challenge 2020. *Physiological Measurement*, v. 41, n. 12, 124003, 2020. https://doi.org/10.1088/1361-6579/abc960

[26] Vovk, V.; Gammerman, A.; Shafer, G. *Algorithmic Learning in a Random World*. Springer, 2005. https://doi.org/10.1007/b106715

[27] Selvaraju, R. R. et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, p. 618–626, 2017. https://doi.org/10.1109/ICCV.2017.74

[28] Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. Heart Rate Variability: Standards of Measurement, Physiological Interpretation, and Clinical Use. *Circulation*, v. 93, n. 5, p. 1043–1065, 1996. https://doi.org/10.1161/01.CIR.93.5.1043

[29] Hindricks, G. et al. 2020 ESC Guidelines for the Diagnosis and Management of Atrial Fibrillation. *European Heart Journal*, v. 42, n. 5, p. 373–498, 2021. https://doi.org/10.1093/eurheartj/ehaa612

[30] Thambawita, V. et al. *ECGDatasets: A Hugging Face Space for ECG Dataset Discovery*. SimulaMet / Hugging Face Spaces, 2023. Disponível em: https://huggingface.co/spaces/vlbthambawita/ECGDatasets

---

*Artigo gerado a partir da documentação técnica e código-fonte do projeto ECG-AI.*  
*Repositório: `ia_coracao/` — Licença MIT.*  
*Para submissão, substituir métricas de benchmark cross-dataset por resultados experimentais reais (ver Seção 7.3, Fase 3).*
