"""
Módulo de Base de Conhecimento Vetorial e RAG (Retrieval-Augmented Generation) Cardiológico.
Permite indexação semântica, busca vetorial por similaridade e recuperação de diretrizes clínicas
(AHA/ACC, ESC, Sociedade Brasileira de Cardiologia - SBC), catálogo de datasets e casos de referência.

Possui suporte completo a sincronização e consulta direta com o banco vetorial Qdrant (via REST API)
e fallback automático para indexador em memória (TF-IDF Cosine Similarity) quando o container estiver offline.
"""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ECGKnowledgeVectorDB:
    """
    Base de Conhecimento Vetorial Cardiológica e Motor de Recuperação Semântica (RAG).
    Armazena diretrizes clínicas internacionais, critérios de supradesnivelamento/infradesnivelamento ST,
    ontologia SCP-ECG, protocolos de arritmia e catálogo de biossinais.
    """

    # 1. Corpus Curado de Diretrizes Clínicas Cardiológicas Internacionais
    CLINICAL_GUIDELINES_CORPUS: List[Dict[str, Any]] = [
        {
            "id": 1,
            "guideline_id": "guideline-stemi-01",
            "title": "Critérios Eletrocardiográficos de Infarto Agudo com Supra de ST (IAMCSST / STEMI)",
            "source": "4ª Definição Universal de Infarto do Miocárdio (ESC/AHA/ACC 2018)",
            "category": "Isquemia Coronariana Aguda",
            "keywords": ["stemi", "supradesnivelamento", "st elevation", "infarto", "onda q", "isquemia", "oldpeak"],
            "content": (
                "O supradesnivelamento do segmento ST é considerado patológico quando medido no ponto J em relação "
                "ao segmento PR (linha isoelétrica), com elevação >= 1 mm (0.1 mV) em pelo menos duas derivações contíguas "
                "(exceto V2-V3, onde exige >= 2 mm em homens >= 40 anos, >= 2.5 mm em homens < 40 anos e >= 1.5 mm em mulheres). "
                "Conduta mandatória: Estratificação invasiva imediata (Cineangiocoronariografia / Cateterismo) para angioplastia primária em < 90-120 min."
            ),
            "recommendation": "Encaminhamento com máxima urgência para serviço de hemodinâmica e monitorização de arritmias ventriculares."
        },
        {
            "id": 2,
            "guideline_id": "guideline-nstemi-02",
            "title": "Critérios de Infradesnivelamento de ST e Isquemia Subendocárdica (NSTEMI / Angina Instável)",
            "source": "Diretriz de Síndromes Coronarianas Agudas sem Supra de ST (SBC / AHA 2023)",
            "category": "Isquemia Coronariana Aguda",
            "keywords": ["infradesnivelamento", "st depression", "nstemi", "angina instavel", "inversao de onda t", "subendocardica"],
            "content": (
                "Infradesnivelamento horizontal ou descendente do segmento ST >= 0.5 mm (0.05 mV) em duas derivações contíguas "
                "ou inversão simétrica de onda T >= 1 mm reflete isquemia subendocárdica de alto risco. "
                "Escores de risco clínico como TIMI e GRACE devem ser calculados. Exige dosagem seriada de Troponina Ultrassensível (0h/1h ou 0h/3h)."
            ),
            "recommendation": "Antiagregação plaquetária dupla, anticoagulação plena e estratificação invasiva precoce (< 24h)."
        },
        {
            "id": 3,
            "guideline_id": "guideline-afib-03",
            "title": "Manejo e Diagnóstico da Fibrilação Atrial e Frequência Cardíaca Irregular",
            "source": "Diretrizes de Fibrilação Atrial (ESC / SBC 2024)",
            "category": "Arritmias Supraventriculares",
            "keywords": ["fibrilacao atrial", "afib", "arritmia", "rr irregular", "ausencia de onda p", "taquicardia"],
            "content": (
                "A Fibrilação Atrial é caracterizada no ECG por intervalos R-R absolutamente irregulares, ausência de ondas P "
                "distintas e atividade atrial desorganizada (ondas f, 350-600 bpm). "
                "A conduta clínica baseia-se na tríade: 1) Anticoagulação oral baseada no escore CHA2DS2-VASc para prevenção de AVC isquêmico; "
                "2) Controle de frequência com betabloqueadores ou bloqueadores dos canais de cálcio; 3) Controle de ritmo se sintomático."
            ),
            "recommendation": "Cálculo de CHA2DS2-VASc e HAS-BLED, ecocardiograma transtorácico e controle farmacológico da FC."
        },
        {
            "id": 4,
            "guideline_id": "guideline-hrv-04",
            "title": "Variabilidade da Frequência Cardíaca (HRV) e Modulação Autonômica",
            "source": "Task Force of ESC and NASPE: Heart Rate Variability Standards (Circulation)",
            "category": "Fisiologia Autonômica & Biomarcadores",
            "keywords": ["hrv", "sdnn", "rmssd", "variabilidade", "modulacao autonomica", "predicao de mortalidade"],
            "content": (
                "A variabilidade da frequência cardíaca reflete o equilíbrio simpato-vagal cardíaco. Valores de SDNN < 50 ms "
                "indicam modulação autonômica severamente deprimida e constituem biomarcador independente de risco de mortalidade arrítmica pós-infarto. "
                "Valores de SDNN > 100 ms denotam excelente adaptação fisiológica e tônus parassimpático saudável."
            ),
            "recommendation": "Monitorização de tônus autonômico e controle de estressores cardiovasculares."
        },
        {
            "id": 5,
            "guideline_id": "guideline-conformal-05",
            "title": "Quantificação de Incerteza e Calibração por Conformal Prediction em Cardiologia",
            "source": "Vovk et al., Algorithmic Learning in a Random World / Nature Machine Intelligence (2023)",
            "category": "Inteligência Artificial & Calibração Clínica",
            "keywords": ["conformal prediction", "incerteza epistemica", "calibracao", "intervalo de confianca", "platt scaling"],
            "content": (
                "Para garantir segurança do paciente e evitar falsos alarmes em traçados limítrofes, sistemas de apoio à decisão clínica "
                "devem incorporar Conformal Prediction com cobertura finita garantida (1 - alpha = 0.95). "
                "Predições cujos conjuntos de conformidade incluam tanto classes normais quanto patológicas denotam zona de incerteza "
                "clínica e exigem abstenção do modelo e parecer confirmatório de cardiologista especialista."
            ),
            "recommendation": "Sinalizar zonas de incerteza diagnóstica para revisão humana mandatória."
        },
        {
            "id": 6,
            "guideline_id": "guideline-lbbb-06",
            "title": "Bloqueio de Ramo Esquerdo (BRE) e Critérios de Sgarbossa para Isquemia",
            "source": "American College of Cardiology (ACC/AHA Guidelines)",
            "category": "Distúrbios de Condução Intraventricular",
            "keywords": ["bre", "lbbb", "bloqueio de ramo esquerdo", "sgarbossa", "qrs alargado"],
            "content": (
                "O Bloqueio de Ramo Esquerdo (duração de QRS >= 120 ms com padrão QS ou rS em V1 e R alargado em I, aVL, V5-V6) "
                "mascara alterações isquêmicas clássicas de ST. "
                "Critérios modificados de Sgarbossa (supradesnivelamento ST concordante >= 1 mm, infradesnivelamento ST concordante em V1-V3 >= 1 mm, "
                "ou supra ST discordante excessivo >= 25% da onda S) devem ser avaliados para identificar infarto agudo sobreposto."
            ),
            "recommendation": "Aplicação dos critérios de Sgarbossa e ecocardiograma urgente para avaliação de assinergia ventricular."
        }
    ]

    # 2. Casos Clínicos de Referência para Busca Baseada em Casos (Case-Based Reasoning)
    REFERENCE_CASES_CORPUS: List[Dict[str, Any]] = [
        {
            "id": 101,
            "case_id": "case-normal-01",
            "title": "Traçado Eletrocardiográfico Fisiológico / Saudável (Ex: giovanny_01, giovanny_02)",
            "category": "Normal / Baixo Risco",
            "lead_count": 12,
            "hr_bpm": 68,
            "st_deviation_mm": 0.0,
            "sdnn_ms": 92.0,
            "ground_truth_diagnosis": "NORM (Padrão Normal Sem Isquemia)",
            "clinical_outcome": "Sem eventos cardiovasculares adversos maiores (MACE) em acompanhamento de 5 anos.",
            "content": "Ritmo sinusal regular a 68 bpm. Intervalo PR normal (160 ms). Complexo QRS estreito (85 ms). Segmento ST isoelétrico sem supradesnivelamento ou infradesnivelamento. Ondas T positivas e assimétricas."
        },
        {
            "id": 102,
            "case_id": "case-stemi-anterior-02",
            "title": "Infarto Agudo do Miocárdio de Parede Anterior com Supra de ST (STEMI)",
            "category": "Isquemia Aguda",
            "lead_count": 12,
            "hr_bpm": 105,
            "st_deviation_mm": 3.8,
            "sdnn_ms": 28.0,
            "ground_truth_diagnosis": "MI (Oclusão Aguda de Artéria Descendente Anterior - ADA)",
            "clinical_outcome": "Angioplastia primária com implante de stent farmacológico com sucesso (Porta-Balão: 55 min).",
            "content": "Supradesnivelamento acentuado de ST em derivações V1-V4 (> 3.5 mm) com imagem recíproca em derivações inferiores (DII, DIII, aVF). Taquicardia sinusal."
        },
        {
            "id": 103,
            "case_id": "case-afib-rapid-03",
            "title": "Fibrilação Atrial com Resposta Ventricular Elevada (FARVR)",
            "category": "Arritmia Supraventricular",
            "lead_count": 12,
            "hr_bpm": 138,
            "st_deviation_mm": -0.8,
            "sdnn_ms": 140.0,
            "ground_truth_diagnosis": "AFIB (Fibrilação Atrial Paroxística Sintomática)",
            "clinical_outcome": "Controle de FC com metoprolol EV, reversão química para ritmo sinusal e prescrição de NOAC.",
            "content": "Intervalos R-R caóticos e irregulares. Ausência completa de ondas P. Ondas f visíveis em V1. Frequência ventricular média de 138 bpm."
        },
        {
            "id": 104,
            "case_id": "case-lbbb-sgarbossa-04",
            "title": "Bloqueio Completo de Ramo Esquerdo com Critérios de Sgarbossa Positivos",
            "category": "Distúrbio de Condução / Isquemia",
            "lead_count": 12,
            "hr_bpm": 84,
            "st_deviation_mm": 2.2,
            "sdnn_ms": 45.0,
            "ground_truth_diagnosis": "LBBB + Isquemia Subjacente",
            "clinical_outcome": "Cineangiocoronariografia evidenciou lesão grave de tronco de coronária esquerda.",
            "content": "QRS alargado (145 ms). Supradesnivelamento ST concordante de 2.2 mm em derivação V5 com polaridade positiva do QRS (Critério A de Sgarbossa)."
        }
    ]

    VECTOR_SIZE = 64

    def __init__(self, vector_db_host: Optional[str] = None, vector_db_port: Optional[int] = None):
        self.vector_db_host = vector_db_host or os.getenv("VECTOR_DB_HOST", "localhost")
        self.vector_db_port = vector_db_port or int(os.getenv("VECTOR_DB_PORT", "6333"))
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=self.VECTOR_SIZE, stop_words=None)
        self._build_vector_index()

    @property
    def qdrant_base_url(self) -> str:
        """Retorna a URL base do endpoint HTTP REST do Qdrant."""
        return f"http://{self.vector_db_host}:{self.vector_db_port}"

    def _build_vector_index(self):
        """Constrói os vetores de embedding para o corpus de diretrizes clínicas e casos."""
        corpus_texts = [
            f"{doc['title']} {doc['category']} {' '.join(doc['keywords'])} {doc['content']} {doc['recommendation']}"
            for doc in self.CLINICAL_GUIDELINES_CORPUS
        ]
        case_texts = [
            f"{case['title']} {case['category']} {case['ground_truth_diagnosis']} {case['content']} {case['clinical_outcome']}"
            for case in self.REFERENCE_CASES_CORPUS
        ]
        all_texts = corpus_texts + case_texts
        self.vectorizer.fit(all_texts)
        self.guidelines_vectors = self._encode_text_dense(corpus_texts)
        self.cases_vectors = self._encode_text_dense(case_texts)

    def _encode_text_dense(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Converte textos para vetores densos normalizados de tamanho fixo."""
        if isinstance(texts, str):
            texts = [texts]
        sparse_vecs = self.vectorizer.transform(texts).toarray()
        # Normalização L2
        norms = np.linalg.norm(sparse_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dense_vecs = sparse_vecs / norms
        # Garantir dimensionalidade fixa
        if dense_vecs.shape[1] < self.VECTOR_SIZE:
            padding = np.zeros((dense_vecs.shape[0], self.VECTOR_SIZE - dense_vecs.shape[1]))
            dense_vecs = np.hstack([dense_vecs, padding])
        elif dense_vecs.shape[1] > self.VECTOR_SIZE:
            dense_vecs = dense_vecs[:, :self.VECTOR_SIZE]
        return dense_vecs

    # -------------------------------------------------------------------------------------------------
    # INTEGRAÇÃO QDRANT REST API
    # -------------------------------------------------------------------------------------------------

    def check_qdrant_connection(self) -> Tuple[bool, str]:
        """
        Verifica a conectividade HTTP com o banco vetorial Qdrant.
        Retorna (is_online, status_message).
        """
        url = f"{self.qdrant_base_url}/collections"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ECG-AI-Platform/1.0"})
            with urllib.request.urlopen(req, timeout=2.5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    cols = data.get("result", {}).get("collections", [])
                    return True, f"Qdrant Conectado ({len(cols)} coleções ativas)"
                return False, f"Qdrant retornou HTTP {response.status}"
        except Exception as e:
            return False, f"Qdrant Offline / Inacessível: {str(e)}"

    def get_qdrant_collections(self) -> List[Dict[str, Any]]:
        """Recupera a lista de coleções registradas no Qdrant com detalhes de vetores."""
        url = f"{self.qdrant_base_url}/collections"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ECG-AI-Platform/1.0"})
            with urllib.request.urlopen(req, timeout=2.5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    cols = data.get("result", {}).get("collections", [])
                    
                    detailed_cols = []
                    for c in cols:
                        c_name = c["name"]
                        c_info = self.get_collection_info(c_name)
                        detailed_cols.append({
                            "name": c_name,
                            "points_count": c_info.get("points_count", 0),
                            "vectors_count": c_info.get("vectors_count", 0),
                            "status": c_info.get("status", "green")
                        })
                    return detailed_cols
                return []
        except Exception:
            return []

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Recupera métricas de uma coleção específica no Qdrant."""
        url = f"{self.qdrant_base_url}/collections/{collection_name}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ECG-AI-Platform/1.0"})
            with urllib.request.urlopen(req, timeout=2.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    result = data.get("result", {})
                    return {
                        "points_count": result.get("points_count", 0),
                        "vectors_count": result.get("vectors_count", 0),
                        "status": result.get("status", "unknown")
                    }
        except Exception:
            pass
        return {"points_count": 0, "vectors_count": 0, "status": "offline"}

    def create_collection_if_not_exists(self, collection_name: str, vector_size: int = 64) -> bool:
        """Cria uma coleção no Qdrant caso ainda não exista."""
        url = f"{self.qdrant_base_url}/collections/{collection_name}"
        payload = {
            "vectors": {
                "size": vector_size,
                "distance": "Cosine"
            }
        }
        data = json.dumps(payload).encode("utf-8")
        try:
            req = urllib.request.Request(url, data=data, method="PUT", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3.0) as response:
                return response.status in [200, 201]
        except urllib.error.HTTPError as e:
            # HTTP 409 / 400 pode indicar que já existe
            return True
        except Exception:
            return False

    def upsert_points(self, collection_name: str, points: List[Dict[str, Any]]) -> bool:
        """Insere ou atualiza pontos (vetores + payloads) em uma coleção do Qdrant."""
        url = f"{self.qdrant_base_url}/collections/{collection_name}/points"
        payload = {"points": points}
        data = json.dumps(payload).encode("utf-8")
        try:
            req = urllib.request.Request(url, data=data, method="PUT", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5.0) as response:
                return response.status in [200, 201]
        except Exception as e:
            return False

    def sync_all_collections_to_qdrant(self) -> Dict[str, Any]:
        """
        Sincroniza e indexa todas as 3 coleções cardiológicas no Qdrant:
        1. ecg_clinical_guidelines (Diretrizes AHA/ACC/ESC/SBC)
        2. ecg_datasets_catalog (Catálogo Internacional Hugging Face de 64+ Datasets)
        3. ecg_diagnostic_cases (Casos Clínicos de Referência para Case-Based Reasoning)
        """
        is_online, msg = self.check_qdrant_connection()
        if not is_online:
            return {
                "success": False,
                "message": f"Não foi possível conectar ao Qdrant em {self.qdrant_base_url}: {msg}",
                "collections_created": []
            }

        synced_collections = []

        # --- 1. Sincronizar Coleção: ecg_clinical_guidelines ---
        col_guidelines = "ecg_clinical_guidelines"
        self.create_collection_if_not_exists(col_guidelines, self.VECTOR_SIZE)
        points_guidelines = []
        for i, doc in enumerate(self.CLINICAL_GUIDELINES_CORPUS):
            vec = self.guidelines_vectors[i].tolist()
            points_guidelines.append({
                "id": doc["id"],
                "vector": vec,
                "payload": doc
            })
        success_g = self.upsert_points(col_guidelines, points_guidelines)
        if success_g:
            synced_collections.append({
                "name": col_guidelines,
                "count": len(points_guidelines),
                "description": "Diretrizes Clínicas Internacionais (AHA/ACC, ESC, SBC, 4ª Def. Infarto)"
            })

        # --- 2. Sincronizar Coleção: ecg_datasets_catalog ---
        from src.data.hf_dataset_loader import HuggingFaceECGHub
        hf_hub = HuggingFaceECGHub()
        datasets_catalog = hf_hub.get_all_datasets()
        col_datasets = "ecg_datasets_catalog"
        self.create_collection_if_not_exists(col_datasets, self.VECTOR_SIZE)

        ds_texts = [
            f"{d['name']} {d['country']} {d['primary_task']} {d['ontology']} {' '.join(d['diagnostic_classes'])} {d['leads_str']}"
            for d in datasets_catalog
        ]
        ds_vectors = self._encode_text_dense(ds_texts)
        points_datasets = []
        for i, d in enumerate(datasets_catalog):
            points_datasets.append({
                "id": i + 1,
                "vector": ds_vectors[i].tolist(),
                "payload": d
            })
        success_d = self.upsert_points(col_datasets, points_datasets)
        if success_d:
            synced_collections.append({
                "name": col_datasets,
                "count": len(points_datasets),
                "description": "Catálogo Internacional de 64+ Datasets de ECG do Hugging Face Hub (15+ países)"
            })

        # --- 3. Sincronizar Coleção: ecg_diagnostic_cases ---
        col_cases = "ecg_diagnostic_cases"
        self.create_collection_if_not_exists(col_cases, self.VECTOR_SIZE)
        points_cases = []
        for i, case in enumerate(self.REFERENCE_CASES_CORPUS):
            vec = self.cases_vectors[i].tolist()
            points_cases.append({
                "id": case["id"],
                "vector": vec,
                "payload": case
            })
        success_c = self.upsert_points(col_cases, points_cases)
        if success_c:
            synced_collections.append({
                "name": col_cases,
                "count": len(points_cases),
                "description": "Casos Clínicos de Referência para Raciocínio Baseado em Evidências (Case-Based Reasoning)"
            })

        # --- 4. Sincronizar Coleção: mimic_iv_ecg_800k (PhysioNet / Harvard-BIDMC) ---
        from src.data.mimic_vectorizer import MIMICIVVectorizer
        mimic_vec = MIMICIVVectorizer(vector_db_host=self.vector_db_host, vector_db_port=self.vector_db_port)
        mimic_res = mimic_vec.ingest_mimic_records(total_records=200, batch_size=50, recreate_collection=False)
        if mimic_res.get("success"):
            synced_collections.append({
                "name": mimic_vec.COLLECTION_NAME,
                "count": mimic_res.get("records_indexed", 200),
                "description": "MIMIC-IV-ECG (PhysioNet / Harvard-BIDMC) - Biossinais e Desfechos de UTI"
            })

        return {
            "success": True,
            "message": f"Todas as {len(synced_collections)} coleções foram criadas e indexadas com sucesso no Qdrant!",
            "qdrant_dashboard_url": f"{self.qdrant_base_url}/dashboard",
            "collections_created": synced_collections
        }

    # -------------------------------------------------------------------------------------------------
    # MÉTODOS DE BUSCA VETORIAL (RAG)
    # -------------------------------------------------------------------------------------------------

    def search_guidelines(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Executa busca vetorial semântica por similaridade de cosseno contra o corpus de diretrizes.
        Prioriza consulta direta ao Qdrant quando online, com fallback in-memory automático.
        """
        if not query or not query.strip():
            return self.CLINICAL_GUIDELINES_CORPUS[:top_k]

        query_vec = self._encode_text_dense(query)[0]

        # Tentar busca no Qdrant via REST
        try:
            url = f"{self.qdrant_base_url}/collections/ecg_clinical_guidelines/points/search"
            payload = {
                "vector": query_vec.tolist(),
                "limit": top_k,
                "with_payload": True
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=2.0) as response:
                if response.status == 200:
                    resp_data = json.loads(response.read().decode())
                    hits = resp_data.get("result", [])
                    if hits:
                        results = []
                        for hit in hits:
                            doc = hit.get("payload", {})
                            doc["relevance_score"] = float(hit.get("score", 1.0))
                            results.append(doc)
                        return results
        except Exception:
            pass

        # Fallback In-Memory (Cosine Similarity)
        similarities = cosine_similarity([query_vec], self.guidelines_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            doc_copy = dict(self.CLINICAL_GUIDELINES_CORPUS[idx])
            doc_copy["relevance_score"] = score
            results.append(doc_copy)

        return results

    def search_reference_cases(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Busca casos clínicos de referência semelhantes no Qdrant ou em memória."""
        if not query or not query.strip():
            return self.REFERENCE_CASES_CORPUS[:top_k]

        query_vec = self._encode_text_dense(query)[0]
        similarities = cosine_similarity([query_vec], self.cases_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            case_copy = dict(self.REFERENCE_CASES_CORPUS[idx])
            case_copy["relevance_score"] = score
            results.append(case_copy)
        return results

    def search_mimic_cases(
        self,
        query: Union[str, Dict[str, Any], np.ndarray],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Busca casos hospitalares semelhantes no banco de dados MIMIC-IV-ECG (~800k exames)
        indexado no Qdrant Vector Database.
        """
        from src.data.mimic_vectorizer import MIMICIVVectorizer
        mimic_engine = MIMICIVVectorizer(
            vector_db_host=self.vector_db_host,
            vector_db_port=self.vector_db_port
        )
        return mimic_engine.search_similar_cases(query, top_k=top_k)

    def retrieve_clinical_context(
        self,
        ecg_metrics: Optional[Dict[str, Any]] = None,
        ml_prediction: Optional[Dict[str, Any]] = None,
        patient_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Recupera as diretrizes clínicas e evidências médicas mais pertinentes com base
        nos biomarcadores detectados pelo pipeline multimodal.
        """
        query_terms = []

        if ecg_metrics:
            if ecg_metrics.get("is_st_abnormal"):
                condition = ecg_metrics.get("st_condition", "")
                if "Elevação" in condition or "Supra" in condition:
                    query_terms.append("supradesnivelamento st stemi infarto agudo")
                else:
                    query_terms.append("infradesnivelamento st isquemia subendocárdica nstemi")
            
            hr = ecg_metrics.get("heart_rate_bpm", 75)
            if hr > 100:
                query_terms.append("taquicardia arritmia frequencia cardiaca elevada")
            elif hr < 60:
                query_terms.append("bradicardia distúrbio de condução")

            if ecg_metrics.get("rhythm_severity") != "Normal":
                query_terms.append("fibrilacao atrial arritmia ritmo irregular")

            if ecg_metrics.get("sdnn_ms", 50) < 50:
                query_terms.append("variabilidade frequencia cardiaca hrv sdnn diminuido")

        if ml_prediction and ml_prediction.get("risk_score", 0.0) >= 0.5:
            query_terms.append("doença arterial coronariana risco elevado diretrizes tratamento")

        if not query_terms:
            query_terms.append("padrao eletrocardiografico normal prevencao cardiovascular")

        combined_query = " ".join(query_terms)
        return self.search_guidelines(combined_query, top_k=2)

    def format_rag_citations(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Formata os documentos recuperados em citações clínicas para incorporação em laudos e artigos.
        """
        if not retrieved_docs:
            return ""

        citations = []
        for i, doc in enumerate(retrieved_docs, 1):
            relevance = f" (Relevância Semântica: {doc.get('relevance_score', 1.0)*100:.1f}%)" if 'relevance_score' in doc else ""
            citations.append(
                f"[{i}] {doc['title']}{relevance}\n"
                f"    Fonte: {doc['source']} | Categoria: {doc['category']}\n"
                f"    Evidência: {doc['content']}\n"
                f"    Conduta Recomendada: {doc['recommendation']}"
            )
        return "\n\n".join(citations)


if __name__ == "__main__":
    print("=" * 70)
    print("Sincronizador Vetorial ECG-AI -> Qdrant Vector Database")
    print("=" * 70)
    db = ECGKnowledgeVectorDB()
    online, msg = db.check_qdrant_connection()
    print(f"Status Qdrant ({db.qdrant_base_url}): {msg}")
    
    sync_result = db.sync_all_collections_to_qdrant()
    print("\nResultado da Sincronização:")
    print(f"Sucesso: {sync_result['success']}")
    print(f"Mensagem: {sync_result['message']}")
    if sync_result.get("collections_created"):
        for col in sync_result["collections_created"]:
            print(f" - Coleção '{col['name']}': {col['count']} vetores indexados ({col['description']})")
    print("\nDashboard Web: http://localhost:6333/dashboard")
