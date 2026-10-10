"""
Data Loader e Utilitários de Pré-Processamento para Dados Cardiológicos.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# Definições padronizadas de features clínicas baseadas nos dados de referência
CATEGORICAL_FEATURES = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
NUMERICAL_FEATURES = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
TARGET_COLUMN = 'HeartDisease'

# Mapeamentos e descrições semânticas para interpretação acadêmica e médica
FEATURE_METADATA = {
    'Age': {'label': 'Idade (anos)', 'type': 'numeric', 'range': (18, 100)},
    'Sex': {'label': 'Sexo', 'type': 'categorical', 'values': {'M': 'Masculino', 'F': 'Feminino'}},
    'ChestPainType': {
        'label': 'Tipo de Dor Torácica',
        'type': 'categorical',
        'values': {
            'TA': 'Angina Típica (Typical Angina)',
            'ATA': 'Angina Atípica (Atypical Angina)',
            'NAP': 'Dor Não Anginosa (Non-Anginal Pain)',
            'ASY': 'Assintomático (Asymptomatic)'
        }
    },
    'RestingBP': {'label': 'Pressão Arterial em Repouso (mmHg)', 'type': 'numeric', 'range': (60, 240)},
    'Cholesterol': {'label': 'Colesterol Sérico (mg/dL)', 'type': 'numeric', 'range': (0, 700)},
    'FastingBS': {
        'label': 'Glicemia em Jejum > 120 mg/dL',
        'type': 'categorical_binary',
        'values': {0: 'Normal (<= 120 mg/dL)', 1: 'Elevada (> 120 mg/dL)'}
    },
    'RestingECG': {
        'label': 'Eletrocardiograma em Repouso',
        'type': 'categorical',
        'values': {
            'Normal': 'Normal',
            'ST': 'Anormalidade da Onda ST-T (Inversão T / Elevação/Depressão ST > 0.05 mV)',
            'LVH': 'Hipertrofia Ventricular Esquerda (Critérios de Estes)'
        }
    },
    'MaxHR': {'label': 'Frequência Cardíaca Máxima Atingida (bpm)', 'type': 'numeric', 'range': (50, 230)},
    'ExerciseAngina': {
        'label': 'Angina Induzida por Exercício',
        'type': 'categorical',
        'values': {'Y': 'Sim', 'N': 'Não'}
    },
    'Oldpeak': {'label': 'Depressão de ST induzida por esforço em relação ao repouso (Oldpeak)', 'type': 'numeric', 'range': (-3.0, 7.0)},
    'ST_Slope': {
        'label': 'Inclinação do Segmento ST no pico do esforço',
        'type': 'categorical',
        'values': {
            'Up': 'Ascendente (Upsloping - menor risco)',
            'Flat': 'Plano (Horizontal - risco intermediário/alto)',
            'Down': 'Descendente (Downsloping - alto risco isquêmico)'
        }
    },
    'HeartDisease': {
        'label': 'Diagnóstico de Doença Cardíaca',
        'type': 'target',
        'values': {0: 'Normal / Baixo Risco', 1: 'Cardiopatia Presente / Alto Risco'}
    }
}


def load_heart_dataset(
    filepath: Optional[Union[str, Path]] = None,
    treated: bool = True
) -> pd.DataFrame:
    """
    Carrega o dataset cardiológico tratando automaticamente delimitadores,
    codificações e tipos de dados.
    
    Args:
        filepath: Caminho do arquivo CSV. Se None, busca em docs/heart_tratado.csv ou docs/heart.csv.
        treated: Se True, prioriza o dataset tratado.
        
    Returns:
        pd.DataFrame carregado e limpo.
    """
    if filepath is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        target_file = "heart_tratado.csv" if treated else "heart.csv"
        filepath = base_dir / "docs" / target_file
        if not filepath.exists():
            fallback_file = "heart.csv" if treated else "heart_tratado.csv"
            filepath = base_dir / "docs" / fallback_file

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado em: {filepath}")

    # Detecção automática de separador (, ou ;)
    try:
        df = pd.read_csv(filepath, sep=';', encoding='utf-8')
        if len(df.columns) <= 1:
            df = pd.read_csv(filepath, sep=',', encoding='utf-8')
    except Exception:
        df = pd.read_csv(filepath, sep=None, engine='python', encoding='utf-8')

    # Limpeza de nomes de colunas
    df.columns = [c.strip() for c in df.columns]

    # Validação e saneamento dos tipos
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce').fillna(0).astype(int)

    return df


def split_heart_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide previsores e alvo em conjuntos de treino e teste estratificados.
    """
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não está presente no DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(
    scaling: str = 'standard',
    encoding: str = 'onehot'
) -> ColumnTransformer:
    """
    Constrói o transformador de colunas Scikit-Learn de acordo com as melhores práticas
    para evitar vazamento de dados (data leakage).
    
    Args:
        scaling: 'standard', 'minmax', 'robust' ou 'none'
        encoding: 'onehot' ou 'ordinal'
    """
    # Pipeline numérico
    num_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scaling == 'standard':
        num_steps.append(('scaler', StandardScaler()))
    elif scaling == 'minmax':
        num_steps.append(('scaler', MinMaxScaler()))
    elif scaling == 'robust':
        num_steps.append(('scaler', RobustScaler()))
        
    num_pipeline = Pipeline(num_steps)

    # Pipeline categórico
    cat_steps = [('imputer', SimpleImputer(strategy='most_frequent'))]
    if encoding == 'onehot':
        cat_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
    elif encoding == 'ordinal':
        cat_steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
    else:
        cat_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))

    cat_pipeline = Pipeline(cat_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor
