"""
Testes unitários para o carregamento e pré-processamento de dados.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data.loader import (
    load_heart_dataset,
    split_heart_data,
    build_preprocessor,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN
)


def test_load_heart_dataset_exists():
    df = load_heart_dataset(treated=True)
    assert df is not None
    assert len(df) > 100
    assert TARGET_COLUMN in df.columns
    for col in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
        assert col in df.columns


def test_split_heart_data():
    df = load_heart_dataset(treated=True)
    X_train, X_test, y_train, y_test = split_heart_data(df, test_size=0.2, random_state=42)
    
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
    assert TARGET_COLUMN not in X_train.columns
    assert len(y_train.unique()) == 2


def test_build_preprocessor():
    preprocessor = build_preprocessor(scaling='standard', encoding='onehot')
    assert preprocessor is not None
    
    df = load_heart_dataset(treated=True)
    X = df.drop(columns=[TARGET_COLUMN])
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] >= len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)
