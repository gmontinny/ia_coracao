"""
Testes unitários para os classificadores tabulares e ensemble de IA.
"""

import pytest
import numpy as np
import pandas as pd
from src.data.loader import load_heart_dataset, split_heart_data, TARGET_COLUMN
from src.models.tabular_models import TabularECGClassifier, train_benchmark_suite


def test_tabular_classifier_fit_predict():
    df = load_heart_dataset(treated=True)
    X_train, X_test, y_train, y_test = split_heart_data(df, test_size=0.25, random_state=42)

    for model_name in ['xgboost', 'random_forest', 'logistic_regression']:
        clf = TabularECGClassifier(model_type=model_name, random_state=42)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        probas = clf.predict_proba(X_test)

        assert len(preds) == len(X_test)
        assert probas.shape == (len(X_test), 2)
        assert np.all((preds == 0) | (preds == 1))

        eval_res = clf.evaluate(X_test, y_test)
        assert 'accuracy' in eval_res
        assert 'roc_auc' in eval_res
        assert eval_res['accuracy'] > 0.70


def test_single_patient_prediction():
    df = load_heart_dataset(treated=True)
    X_train, _, y_train, _ = split_heart_data(df, random_state=42)

    clf = TabularECGClassifier(model_type='xgboost', random_state=42)
    clf.fit(X_train, y_train)

    sample_patient = {
        'Age': 54,
        'Sex': 'M',
        'ChestPainType': 'ASY',
        'RestingBP': 140,
        'Cholesterol': 260,
        'FastingBS': 0,
        'RestingECG': 'Normal',
        'MaxHR': 130,
        'ExerciseAngina': 'Y',
        'Oldpeak': 1.5,
        'ST_Slope': 'Flat'
    }

    res = clf.predict_patient(sample_patient)
    assert 'prediction' in res
    assert 'risk_score' in res
    assert 0.0 <= res['risk_score'] <= 1.0
