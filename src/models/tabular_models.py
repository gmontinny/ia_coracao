"""
Modelos de Machine Learning e Redes Neurais Tabulares para Diagnóstico Cardiológico.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from ..data.loader import build_preprocessor, load_heart_dataset, split_heart_data, TARGET_COLUMN


class TabularECGClassifier:
    """
    Classificador Ensemble e Suite de Modelos Tabulares para Risco Cardiológico.
    Implementa modelos do estado-da-arte e métricas de acurácia clínica.
    """

    SUPPORTED_MODELS = [
        'xgboost',
        'lightgbm',
        'random_forest',
        'neural_network_mlp',
        'svm',
        'logistic_regression',
        'naive_bayes',
        'ensemble_voting'
    ]

    def __init__(
        self,
        model_type: str = 'xgboost',
        scaling: str = 'standard',
        encoding: str = 'onehot',
        random_state: int = 42,
        **model_params
    ):
        self.model_type = model_type.lower()
        self.scaling = scaling
        self.encoding = encoding
        self.random_state = random_state
        self.model_params = model_params
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted: bool = False
        self.feature_names_out_: Optional[List[str]] = None
        self._init_pipeline()

    def _get_base_estimator(self):
        """Retorna a instância do estimador configurado."""
        if self.model_type == 'xgboost':
            params = {
                'max_depth': 3,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'random_state': self.random_state,
                'eval_metric': 'logloss'
            }
            params.update(self.model_params)
            return XGBClassifier(**params)

        elif self.model_type == 'lightgbm':
            params = {
                'num_leaves': 31,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'random_state': self.random_state,
                'verbose': -1
            }
            params.update(self.model_params)
            return LGBMClassifier(**params)

        elif self.model_type == 'random_forest':
            params = {
                'n_estimators': 250,
                'max_depth': 8,
                'min_samples_split': 4,
                'min_samples_leaf': 2,
                'random_state': self.random_state
            }
            params.update(self.model_params)
            return RandomForestClassifier(**params)

        elif self.model_type == 'neural_network_mlp':
            params = {
                'hidden_layer_sizes': (64, 32, 16),
                'activation': 'relu',
                'solver': 'adam',
                'max_iter': 500,
                'early_stopping': True,
                'random_state': self.random_state
            }
            params.update(self.model_params)
            return MLPClassifier(**params)

        elif self.model_type == 'svm':
            params = {
                'C': 1.0,
                'kernel': 'rbf',
                'probability': True,
                'random_state': self.random_state
            }
            params.update(self.model_params)
            return SVC(**params)

        elif self.model_type == 'logistic_regression':
            params = {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': self.random_state
            }
            params.update(self.model_params)
            return LogisticRegression(**params)

        elif self.model_type == 'naive_bayes':
            params = {}
            params.update(self.model_params)
            return GaussianNB(**params)

        elif self.model_type == 'ensemble_voting':
            clf_xgb = XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=200, eval_metric='logloss', random_state=self.random_state)
            clf_rf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=self.random_state)
            clf_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=self.random_state)
            clf_lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
            
            return VotingClassifier(
                estimators=[
                    ('xgb', clf_xgb),
                    ('rf', clf_rf),
                    ('mlp', clf_mlp),
                    ('lr', clf_lr)
                ],
                voting='soft'
            )
        else:
            raise ValueError(f"Modelo não suportado: {self.model_type}. Opções: {self.SUPPORTED_MODELS}")

    def _init_pipeline(self):
        """Inicializa o pipeline completo com pré-processador e estimador."""
        preprocessor = build_preprocessor(scaling=self.scaling, encoding=self.encoding)
        estimator = self._get_base_estimator()
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', estimator)
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TabularECGClassifier':
        """Treina o pipeline nos dados informados."""
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Gera previsões binárias (0 = Normal, 1 = Cardiopatia)."""
        if not self.is_fitted:
            raise RuntimeError("O modelo precisa ser treinado antes de executar previsões.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Gera probabilidades calibradas [P(Normal), P(Cardiopatia)]."""
        if not self.is_fitted:
            raise RuntimeError("O modelo precisa ser treinado antes de executar previsões.")
        return self.pipeline.predict_proba(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Gera métricas clínicas e acadêmicas detalhadas para avaliação do modelo.
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall (minimização de falso negativo)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Especificidade
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        brier = brier_score_loss(y_test, y_proba)

        return {
            'model': self.model_type,
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'sensitivity_recall': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'brier_score': brier,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_raw': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

    def cross_validate_academic(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Executa validação cruzada k-fold estratificada computando métricas completas com desvio padrão.
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        scoring = {
            'accuracy': 'accuracy',
            'balanced_accuracy': 'balanced_accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }

        scores = cross_validate(self.pipeline, X, y, cv=skf, scoring=scoring, return_train_score=True)

        results = {}
        for metric in scoring.keys():
            test_metric = f"test_{metric}"
            train_metric = f"train_{metric}"
            results[f"{metric}_mean"] = float(np.mean(scores[test_metric]))
            results[f"{metric}_std"] = float(np.std(scores[test_metric]))
            results[f"{metric}_train_mean"] = float(np.mean(scores[train_metric]))
            
        results['model'] = self.model_type
        results['cv_folds'] = cv
        return results

    def get_feature_importances(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula e formata a importância das features para interpretabilidade clínica.
        """
        if not self.is_fitted:
            raise RuntimeError("O modelo precisa ser treinado para calcular importância das features.")

        clf = self.pipeline.named_steps['classifier']
        preprocessor = self.pipeline.named_steps['preprocessor']

        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feat_{i}" for i in range(15)]

        importances = None
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            importances = np.abs(clf.coef_[0])

        if importances is not None and len(importances) == len(feature_names):
            # Normalizar para percentual
            total = np.sum(importances)
            norm_imp = (importances / total) if total > 0 else importances
            feature_imp = {str(k): float(v) for k, v in zip(feature_names, norm_imp)}
            return dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True))

        return {}

    def predict_patient(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza a predição para um único paciente e retorna o laudo estruturado de probabilidades.
        """
        df_patient = pd.DataFrame([patient_dict])
        pred = int(self.predict(df_patient)[0])
        probas = self.predict_proba(df_patient)[0]

        risk_score = float(probas[1])
        if risk_score < 0.30:
            risk_category = "Baixo Risco"
            color = "green"
        elif risk_score < 0.65:
            risk_category = "Risco Moderado"
            color = "orange"
        else:
            risk_category = "Alto Risco / Cardiopatia Provável"
            color = "red"

        return {
            'prediction': pred,
            'prediction_label': 'Cardiopatia Presente' if pred == 1 else 'Normal / Ausência de Cardiopatia',
            'risk_score': risk_score,
            'probability_normal': float(probas[0]),
            'probability_disease': float(probas[1]),
            'risk_category': risk_category,
            'risk_color': color
        }

    def save(self, filepath: Union[str, Path]):
        """Salva o modelo serializado."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'TabularECGClassifier':
        """Carrega o modelo serializado."""
        return joblib.load(filepath)


def train_benchmark_suite(
    df: Optional[pd.DataFrame] = None,
    cv: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, TabularECGClassifier]]:
    """
    Treina e avalia todos os modelos suportados para gerar a tabela de benchmark acadêmico.
    """
    if df is None:
        df = load_heart_dataset(treated=True)

    X_train, X_test, y_train, y_test = split_heart_data(df, random_state=random_state)

    results = []
    models_dict = {}

    for model_name in TabularECGClassifier.SUPPORTED_MODELS:
        model = TabularECGClassifier(model_type=model_name, random_state=random_state)
        model.fit(X_train, y_train)
        eval_metrics = model.evaluate(X_test, y_test)
        cv_metrics = model.cross_validate_academic(df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN], cv=cv)

        row = {
            'Modelo': model_name.replace('_', ' ').title(),
            'Acurácia (Teste)': eval_metrics['accuracy'],
            'Acurácia CV (Média ± DP)': f"{cv_metrics['accuracy_mean']:.3f} ± {cv_metrics['accuracy_std']:.3f}",
            'Sensibilidade / Recall': eval_metrics['sensitivity_recall'],
            'Especificidade': eval_metrics['specificity'],
            'Precisão': eval_metrics['precision'],
            'F1-Score': eval_metrics['f1_score'],
            'ROC-AUC': eval_metrics['roc_auc'],
            'PR-AUC': eval_metrics['pr_auc'],
            'Brier Score': eval_metrics['brier_score']
        }
        results.append(row)
        models_dict[model_name] = model

    df_results = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False).reset_index(drop=True)
    return df_results, models_dict
