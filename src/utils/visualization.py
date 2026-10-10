"""
Utilitários de Visualização Gráfica e Figuras de Qualidade para Publicação Científica.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.calibration import calibration_curve


def plot_roc_comparison(models_dict: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> plt.Figure:
    """
    Plota as curvas ROC comparativas de múltiplos modelos com os respectivos scores AUC.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    colors = ['#0d6efd', '#198754', '#dc3545', '#fd7e14', '#6f42c1', '#20c997', '#d63384', '#0dcaf0']
    
    for idx, (name, model) in enumerate(models_dict.items()):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            label_name = name.replace('_', ' ').title()
            color = colors[idx % len(colors)]
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{label_name} (AUC = {roc_auc:.3f})")
        except Exception as e:
            ax.plot([], [], label=f"{name} (erro: {e})")
            continue

    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Aleatório / Linha de Base (AUC = 0.500)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade / Recall)', fontsize=11, fontweight='bold')
    ax.set_title('Curvas ROC - Comparativo de Modelos Preditivos de Risco Cardiológico', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    return fig


def plot_pr_comparison(models_dict: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> plt.Figure:
    """
    Plota curvas de Precisão-Recall comparativas.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    colors = ['#0d6efd', '#198754', '#dc3545', '#fd7e14', '#6f42c1', '#20c997', '#d63384', '#0dcaf0']

    for idx, (name, model) in enumerate(models_dict.items()):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
            label_name = name.replace('_', ' ').title()
            color = colors[idx % len(colors)]
            ax.plot(rec, prec, color=color, lw=2, label=f"{label_name} (PR-AUC = {pr_auc:.3f})")
        except Exception as e:
            ax.plot([], [], label=f"{name} (erro: {e})")
            continue

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Sensibilidade / Recall (Taxa de Identificação de Cardiopatia)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Precisão (Valor Preditivo Positivo)', fontsize=11, fontweight='bold')
    ax.set_title('Curvas de Precisão-Recall para Estratificação de Risco', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', frameon=True, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    return fig


def plot_confusion_matrix_heatmap(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray], model_name: str = "Modelo") -> plt.Figure:
    """
    Gera mapa de calor da matriz de confusão com detalhamento clínico.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    labels = [
        [f"Verdadeiro Negativo\n(TN = {tn})", f"Falso Positivo\n(FP = {fp})"],
        [f"Falso Negativo (Risco!)\n(FN = {fn})", f"Verdadeiro Positivo\n(TP = {tp})"]
    ]

    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Pred: Normal (0)', 'Pred: Cardiopatia (1)'],
                yticklabels=['Real: Normal (0)', 'Real: Cardiopatia (1)'],
                annot_kws={'size': 10, 'weight': 'bold'})

    ax.set_title(f'Matriz de Confusão Clínica - {model_name}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Condição Clínica Real', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predição do Sistema de IA', fontsize=11, fontweight='bold')
    
    return fig


def plot_feature_importances_chart(feature_importances: Dict[str, float], top_n: int = 10, title: str = "Importância das Variáveis Clínicas") -> plt.Figure:
    """
    Gera gráfico de barras horizontais com as variáveis mais determinantes no modelo.
    """
    items = list(feature_importances.items())[:top_n]
    if not items:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sem dados de importância para o modelo", ha='center')
        return fig

    feats, values = zip(*items)
    
    # Formatação amigável dos nomes
    clean_feats = [f.replace('num__', '').replace('cat__', '').replace('_', ' ') for f in feats]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    y_pos = np.arange(len(clean_feats))
    
    ax.barh(y_pos, values, color='#0d6efd', edgecolor='#0b5ed7', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_feats, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Importância Relativa Normalizada', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='x', linestyle=':', alpha=0.6)

    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f"{v * 100:.1f}%", va='center', fontsize=9, fontweight='bold', color='#212529')

    return fig


def plot_calibration_curve_custom(models_dict: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> plt.Figure:
    """
    Gera diagrama de calibração de probabilidade diagnóstica.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot([0, 1], [0, 1], "k:", label="Perfeitamente Calibrado")

    for name, model in models_dict.items():
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=8)
            label_name = name.replace('_', ' ').title()
            ax.plot(prob_pred, prob_true, "s-", label=label_name)
        except Exception as e:
            ax.plot([], [], label=f"{name} (erro: {e})")
            continue

    ax.set_ylabel("Fração de Casos Positivos Reais", fontsize=11, fontweight='bold')
    ax.set_xlabel("Probabilidade Média Prevista", fontsize=11, fontweight='bold')
    ax.set_title("Diagrama de Confiabilidade e Calibração Diagnóstica", fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    return fig
