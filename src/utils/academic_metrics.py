"""
Gerador de Tabelas e Métricas Estatísticas para Publicações Acadêmicas (LaTeX / Markdown).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def _fmt(value: Any) -> str:
    """Formata um valor numérico ou string para célula de tabela."""
    return f"{value:.3f}" if isinstance(value, (int, float)) else str(value)


def generate_latex_table(df_benchmark: pd.DataFrame, caption: str = "Performance Comparison of Machine Learning Models for ECG and Cardiovascular Risk Assessment", label: str = "tab:ecg_benchmark") -> str:
    """
    Exporta o dataframe de benchmark no formato de tabela LaTeX (compatível com IEEE/Elsevier/Springer).
    """
    latex_code = f"""
\\begin{{table*}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{lcccccccc}}
\\hline
\\textbf{{Model}} & \\textbf{{Acc (Test)}} & \\textbf{{Acc (CV)}} & \\textbf{{Sensitivity}} & \\textbf{{Specificity}} & \\textbf{{Precision}} & \\textbf{{F1-Score}} & \\textbf{{ROC-AUC}} & \\textbf{{Brier}} \\\\
\\hline
"""
    for _, row in df_benchmark.iterrows():
        cols = [
            str(row.get('Modelo', 'Model')),
            _fmt(row.get('Acurácia (Teste)', 0.0)),
            str(row.get('Acurácia CV (Média ± DP)', '-')),
            _fmt(row.get('Sensibilidade / Recall', 0.0)),
            _fmt(row.get('Especificidade', 0.0)),
            _fmt(row.get('Precisão', 0.0)),
            _fmt(row.get('F1-Score', 0.0)),
            _fmt(row.get('ROC-AUC', 0.0)),
            _fmt(row.get('Brier Score', 0.0)),
        ]
        latex_code += " & ".join(cols) + " \\\\\n"

    latex_code += """\\hline
\\end{tabular}
\\end{table*}
"""
    return latex_code.strip()


def generate_markdown_table(df_benchmark: pd.DataFrame) -> str:
    """
    Exporta o dataframe de benchmark no formato Markdown formatado para relatórios.
    """
    return df_benchmark.to_markdown(index=False)


def calculate_metric_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_bootstraps: int = 1000,
    alpha: float = 0.95
) -> Dict[str, Tuple[float, float]]:
    """
    Calcula intervalos de confiança de 95% via bootstrap não paramétrico para
    acurácia, sensibilidade, especificidade, F1-Score e ROC-AUC (quando y_proba fornecido).
    """
    from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

    rng = np.random.RandomState(42)
    n_samples = len(y_true)
    lower_p = (1.0 - alpha) / 2.0
    upper_p = (1.0 + alpha) / 2.0

    scores: Dict[str, List[float]] = {
        'accuracy': [], 'sensitivity': [], 'specificity': [], 'f1': []
    }
    if y_proba is not None:
        scores['roc_auc'] = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, n_samples)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        tn = np.sum((yt == 0) & (yp == 0))
        fp = np.sum((yt == 0) & (yp == 1))
        scores['accuracy'].append(accuracy_score(yt, yp))
        scores['sensitivity'].append(recall_score(yt, yp, zero_division=0))
        scores['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        scores['f1'].append(f1_score(yt, yp, zero_division=0))
        if y_proba is not None:
            scores['roc_auc'].append(roc_auc_score(yt, y_proba[idx]))

    result: Dict[str, Tuple[float, float]] = {}
    for metric, vals in scores.items():
        if vals:
            sv = np.sort(vals)
            result[f"{metric}_ci_{int(alpha*100)}"] = (
                float(sv[int(lower_p * len(sv))]),
                float(sv[int(upper_p * len(sv))])
            )
    return result
