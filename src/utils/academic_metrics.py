"""
Gerador de Tabelas e Métricas Estatísticas para Publicações Acadêmicas (LaTeX / Markdown).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


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
        model_name = row.get('Modelo', 'Model')
        acc_test = f"{row.get('Acurácia (Teste)', 0.0):.3f}" if isinstance(row.get('Acurácia (Teste)'), (int, float)) else str(row.get('Acurácia (Teste)'))
        acc_cv = str(row.get('Acurácia CV (Média ± DP)', '-'))
        sens = f"{row.get('Sensibilidade / Recall', 0.0):.3f}" if isinstance(row.get('Sensibilidade / Recall'), (int, float)) else str(row.get('Sensibilidade / Recall'))
        spec = f"{row.get('Especificidade', 0.0):.3f}" if isinstance(row.get('Especificidade'), (int, float)) else str(row.get('Especificidade'))
        prec = f"{row.get('Precisão', 0.0):.3f}" if isinstance(row.get('Precisão'), (int, float)) else str(row.get('Precisão'))
        f1 = f"{row.get('F1-Score', 0.0):.3f}" if isinstance(row.get('F1-Score'), (int, float)) else str(row.get('F1-Score'))
        roc_auc = f"{row.get('ROC-AUC', 0.0):.3f}" if isinstance(row.get('ROC-AUC'), (int, float)) else str(row.get('ROC-AUC'))
        brier = f"{row.get('Brier Score', 0.0):.3f}" if isinstance(row.get('Brier Score'), (int, float)) else str(row.get('Brier Score'))

        latex_code += f"{model_name} & {acc_test} & {acc_cv} & {sens} & {spec} & {prec} & {f1} & {roc_auc} & {brier} \\\\\n"

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


def calculate_metric_confidence_intervals(y_true: np.ndarray, y_pred: np.ndarray, n_bootstraps: int = 1000, alpha: float = 0.95) -> Dict[str, Tuple[float, float]]:
    """
    Calcula intervalos de confiança de 95% via bootstrap não paramétrico para métricas clínicas.
    """
    rng = np.random.RandomState(42)
    bootstrapped_scores = []
    
    n_samples = len(y_true)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = np.mean(y_true[indices] == y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(bootstrapped_scores)
    lower = sorted_scores[int((1.0 - alpha) / 2.0 * len(sorted_scores))]
    upper = sorted_scores[int((1.0 + alpha) / 2.0 * len(sorted_scores))]

    return {
        'accuracy_ci_95': (float(lower), float(upper))
    }
