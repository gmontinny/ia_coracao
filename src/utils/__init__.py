"""
Módulo de Utilitários de Visualização e Métricas Acadêmicas.
"""

from .visualization import (
    plot_roc_comparison,
    plot_pr_comparison,
    plot_confusion_matrix_heatmap,
    plot_feature_importances_chart,
    plot_calibration_curve_custom
)
from .academic_metrics import (
    generate_latex_table,
    generate_markdown_table,
    calculate_metric_confidence_intervals
)

__all__ = [
    "plot_roc_comparison",
    "plot_pr_comparison",
    "plot_confusion_matrix_heatmap",
    "plot_feature_importances_chart",
    "plot_calibration_curve_custom",
    "generate_latex_table",
    "generate_markdown_table",
    "calculate_metric_confidence_intervals"
]
