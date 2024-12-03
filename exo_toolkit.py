import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score


def plot_confusion_matrix(y_obs, y_exp):
    cm = confusion_matrix(y_obs, y_exp)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='viridis')
    plt.title("Confusion Matrix")
    plt.show()

def evaluate_thresholds(y_true, y_probs, thresholds=None):
    """
    Evaluate precision, recall, and F1-score for a range of thresholds.

    Parameters:
    - y_true: Ground truth (array-like)
    - y_probs: Predicted probabilities or scores (array-like)
    - thresholds: List or array of thresholds to evaluate. Default is np.linspace(0, 1, 100).

    Returns:
    - metrics: Dictionary with keys 'thresholds', 'precision', 'recall', 'f1'
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)

    precisions, recalls, f1s = [], [], []

    for thresh in thresholds:
        y_pred = (y_probs >= thresh)
        precisions.append(precision_score(y_true, y_pred))
        recalls.append(recall_score(y_true, y_pred))
        f1s.append(f1_score(y_true, y_pred))

    return {
        "thresholds": thresholds,
        "precision": np.array(precisions),
        "recall": np.array(recalls),
        "f1": np.array(f1s),
    }


def find_optimal_threshold(metrics, metric="f1"):
    """
    Find the optimal threshold for a specific metric.

    Parameters:
    - metrics: Dictionary returned by `evaluate_thresholds`
    - metric: Metric to optimize ('precision', 'recall', or 'f1')

    Returns:
    - optimal_threshold: The threshold that optimizes the chosen metric
    """
    metric_values = metrics[metric]
    optimal_idx = np.argmax(metric_values)
    optimal_threshold = metrics["thresholds"][optimal_idx]
    return optimal_threshold


def plot_metrics(metrics):
    """
    Plot precision, recall, and F1-score vs. thresholds.

    Parameters:
    - metrics: Dictionary returned by `evaluate_thresholds`
    """
    thresholds = metrics["thresholds"]
    plt.plot(thresholds, metrics["precision"], label="Precision")
    plt.plot(thresholds, metrics["recall"], label="Recall")
    plt.plot(thresholds, metrics["f1"], label="F1 Score")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Metrics vs. Threshold")
    plt.legend()
    plt.show()