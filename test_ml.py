import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.model import train_model, inference, compute_model_metrics


def test_train_model_returns_model():
    """
    Testa se train_model retorna um modelo treinado do tipo esperado.
    """
    # dados falsos simples
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference_output_shape():
    """
    Testa se inference retorna previsões com o mesmo número de amostras.
    """
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])

    model = train_model(X, y)
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]


def test_compute_model_metrics_values():
    """
    Testa se compute_model_metrics retorna valores corretos
    para um caso simples conhecido.
    """
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # cálculos esperados:
    # TP=1, FP=0, FN=1
    # precision = 1 / (1+0) = 1.0
    # recall = 1 / (1+1) = 0.5
    # F1 = 2*(1*0.5)/(1+0.5) = 0.6666...
    assert pytest.approx(precision, 0.01) == 1.0
    assert pytest.approx(recall, 0.01) == 0.5
    assert pytest.approx(fbeta, 0.01) == 0.6666
