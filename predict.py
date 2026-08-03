"""День 6 — inference pipeline для новых текстов.

Загружает сохранённую best-модель и векторайзер, применяет ТОТ ЖЕ препроцессинг,
что и при обучении (preprocess.clean_text), и классифицирует новые новости —
без переобучения.

Формат входа : строка или список строк (сырой текст новости).
Формат выхода: список dict-ов {text, label (id), sentiment (str), confidence}.

Использование из кода:
    from predict import predict_sentiment
    predict_sentiment("Apple shares soared after record earnings")

Запуск как скрипт — демонстрация на нескольких примерах:
    python predict.py
    python predict.py "your custom headline here"
"""
from __future__ import annotations

import sys
from functools import lru_cache

import joblib
import numpy as np

from config import MODELS_DIR, ID2LABEL
from preprocess import clean_text
# Импорт нужен для десериализации кастомного векторайзера из train_improved.
from train_improved import WordCharVectorizer, LemmaTfidfVectorizer  # noqa: F401


@lru_cache(maxsize=1)
def _load():
    """Ленивая загрузка модели и векторайзера (кэшируется между вызовами)."""
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    vectorizer = joblib.load(MODELS_DIR / "best_vectorizer.pkl")
    return model, vectorizer


def _confidence(model, X) -> np.ndarray:
    """Возвращает уверенность в [0,1] независимо от типа модели.

    LogReg -> max предсказанной вероятности; LinearSVC (без predict_proba)
    -> softmax по значениям decision_function.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X).max(axis=1)
    scores = model.decision_function(X)
    scores = np.atleast_2d(scores)
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    return (exp / exp.sum(axis=1, keepdims=True)).max(axis=1)


def predict_sentiment(texts) -> list[dict]:
    """Классифицирует один текст или список текстов.

    Всегда прогоняет тот же clean_text, что и на обучении, поэтому inference
    согласован с train. Возвращает список результатов (по одному на текст).
    """
    if isinstance(texts, str):
        texts = [texts]
    model, vectorizer = _load()

    cleaned = [clean_text(t) for t in texts]
    X = vectorizer.transform(cleaned)
    preds = model.predict(X)
    conf = _confidence(model, X)

    return [
        {
            "text": raw,
            "label": int(p),
            "sentiment": ID2LABEL[int(p)],
            "confidence": round(float(c), 4),
        }
        for raw, p, c in zip(texts, preds, conf)
    ]


DEMO_TEXTS = [
    "Apple shares soared after the company reported record quarterly earnings.",
    "The company announced massive layoffs and slashed its full-year guidance.",
    "The central bank left interest rates unchanged, in line with expectations.",
    "$TSLA plunged 8% on weak delivery numbers.",
    "Revenue was roughly flat versus the prior quarter.",
]


def main() -> None:
    texts = sys.argv[1:] if len(sys.argv) > 1 else DEMO_TEXTS
    for r in predict_sentiment(texts):
        print(f"[{r['sentiment']:>8}] (conf={r['confidence']:.2f})  {r['text']}")


if __name__ == "__main__":
    main()
