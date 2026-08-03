"""День 3 — baseline модель классификации тональности.

Pipeline: TF-IDF (1-2 граммы) -> Logistic Regression.
Это точка отсчёта, относительно которой оцениваются улучшения (День 4).

Запуск:
    python train_baseline.py

Сохраняет:
  * models/baseline_model.pkl, models/baseline_vectorizer.pkl
  * reports/baseline_result.txt (macro F1 + classification report)
  * reports/split_indices.json (индексы train/test для честного сравнения)
"""
from __future__ import annotations

import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from config import REPORTS_DIR, MODELS_DIR, RANDOM_STATE, ID2LABEL
from preprocess import load_and_preprocess


def get_split():
    """Готовит данные и делает стратифицированный train/test split.

    Возвращает (df, train_idx, test_idx). Индексы сохраняются на диск, чтобы
    День 4 и День 5 сравнивались на ровно том же test set.
    """
    df = load_and_preprocess()
    train_idx, test_idx = train_test_split(
        df.index.to_numpy(),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"].to_numpy(),
    )
    (REPORTS_DIR / "split_indices.json").write_text(
        json.dumps({"train": train_idx.tolist(), "test": test_idx.tolist()}),
        encoding="utf-8",
    )
    return df, train_idx, test_idx


def main() -> None:
    df, train_idx, test_idx = get_split()

    X_train_text = df.loc[train_idx, "text_clean"]
    X_test_text = df.loc[test_idx, "text_clean"]
    y_train = df.loc[train_idx, "label"].to_numpy()
    y_test = df.loc[test_idx, "label"].to_numpy()

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(max_iter=200, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    target_names = [ID2LABEL[i] for i in sorted(ID2LABEL)]
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(report)
    print(f"Baseline macro F1: {macro_f1:.4f}")

    joblib.dump(model, MODELS_DIR / "baseline_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "baseline_vectorizer.pkl")

    result_text = (
        "=== Baseline: TF-IDF (1,2) + LogisticRegression(max_iter=200, n_jobs=-1) ===\n"
        f"Train size: {len(train_idx)}, Test size: {len(test_idx)}\n\n"
        f"{report}\n"
        f"Baseline macro F1: {macro_f1:.4f}\n"
    )
    (REPORTS_DIR / "baseline_result.txt").write_text(result_text, encoding="utf-8")
    print(f"\nСохранено: models/baseline_*.pkl, reports/baseline_result.txt")


if __name__ == "__main__":
    main()
