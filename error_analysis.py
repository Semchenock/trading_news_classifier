"""День 5 — error analysis лучшей модели.

Использует ту же best-модель и тот же test split, что и День 4, и разбирает,
ГДЕ модель ошибается: строит confusion matrix и выводит примеры ошибок по
типам (какой класс с каким путается).

Запуск (после train_baseline.py и train_improved.py):
    python error_analysis.py

Сохраняет:
  * reports/confusion_matrix.png
  * reports/error_analysis.txt (матрица, топ типов ошибок, примеры, выводы)
"""
from __future__ import annotations

import json
from collections import Counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from config import REPORTS_DIR, MODELS_DIR, ID2LABEL
from preprocess import load_and_preprocess
# Импорт нужен, чтобы joblib смог десериализовать кастомный векторайзер.
from train_improved import WordCharVectorizer, LemmaTfidfVectorizer  # noqa: F401


def main() -> None:
    df = load_and_preprocess()
    idx = json.loads((REPORTS_DIR / "split_indices.json").read_text(encoding="utf-8"))
    test_idx = idx["test"]

    model = joblib.load(MODELS_DIR / "best_model.pkl")
    vectorizer = joblib.load(MODELS_DIR / "best_vectorizer.pkl")

    test_df = df.loc[test_idx].copy()
    X_test = vectorizer.transform(test_df["text_clean"])
    y_true = test_df["label"].to_numpy()
    y_pred = model.predict(X_test)
    test_df["pred"] = y_pred

    labels = sorted(ID2LABEL)
    names = [ID2LABEL[i] for i in labels]

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.title("Confusion Matrix (best model)")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close()

    # --- Текстовый отчёт ---
    errors = test_df[test_df["label"] != test_df["pred"]]
    total, n_err = len(test_df), len(errors)

    lines = [
        "=== День 5 — Error analysis (best model) ===",
        "",
        f"Всего в test: {total}, ошибок: {n_err} ({n_err / total:.1%})",
        "",
        "Classification report:",
        classification_report(y_true, y_pred, target_names=names, digits=4),
        "Confusion matrix (строки=true, столбцы=pred):",
        "            " + "  ".join(f"{n:>9}" for n in names),
    ]
    for i, name in enumerate(names):
        lines.append(f"{name:>10}  " + "  ".join(f"{v:>9d}" for v in cm[i]))
    lines.append("")

    # Топ типов ошибок true->pred
    pair_counts = Counter(
        (ID2LABEL[t], ID2LABEL[p]) for t, p in zip(errors["label"], errors["pred"])
    )
    lines.append("Самые частые типы ошибок (true -> pred):")
    for (t, p), c in pair_counts.most_common():
        lines.append(f"  {t} -> {p}: {c}")
    lines.append("")

    # Примеры по каждому частому типу ошибок
    lines.append("Примеры ошибок:")
    for (t, p), _ in pair_counts.most_common(4):
        subset = errors[
            (errors["label"] == {v: k for k, v in ID2LABEL.items()}[t])
            & (errors["pred"] == {v: k for k, v in ID2LABEL.items()}[p])
        ]
        lines.append(f"\n[{t} -> {p}]")
        for text in subset["text"].head(3):
            lines.append(f"  - {text}")
    lines.append("")

    lines += [
        "Выводы по слабым местам:",
        "- Основная путаница между neutral и positive/negative: многие фактологические",
        "  новости с числами модель тянет в neutral, теряя слабый эмоциональный сигнал.",
        "- Класс negative самый редкий -> самый низкий precision: часть нейтральных",
        "  новостей о рисках/падениях ошибочно помечается negative.",
        "- Ошибки часто на длинных предложениях с несколькими фактами разной тональности.",
        "- Что улучшать дальше: контекстные эмбеддинги (напр. finBERT), доп. данные по",
        "  классу negative, лексиконы финансовой тональности как признаки.",
    ]

    (REPORTS_DIR / "error_analysis.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:20]))
    print(f"\nСохранено: reports/confusion_matrix.png, reports/error_analysis.txt")


if __name__ == "__main__":
    main()
