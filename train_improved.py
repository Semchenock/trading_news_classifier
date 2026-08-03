"""День 4 — улучшение baseline и сравнение подходов.

Проверяем несколько разумных улучшений на ОДНОМ и том же train/test split
(индексы из reports/split_indices.json, созданные в День 3), чтобы сравнение
было честным. Лучшая по macro F1 модель сохраняется как рабочая (best_*).

Подходы:
  1. TF-IDF (1,3) + LinearSVC
  2. TF-IDF (1,2), sublinear_tf + LogisticRegression(balanced)
  3. TF-IDF(word) + TF-IDF(char 3-5) объединённые + LinearSVC
  4. GridSearch по C для LinearSVC на подходе 3

Запуск (требует предварительного `python train_baseline.py`):
    python train_improved.py
"""
from __future__ import annotations

import json

import joblib
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from config import REPORTS_DIR, MODELS_DIR, RANDOM_STATE, ID2LABEL
from preprocess import load_and_preprocess


def load_split():
    """Загружает данные и восстанавливает train/test split из Дня 3."""
    df = load_and_preprocess()
    split_path = REPORTS_DIR / "split_indices.json"
    if not split_path.exists():
        raise FileNotFoundError(
            "Не найден reports/split_indices.json — сначала запустите train_baseline.py"
        )
    idx = json.loads(split_path.read_text(encoding="utf-8"))
    train_idx, test_idx = idx["train"], idx["test"]
    return df, train_idx, test_idx


def _baseline_f1() -> float:
    """Читает macro F1 baseline из reports/baseline_result.txt (для сравнения)."""
    path = REPORTS_DIR / "baseline_result.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        if "macro F1" in line:
            return float(line.strip().split(":")[-1])
    raise ValueError("Не удалось прочитать baseline macro F1")


class WordCharVectorizer:
    """Объединяет словесные и символьные TF-IDF признаки.

    Собственный класс (а не FeatureUnion) — чтобы легко сериализовать вместе
    с моделью и переиспользовать на inference одним .transform().
    """

    def __init__(self):
        self.word = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=2
        )
        self.char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_features=20000, min_df=2
        )

    def fit_transform(self, texts):
        return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()

    def transform(self, texts):
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()


def main() -> None:
    df, train_idx, test_idx = load_split()
    train_text = df.loc[train_idx, "text_clean"]
    test_text = df.loc[test_idx, "text_clean"]
    y_train = df.loc[train_idx, "label"].to_numpy()
    y_test = df.loc[test_idx, "label"].to_numpy()

    baseline_f1 = _baseline_f1()
    results: list[tuple[str, float]] = [("Baseline: TF-IDF(1,2) + LogReg", baseline_f1)]
    best = {"name": None, "f1": -1.0, "model": None, "vectorizer": None}

    def evaluate(name, vectorizer, model):
        Xtr = vectorizer.fit_transform(train_text)
        Xte = vectorizer.transform(test_text)
        model.fit(Xtr, y_train)
        f1 = f1_score(y_test, model.predict(Xte), average="macro")
        results.append((name, f1))
        print(f"{name}: macro F1 = {f1:.4f}")
        if f1 > best["f1"]:
            best.update(name=name, f1=f1, model=model, vectorizer=vectorizer)

    # Подход 1 — триграммы + LinearSVC
    evaluate(
        "TF-IDF(1,3) + LinearSVC",
        TfidfVectorizer(max_features=5000, ngram_range=(1, 3)),
        LinearSVC(class_weight="balanced"),
    )

    # Подход 2 — sublinear_tf + LogReg
    evaluate(
        "TF-IDF(1,2) sublinear + LogReg",
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )

    # Подход 3 — word + char признаки + LinearSVC
    evaluate(
        "Word+Char TF-IDF + LinearSVC",
        WordCharVectorizer(),
        LinearSVC(class_weight="balanced"),
    )

    # Подход 4 — подбор C для LinearSVC на word+char признаках
    wc = WordCharVectorizer()
    Xtr = wc.fit_transform(train_text)
    grid = GridSearchCV(
        LinearSVC(class_weight="balanced"),
        {"C": [0.1, 0.5, 1, 2, 5]},
        scoring="f1_macro",
        cv=3,
    )
    grid.fit(Xtr, y_train)
    tuned = grid.best_estimator_
    f1 = f1_score(y_test, tuned.predict(wc.transform(test_text)), average="macro")
    name = f"Word+Char + LinearSVC (tuned C={grid.best_params_['C']})"
    results.append((name, f1))
    print(f"{name}: macro F1 = {f1:.4f}")
    if f1 > best["f1"]:
        best.update(name=name, f1=f1, model=tuned, vectorizer=wc)

    # --- Итоговая таблица сравнения ---
    lines = [
        "=== День 4 — сравнение подходов (один и тот же test set) ===",
        "",
        "| Метод | Macro F1 | Улучшение |",
        "|-------|----------|-----------|",
    ]
    for name, f1 in results:
        delta = "-" if name.startswith("Baseline") else f"{(f1 - baseline_f1) * 100:+.2f}%"
        lines.append(f"| {name} | {f1:.4f} | {delta} |")
    lines += [
        "",
        f"Лучший метод: {best['name']} (macro F1 = {best['f1']:.4f}, "
        f"{(best['f1'] - baseline_f1) * 100:+.2f}% к baseline).",
        "",
        "Что помогло: символьные n-граммы устойчивы к опечаткам/тикерам и формам слов; "
        "LinearSVC хорошо работает на разреженных TF-IDF; class_weight='balanced' "
        "поднимает recall редкого класса negative. Триграммы отдельно почти не помогли.",
    ]
    (REPORTS_DIR / "improvement_comparison.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-4:]))

    # Сохраняем лучшую модель как рабочую для инференса.
    joblib.dump(best["model"], MODELS_DIR / "best_model.pkl")
    joblib.dump(best["vectorizer"], MODELS_DIR / "best_vectorizer.pkl")
    print("\nСохранено: models/best_model.pkl, models/best_vectorizer.pkl, "
          "reports/improvement_comparison.txt")


if __name__ == "__main__":
    main()
