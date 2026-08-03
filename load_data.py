"""День 1 — загрузка датасета, первичный анализ (EDA) и базовые выводы.

Запуск:
    python load_data.py

Делает:
  * загружает data/data.csv, приводит колонки к каноничным (text, sentiment);
  * печатает размер, типы, пропуски, распределение классов, статистику длин;
  * сохраняет графики в reports/ (распределение классов, длины текстов);
  * сохраняет краткие выводы EDA в reports/day1_eda_notes.txt.
"""
from __future__ import annotations

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # без GUI, только сохранение в файлы
import matplotlib.pyplot as plt
import seaborn as sns

from config import DATA_PATH, REPORTS_DIR, COLUMN_MAP


def load_raw() -> pd.DataFrame:
    """Загружает CSV и приводит имена колонок к каноничным (text, sentiment)."""
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=COLUMN_MAP)
    if not {"text", "sentiment"}.issubset(df.columns):
        raise ValueError(
            f"Ожидались колонки text/sentiment после переименования, получено: {list(df.columns)}"
        )
    return df


def basic_overview(df: pd.DataFrame) -> list[str]:
    """Печатает и возвращает строки первичного анализа."""
    lines: list[str] = []

    def out(msg: str) -> None:
        print(msg)
        lines.append(msg)

    out(f"Размерность (строки, колонки): {df.shape}")
    out("")
    out("Первые 5 строк:")
    out(df.head().to_string())
    out("")
    out("Типы данных / непустые значения:")
    out(df.dtypes.to_string())
    out("")
    out("Пропуски по колонкам:")
    out(df.isnull().sum().to_string())
    out("")

    # Удаляем строки без текста и дубликаты — фиксируем, сколько потеряли.
    before = len(df)
    df.dropna(subset=["text"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    out(f"После удаления пропусков и дубликатов текста: {before} -> {len(df)} строк")
    out("")

    out("Распределение классов (кол-во):")
    out(df["sentiment"].value_counts().to_string())
    out("")
    out("Распределение классов (доля):")
    out((df["sentiment"].value_counts(normalize=True).round(3)).to_string())
    out("")

    return lines


def length_analysis(df: pd.DataFrame) -> list[str]:
    """Считает длину текста и число слов, возвращает статистику."""
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()

    lines = ["Статистика длины текста (символы):", df["text_length"].describe().round(1).to_string(),
             "", "Статистика количества слов:", df["word_count"].describe().round(1).to_string(), ""]
    for line in lines:
        print(line)
    return lines


def examples_per_class(df: pd.DataFrame) -> list[str]:
    """Печатает по одному примеру на каждый класс."""
    lines = ["Примеры по классам:"]
    for sentiment in sorted(df["sentiment"].unique()):
        example = df.loc[df["sentiment"] == sentiment, "text"].iloc[0]
        lines.append(f"[{sentiment}] {example}")
    lines.append("")
    for line in lines:
        print(line)
    return lines


def make_plots(df: pd.DataFrame) -> None:
    """Сохраняет графики распределения классов и длин текстов."""
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 5))
    order = df["sentiment"].value_counts().index
    sns.countplot(data=df, x="sentiment", order=order, hue="sentiment", legend=False)
    plt.title("Распределение классов")
    plt.xlabel("Тональность")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "sentiment_distribution.png", dpi=120)
    plt.close()

    plt.figure(figsize=(7, 5))
    df["text_length"].hist(bins=50)
    plt.xlabel("Длина текста (символы)")
    plt.ylabel("Количество")
    plt.title("Распределение длин текстов")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "text_length_distribution.png", dpi=120)
    plt.close()

    print(f"Графики сохранены в {REPORTS_DIR}")


def main() -> None:
    df = load_raw()
    notes: list[str] = ["=== День 1 — EDA финансовых новостей ===", ""]
    notes += basic_overview(df)
    notes += length_analysis(df)
    notes += examples_per_class(df)
    make_plots(df)

    notes += [
        "Первые наблюдения:",
        "- Задача: классификация тональности новостей на 3 класса (negative/neutral/positive).",
        "- Классы несбалансированы: neutral заметно преобладает — это важно для выбора метрики (macro F1).",
        "- Тексты короткие (одно-два предложения), длина укладывается в разумный диапазон.",
        "- Есть биржевые тикеры ($ESI и т.п.) и числа — потенциально полезные признаки.",
    ]
    for line in notes[-6:]:
        print(line)

    notes_path = REPORTS_DIR / "day1_eda_notes.txt"
    notes_path.write_text("\n".join(notes), encoding="utf-8")
    print(f"\nВыводы EDA сохранены в {notes_path}")


if __name__ == "__main__":
    main()
