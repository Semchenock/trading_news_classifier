"""День 2 — очистка текста и подготовка признаков (переиспользуемый модуль).

Этот модуль — единая точка предобработки для всего проекта. Его используют
и обучение (train_baseline / train_improved), и inference (predict), поэтому
тексты обрабатываются одинаково на всех этапах.

Запуск как скрипт печатает демонстрацию на нескольких примерах:
    python preprocess.py
"""
from __future__ import annotations

import re

import pandas as pd

from config import DATA_PATH, COLUMN_MAP, LABEL2ID
from load_data import load_raw

# Скомпилированные регэкспы (быстрее при применении к тысячам строк).
_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"http\S+|www\.\S+")
_TICKER_RE = re.compile(r"\$[A-Za-z]{1,6}\b")  # биржевой тикер, напр. $ESI
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Базовая нормализация одного текста.

    Шаги: нижний регистр -> убрать html/url -> заменить тикеры и числа
    на плейсхолдеры (сохраняем сигнал, но убираем разреженность) -> схлопнуть
    пробелы. Функция детерминирована и не зависит от внешнего состояния.
    """
    text = str(text).lower()
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _TICKER_RE.sub(" ticker ", text)
    text = _NUMBER_RE.sub(" number ", text)
    text = _MULTISPACE_RE.sub(" ", text)
    return text.strip()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет простые числовые признаки поверх очищенного текста."""
    df["word_count_clean"] = df["text_clean"].str.split().str.len()
    df["char_count"] = df["text_clean"].str.len()
    # Считаем упоминания денег/тикеров по исходному тексту (в очищенном они заменены).
    df["dollar_count"] = df["text"].str.count(r"\$")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Полный препроцессинг датафрейма с колонками text/sentiment.

    Возвращает копию с колонками text_clean, label (int) и доп. признаками.
    Пустые после очистки строки и неизвестные метки удаляются.
    """
    df = df.copy()
    df["text_clean"] = df["text"].apply(clean_text)
    df = df[df["text_clean"].str.len() > 0]

    df["label"] = df["sentiment"].map(LABEL2ID)
    unknown = df["label"].isnull().sum()
    if unknown:
        print(f"Внимание: {unknown} строк с неизвестной меткой удалено")
    df = df[df["label"].notnull()].copy()
    df["label"] = df["label"].astype(int)

    df = add_features(df)
    return df.reset_index(drop=True)


def load_and_preprocess() -> pd.DataFrame:
    """Удобный хелпер: загрузить сырые данные и сразу препроцессить.

    Дедупликация текста здесь повторяется явно, чтобы функция была
    самодостаточной при вызове из train/predict без прогона load_data.main().
    """
    df = load_raw()
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])
    return preprocess(df)


def _demo() -> None:
    samples = [
        "$ESI on lows, down $1.50 to $2.50 BK a real possibility",
        "Net sales <b>ROSE</b> 12% to EUR 131m   from EUR76m.",
        "Market remains stable despite volatility http://news.example.com",
    ]
    print("=== Демонстрация clean_text ===")
    for s in samples:
        print(f"IN : {s}")
        print(f"OUT: {clean_text(s)}")
        print()

    print("=== Проверка на реальных данных ===")
    df = load_and_preprocess()
    print(f"Готово: {len(df)} строк, колонки: {list(df.columns)}")
    print("Распределение label:")
    print(df["label"].value_counts().sort_index().to_string())
    print("\nПример строки после препроцессинга:")
    print(df.loc[0, ["text", "text_clean", "label", "word_count_clean", "dollar_count"]].to_string())


if __name__ == "__main__":
    _demo()
