# Trading News Classifier — анализ тональности финансовых новостей

Учебный ML mini-product, который проходит полный цикл text classification:
**data → preprocessing → baseline → improvement → error analysis → inference**.

Модель классифицирует финансовую новость по тональности на 3 класса:
`negative` / `neutral` / `positive`.

---

## Задача и датасет

- **Задача:** многоклассовая классификация тональности (3 класса).
- **Датасет:** `data/data.csv` (FinancialPhraseBank-style), колонки `Sentence`, `Sentiment`.
- **Объём:** 5 322 уникальных примера после дедупликации.
- **Баланс классов (несбалансирован):** neutral ≈ 54%, positive ≈ 35%, negative ≈ 11%.
  Поэтому основная метрика — **macro F1** (одинаково учитывает редкий класс `negative`).

Внутри проекта колонки приводятся к каноничным именам `text` / `sentiment`
(см. `COLUMN_MAP` в [config.py](config.py)).

---

## Структура проекта

```
trading_news_classifier/
├── data/data.csv           # датасет
├── config.py               # общие пути, маппинг колонок и меток, random_state
├── load_data.py            # День 1: загрузка + EDA
├── preprocess.py           # День 2: очистка текста + признаки (переиспользуемый модуль)
├── train_baseline.py       # День 3: baseline TF-IDF + LogReg
├── train_improved.py       # День 4: сравнение улучшений, выбор лучшей модели
├── error_analysis.py       # День 5: confusion matrix + разбор ошибок
├── predict.py              # День 6: inference на новых текстах
├── models/                 # сохранённые модели/векторайзеры (*.pkl)
├── reports/                # метрики, графики, отчёты
├── notebooks/              # day1_eda .. day7_summary — по ноутбуку на день
└── requirements.txt
```

---

## Установка

```bash
pip install -r requirements.txt
# модель для лемматизации (нужна для одного из подходов Дня 4)
python -m spacy download en_core_web_sm
```

Зависимости: `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `joblib`, `spacy`.

---

## Как запускать

Скрипты рассчитаны на последовательный запуск (каждый использует артефакты предыдущего).
Запускать из корня проекта.

### 1. EDA
```bash
python load_data.py
```
Печатает обзор данных и сохраняет графики + выводы в `reports/`
(`sentiment_distribution.png`, `text_length_distribution.png`, `day1_eda_notes.txt`).

### 2. Обучение baseline
```bash
python train_baseline.py
```
Обучает TF-IDF + Logistic Regression, фиксирует train/test split
(`reports/split_indices.json`) и сохраняет `models/baseline_*.pkl`
+ `reports/baseline_result.txt`.

### 3. Улучшения и выбор лучшей модели
```bash
python train_improved.py
```
Сравнивает несколько подходов на **том же** test set, сохраняет таблицу
(`reports/improvement_comparison.txt`) и лучшую модель `models/best_*.pkl`.

### 4. Error analysis
```bash
python error_analysis.py
```
Строит `reports/confusion_matrix.png` и `reports/error_analysis.txt`.

### 5. Inference (предсказание на новых текстах)
```bash
# демо на встроенных примерах
python predict.py

# своя новость
python predict.py "Apple shares soared after record quarterly earnings"
```

Из кода:
```python
from predict import predict_sentiment

predict_sentiment("The company announced massive layoffs")
# -> [{'text': ..., 'label': 1, 'sentiment': 'neutral', 'confidence': 0.64}]
```

**Формат входа:** строка или список строк (сырой текст).
**Формат выхода:** список `dict` — `{text, label (0/1/2), sentiment, confidence}`.
Inference применяет тот же `clean_text`, что и обучение, — переобучение не требуется.

---

## Результаты

Основная метрика — **macro F1** на отложенном test set (20%, стратифицированный split).

Baseline — `TF-IDF(1,2) + LogisticRegression(max_iter=200, n_jobs=-1)`
(без балансировки классов). Все улучшения сравниваются на **том же** test set.
Улучшение указано в процентных пунктах macro F1 к baseline.

| Метод | Macro F1 | Улучшение |
|-------|----------|-----------|
| Baseline: TF-IDF(1,2) + LogReg | 0.5987 | — |
| TF-IDF(1,2) + RandomForest | 0.5609 | −3.78% |
| TF-IDF(1,3) + LinearSVC | 0.6646 | +6.59% |
| Word+Char TF-IDF + LinearSVC | 0.6800 | +8.13% |
| TF-IDF(1,2) sublinear + LogReg | 0.6948 | +9.61% |
| Word+Char + LinearSVC (tuned C=0.5) | 0.6983 | +9.96% |
| **Lemmatization (spacy) + TF-IDF + LogReg** | **0.7147** | **+11.60%** |

**Лучшая модель:** лемматизация (spacy `en_core_web_sm`) + `TF-IDF(1,2)` +
`LogisticRegression(class_weight='balanced')`. Лемматизация схлопывает формы
слов и снижает разреженность признаков, а `class_weight='balanced'` поднимает
recall редкого класса `negative`. Цель «улучшить macro F1 минимум на 5%»
достигнута (**+11.60 п.п.**). `RandomForest` на разреженных TF-IDF уходит в
majority-класс `neutral` и даёт худший результат.

### Error analysis (кратко)
- Основная путаница — между **neutral и positive** (фактологические новости
  с числами модель тянет в neutral, теряя слабый эмоциональный сигнал).
- Класс **negative** самый редкий → самый низкий precision.
- Тяжелее всего — длинные предложения с несколькими фактами разной тональности.

Полные отчёты — в `reports/`.

---

## Demo-примеры

```
[positive] (conf=0.49)  Apple shares soared after the company reported record quarterly earnings.
[ neutral] (conf=0.64)  The company announced massive layoffs and slashed its full-year guidance.
[positive] (conf=0.46)  The central bank left interest rates unchanged, in line with expectations.
[negative] (conf=0.89)  $TSLA plunged 8% on weak delivery numbers.
[ neutral] (conf=0.48)  Revenue was roughly flat versus the prior quarter.
```

---

## Ограничения модели

- **Классическая ML-модель** (TF-IDF + линейный классификатор): понимает слова
  и n-граммы, но не контекст и сарказм; пример про «massive layoffs» выше
  ошибочно уходит в `neutral`.
- **Только английский** язык и стиль новостных заголовков — на другом домене
  качество упадёт.
- **Дисбаланс классов**: `negative` предсказывается менее надёжно из-за малого
  числа примеров.
- **Confidence** у лучшей модели (LogReg) — это max предсказанной вероятности
  (`predict_proba`); модель не откалибрована, поэтому значение стоит трактовать
  как относительную уверенность, а не как точную вероятность. Для моделей без
  `predict_proba` (напр. `LinearSVC`) `predict.py` использует softmax по
  `decision_function`.
- **Не является инвестиционной рекомендацией** — учебный проект.

**Куда развивать:** контекстные эмбеддинги (например finBERT), больше данных
класса `negative`, финансовые лексиконы тональности как дополнительные признаки.

---

<i>Учебный single-task classification mini-product.</i> 🚀
