<b>Финансовые новости — анализ тональности для трейдеров</b>

<b>НЕОБХОДИМО ДЛЯ НАЧАЛА РАБОТЫ:</b>

Вам понадобится датасет с финансовыми новостями и разметкой тональности.

<b>Откуда взять данные:</b>
1. Kaggle: "Financial Sentiment Analysis" или "Stock News Sentiment"
2. Hugging Face: dataset "financial_phrasebank"
3. Или создайте свой датасет из новостных сайтов

<b>Формат данных (CSV файл):</b>
Должны быть минимум две колонки:
• <code>text</code> — текст новости
• <code>sentiment</code> — тональность (positive/negative/neutral или цифры 0/1/2)

Пример:
<code>text,sentiment</code>
<code>"Apple stocks rose 5% after earnings report",positive</code>
<code>"Company announces massive layoffs",negative</code>
<code>"Market remains stable despite volatility",neutral</code>

---

<b>ЧАСТЬ 1: EDA и разбор данных</b>

<b>1.1 Загрузка и первичный анализ</b>

1. Создайте новую папку для проекта

2. Создайте Jupyter ноутбук или Python скрипт

3. Импортируйте библиотеки:
<code>import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns</code>

4. Загрузите ваш CSV файл:
<code>df = pd.read_csv('ваш_файл.csv')</code>

5. Если колонки называются иначе — переименуйте их:
<code>df = df.rename(columns={'headline': 'text', 'label': 'sentiment'})</code>

6. Выведите размерность: <code>print(df.shape)</code>

7. Выведите первые 5 строк: <code>print(df.head())</code>

8. Выведите информацию о типах данных: <code>print(df.info())</code>

9. Проверьте пропуски: <code>print(df.isnull().sum())</code>

10. Удалите строки с пропусками в тексте: <code>df = df.dropna(subset=['text'])</code>

11. Постройте график распределения классов:
<code>sns.countplot(data=df, x='sentiment')
plt.title('Распределение классов')
plt.savefig('sentiment_distribution.png')</code>

<b>1.2 Анализ длин текстов</b>

1. Создайте колонку с длиной текста:
<code>df['text_length'] = df['text'].str.len()</code>

2. Создайте колонку с количеством слов:
<code>df['word_count'] = df['text'].str.split().str.len()</code>

3. Постройте гистограмму длин:
<code>df['text_length'].hist(bins=50)
plt.xlabel('Длина текста')
plt.ylabel('Количество')
plt.savefig('text_length_distribution.png')</code>

4. Выведите примеры для каждого класса:
<code>for sentiment in df['sentiment'].unique():
    print(f'\n=== Класс {sentiment} ===')
    print(df[df['sentiment'] == sentiment]['text'].iloc[0])</code>

<b>ЧАСТЬ 2: Подготовка данных</b>

<b>2.1 Очистка текста</b>

1. Напишите функцию очистки:
<code>import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text</code>

2. Примените к данным:
<code>df['text_clean'] = df['text'].apply(clean_text)</code>

3. Удалите пустые тексты после очистки:
<code>df = df[df['text_clean'].str.len() > 0]</code>

<b>2.2 Признаки</b>

1. Добавьте колонки с признаками:
<code>df['word_count_clean'] = df['text_clean'].str.split().str.len()
df['char_count'] = df['text_clean'].str.len()
df['dollar_count'] = df['text_clean'].str.count('$')</code>

2. Преобразуйте метки классов в цифры, если они текстовые:
<code>df['label'] = df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
# или используйте LabelEncoder из sklearn</code>

<b>ЧАСТЬ 3: Baseline модель</b>

<b>3.1 Подготовка признаков</b>

1. Импортируйте нужное:
<code>from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split</code>

2. Создайте TF-IDF векторизатор:
<code>vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))</code>

3. Преобразуйте тексты:
<code>X = vectorizer.fit_transform(df['text_clean'])
y = df['label'].values</code>

4. Разделите на train/test:
<code>X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)</code>

<b>3.2 Обучение baseline</b>

1. Импортируйте и создайте модель:
<code>from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

model = LogisticRegression(max_iter=200, n_jobs=-1)</code>

2. Обучите:
<code>model.fit(X_train, y_train)</code>

3. Сделайте предсказания:
<code>y_pred = model.predict(X_test)</code>

4. Выведите отчет:
<code>print(classification_report(y_test, y_pred))</code>

5. Посчитайте macro F1:
<code>f1 = f1_score(y_test, y_pred, average='macro')
print(f'Macro F1: {f1:.4f}')</code>

6. Сохраните baseline результат:
<code>baseline_f1 = f1
with open('baseline_result.txt', 'w') as f:
    f.write(f'Baseline Macro F1: {f1:.4f}')</code>

<b>ЧАСТЬ 4: Улучшение модели</b>

<b>Цель:</b> улучшить macro F1 минимум на 5%

<b>4.1 Выберите подходы (минимум 3):</b>

<b>Подход 1 — Лемматизация:</b>
<code>import spacy
nlp = spacy.load('en_core_web_sm')

def lemmatize(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

df['text_lemm'] = df['text_clean'].apply(lemmatize)</code>

<b>Подход 2 — TF-IDF с триграммами:</b>
<code>vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))</code>

<b>Подход 3 — Другая модель (Random Forest):</b>
<code>from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, n_jobs=-1)</code>

<b>Подход 4 — Другая модель (LinearSVC):</b>
<code>from sklearn.svm import LinearSVC
model = LinearSVC(max_iter=2000)</code>

<b>Подход 5 — Подбор гиперпараметров:</b>
<code>from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=3)
grid.fit(X_train, y_train)
print(f'Лучший C: {grid.best_params_}')</code>

<b>4.2 Реализуйте и протестируйте</b>

1. Для каждого подхода:
   - Подготовьте признаки (если нужно)
   - Обучите модель
   - Посчитайте macro F1
   - Сравните с baseline

2. Сохраняйте лучшие модели:
<code>import joblib
joblib.dump(model, 'best_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')</code>

<b>ЧАСТЬ 5: Сравнение и анализ</b>

<b>5.1 Таблица результатов</b>

Создайте текстовый файл с таблицей:

| Метод | Macro F1 | Улучшение |
|-------|----------|-----------|
| Baseline TF-IDF + LogReg | 0.XX | - |
| [Ваш метод 1] | 0.XX | +X% |
| [Ваш метод 2] | 0.XX | +X% |

<b>5.2 Confusion Matrix</b>

1. Постройте матрицу ошибок:
<code>from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')</code>

<b>5.3 Анализ ошибок</b>

1. Найдите ошибки и сохраните оригинальные тексты до векторизации

2. Выведите несколько примеров ошибок

3. Сохраните анализ в файл <code>error_analysis.txt</code>

<b>ЧАСТЬ 6: Предсказание на новых данных</b>

<b>6.1 Функция для предсказания</b>

1. Напишите функцию:
<code>def predict_sentiment(texts, model, vectorizer, clean_func=None):
    if isinstance(texts, str):
        texts = [texts]
    if clean_func:
        texts = [clean_func(t) for t in texts]
    X = vectorizer.transform(texts)
    predictions = model.predict(X)
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
        return predictions, probs
    return predictions</code>

2. Протестируйте на 3-5 новых примерах и выведите результаты

<b>ЧТО СДАТЬ:</b>
• Jupyter ноутбук или Python скрипты с кодом
• Графики EDA (распределение классов, длины текстов)
• Файл с таблицей сравнения методов
• Confusion Matrix график
• Файл с анализом ошибок
• Обученная лучшая модель

---
<i>Удачи с реализацией!</i> 🚀
