# Pitchfork Review Classifier

Классификация музыкальных рецензий Pitchfork на "хороший" / "плохой" отзыв с помощью классических моделей машинного обучения (TF-IDF + инженерные признаки текста).

Итоговая работа по курсу «Погружение в ИИ: от машинного обучения до БЯМ», Вариант 3.

## О проекте

Датасет - рецензии на музыкальные альбомы с сайта Pitchfork (18 388 рецензий, использована сбалансированная выборка из 5000). Каждой рецензии присвоена числовая оценка (0-10); метка "хороший"/"плохой" отзыв получена делением по медиане оценки (7.2).

Признаки для модели: TF-IDF по тексту рецензии (1000 признаков) + 10 инженерных признаков (длина текста, количество предложений, лексическое разнообразие, количество стоп-слов, пунктуации и т.д.).

Сравнивались 5 моделей: Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost. Лучшая после подбора гиперпараметров - **Logistic Regression** (accuracy ≈ 0.70, F1 ≈ 0.69).

## Структура репозитория

```
├── Кривохат_итоговая.ipynb    - ноутбук: EDA, признаки, обучение и подбор моделей
├── data/
│   └── pitchfork_reviews.csv  - выборка данных (5000 рецензий)
├── main.py                    - FastAPI-сервис
├── feature_engineering.py     - извлечение признаков из текста
├── static/index.html          - минимальный веб-интерфейс
├── model.pkl                  - обученная модель (Logistic Regression)
├── tfidf_vectorizer.pkl       - обученный TF-IDF векторизатор
├── feature_scaler.pkl         - обученный StandardScaler
├── requirements.txt           - зависимости Python
└── sample_predict.csv         - тестовый файл для /predict/file
```

## Как открыть ноутбук

Открыть `Кривохат_итоговая.ipynb` в Google Colab или Jupyter. Данные подгружаются автоматически из `data/pitchfork_reviews.csv` (если клонировать репозиторий целиком).

## Как запустить сервис

### 1. Установить Python

Скачать и установить Python 3.12 или новее с [python.org](https://www.python.org/downloads/) (при установке на Windows обязательно отметить галочку **"Add python.exe to PATH"**).

Проверить установку:

```
python --version
```

### 2. Установить зависимости

Из корня репозитория:

```
pip install -r requirements.txt
```

### 3. Запустить сервис

```
python -m uvicorn main:app --reload
```

Открыть в браузере: **http://127.0.0.1:8000**

## Примеры для проверки

Готовые тексты отзывов (реальные рецензии из датасета) - можно вставить в веб-интерфейс:

**Хороший отзыв:**

Steve Gunn hadn't been born by the time Mike Cooper decided he was done with folk-rock. After becoming one of Britain's brightest young blues pickers and declining an invitation to join the then-fledgling Rolling Stones in the early 60s, Cooper made a string of singer-songwriter records for the British label Pye. That relationship both climaxed and closed in 1972, with the release of Cooper's outlandish The Machine Gun Co.—the last in a series of three transgressive folk-rock explorations, where blooming free jazz and spiraling psychedelics disrupted any simple, sing-and-strum notions. Cooper lost his record contract and, during the next four decades, let his musical investigation and imagination run amuck. He still played the blues, yes, but he also mined electroacoustic improvisation and Hawaiian music, operatic composition and Brion Gysin-like cut-ups. Folk-rock? Nope. Cooper bequeathed that mantle long ago, only for it to be taken up in recent years by the young Gunn and scores of his peers. Once again, they've worked to expand the sounds that such a drab term might entail, picking up the antagonism that once made Cooper an outcast.

But late last year, despite the four decades and the ocean that separates them, Cooper and Gunn rendezvoused in a Portugal airport, launching 10 days of shows and a retreat in a Lisbon studio. The partnership came as a commission from RVNG Intl., the New York-based label that has issued a string of similar one-off encounters—Blues Control, meet Laraaji; Emeralds, meet Alan Howarth—as the aptly named FRKWYS series during the last five years. The edict was simple: Make some new music, and attempt to sort it into a record.

The seven-song result, Cantos de Lisboa, might feel at first like a grab bag, as the pair moves freely between a number of styles, techniques, and instruments. They're both best known for their intimate voices and their intricate picking, but they sing very little here and surround their guitars with a host of distractions. While they begin with twin forlorn guitars on opener "Saudade Do Santos-o-Vehlo", they shift to scraped gongs and scrambled electronics for "Song for Charlie", or at least most of it. "Pony Blues" showcases two dexterous players, winding through lithe licks with the agility of the Shetlands of which Cooper sings. The focus of "Saramago", though, is that of outré players, with scraped strings and manipulated notes, ruptured harmonics and dissonant strums suggesting the acoustic improvisations of Derek Bailey and Eugene Chadbourne. Restless, career-long collaborators on their own, Gunn and Cooper twirl across the electrostatic cello lines of Helena Espvall during "Pena Panorama".

**Плохой отзыв:**

For the sake of transparency, let me start by saying that I'm reviewing Angel Deradoorian's debut EP, Mind Raft, because Angel Deradoorian is the bassist and co-vocalist in Dirty Projectors. I'm guessing she takes this as a mixed blessing: On the one hand, it's probably a little disheartening to consider that the bulk of her audience are DP fans curious about what else the band members might be up to. On the other hand, she's probably fine with the attention, considering that a near-heartbreaking number of albums just as passionate and lovingly crafted as Mind Raft go totally ignored because of time and space limitations. Deradoorian's sound is almost the opposite of the Projectors'-- slow, droning, R&B-influenced folk splattered with blue notes and big drums.

Для проверки пакетной классификации (`/predict/file`) - приложен готовый файл `sample_predict.csv` (5 отзывов):

```
curl -X POST http://127.0.0.1:8000/predict/file -F "file=@sample_predict.csv"
```

## Эндпоинты API

| Метод | Путь | Описание |
|---|---|---|
| GET | `/` | Веб-интерфейс для классификации текста |
| POST | `/predict` | Классификация одного текста (JSON: `{"text": "..."}`) |
| POST | `/predict/file` | Классификация CSV-файла с колонкой `content` |
| GET | `/features` | Информация о признаках модели |
| GET | `/feature/importance` | Топ-признаки, влияющие на предсказание |

## Ограничения модели

Точность (F1 ≈ 0.69) ограничена самой природой задачи: метка "хороший/плохой" получена искусственным делением по медиане оценки, из-за чего отзывы с близкими оценками (например, 7.1 и 7.3) попадают в разные классы, хотя по смыслу почти неотличимы. Модель также хуже справляется с ироничными и сдержанно-негативными формулировками, характерными для профессиональной музыкальной критики.
