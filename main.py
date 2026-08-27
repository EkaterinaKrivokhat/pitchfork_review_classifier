"""
FastAPI-сервис классификации музыкальных рецензий Pitchfork
(хороший отзыв / плохой отзыв) на основе обученной модели
Logistic Regression + TF-IDF + инженерные признаки.

Запуск:
    python -m uvicorn main:app --reload

После запуска открыть в браузере: http://127.0.0.1:8000
"""
import io
import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from scipy.sparse import csr_matrix, hstack

from feature_engineering import ENGINEERED_COLS, ENGINEERED_COLS_DESCRIPTIONS, clean_text, extract_features

app = FastAPI(
    title="Pitchfork Review Classifier",
    description="Классификация музыкальных рецензий: хороший отзыв / плохой отзыв",
    version="1.0",
)

# загрузка обученной модели и препроцессоров один раз при старте сервиса
model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("feature_scaler.pkl")

LABEL_NAMES = {0: "Плохой отзыв", 1: "Хороший отзыв"}

from fastapi.responses import HTMLResponse, JSONResponse

@app.get("/", response_class=HTMLResponse)
def index():
    # отдаёт минимальный HTML-интерфейс из static/index.html
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


class PredictRequest(BaseModel):
    text: str


def build_features(text: str):
    text = clean_text(text)
    feats = extract_features(text)
    X_tfidf = vectorizer.transform([text])
    X_eng = scaler.transform(pd.DataFrame([feats])[ENGINEERED_COLS])
    X = hstack([X_tfidf, csr_matrix(X_eng)]).tocsr()
    return X, feats


def predict_one(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="Текст отзыва пустой")

    X, feats = build_features(text)
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    return {
        "label": pred,
        "label_name": LABEL_NAMES[pred],
        "confidence": round(float(proba[pred]), 3),
        "probabilities": {
            "Плохой отзыв": round(float(proba[0]), 3),
            "Хороший отзыв": round(float(proba[1]), 3),
        },
        "features": {k: round(v, 3) if isinstance(v, float) else v for k, v in feats.items()},
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """Классифицирует один текст отзыва."""
    return predict_one(req.text)

@app.get("/features")
def features():
    """Информация о признаках, которые использует модель."""
    return {
        "n_tfidf_features": len(vectorizer.get_feature_names_out()),
        "engineered_features": ENGINEERED_COLS,
        "engineered_features_description": ENGINEERED_COLS_DESCRIPTIONS,
    }


@app.get("/feature/importance")
def feature_importance(top_n: int = 15):
    """Топ-N признаков, сильнее всего влияющих на предсказание 'хороший'/'плохой'."""
    coefs = model.coef_[0]
    all_names = list(vectorizer.get_feature_names_out()) + ENGINEERED_COLS
    pairs_sorted = sorted(zip(all_names, coefs), key=lambda x: x[1])

    top_bad = pairs_sorted[:top_n]
    top_good = pairs_sorted[::-1][:top_n]

    return {
        "top_features_for_good_review": [
            {"feature": f, "weight": round(float(w), 4)} for f, w in top_good
        ],
        "top_features_for_bad_review": [
            {"feature": f, "weight": round(float(w), 4)} for f, w in top_bad
        ],
    }

@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    """Классифицирует все строки CSV-файла с колонкой 'content' (или 'text')."""
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось прочитать CSV-файл")

    text_col = next((c for c in ("content", "text", "review") if c in df.columns), None)
    if text_col is None:
        raise HTTPException(
            status_code=400,
            detail="В файле должна быть колонка 'content' (или 'text') с текстом отзыва",
        )

    results = [predict_one(str(t)) for t in df[text_col]]

    df_out = df.copy()
    df_out["predicted_label"] = [r["label"] for r in results]
    df_out["predicted_label_name"] = [r["label_name"] for r in results]
    df_out["confidence"] = [r["confidence"] for r in results]

    return JSONResponse(
        content={"n_rows": len(df_out), "results": df_out.to_dict(orient="records")}
    )