import os
import mlflow
import logging
import pandas as pd
from dotenv import load_dotenv
from typing import Literal, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Schemas
SentimentLabel = Literal["negative", "neutral", "positive"]

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Crypto news for sentiment classification (title + text)",
        examples=["Bitcoin price surges after ETF approval"]
    )

class PredictResponse(BaseModel):
    sentiment: SentimentLabel = Field(
        ...,
        description="Predicted sentiment label"
    )
    model_uri: str = Field(
        ...,
        description="MLflow model URI used for this prediction (production alias)"
    )

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


def fast_api(registered_model_name: str) -> FastAPI:
    
    # Load URI
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_uri = f"models:/{registered_model_name}@production"

    # Load prod model
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{model_uri}': {e}")

    app = FastAPI(
        title="Text Sentiment Classification API",
        version="1.0.0",
        openapi_tags=[
            {"name": "Health", "description": "Service health and readiness checks"},
            {"name": "Prediction", "description": "Model inference endpoints"}
        ]
    )

    # Health
    @app.get("/health", tags=["Health"])
    def health():
        return {
            "status": "OK",
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "model_uri": model_uri
        }

    # Prediction
    @app.post(
        "/predict",
        tags=["Prediction"],
        response_model=PredictResponse,
        responses={
            400: {"model": ErrorResponse, "description": "Bad request (missing or invalid input)"},
            503: {"model": ErrorResponse, "description": "Model not available (MLflow/model loading error)"}
        }
    )
    async def predict(req: PredictRequest):
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="Field 'text' is required")
        
        # MLflow signature expects a DataFrame with column 'title_text'
        try:
            pred = model.predict(pd.DataFrame({"title_text": [req.text]}))
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model prediction failed: {e}")

        return {"model_uri": model_uri, "sentiment": str(pred[0])}

    return app


# uvicorn serve:app --reload
# Swagger: http://localhost:8000/docs
# Health: http://localhost:8000/health

load_dotenv()
app = fast_api(registered_model_name="TF-IDF Logistic Regression")