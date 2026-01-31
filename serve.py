import os
import time
import mlflow
import logging
import pandas as pd
from dotenv import load_dotenv
from typing import Literal, Optional
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, HTTPException

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
        description="Crypto news text for sentiment classification (title + text)",
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
    latency_ms: int = Field(
        ...,
        description="Inference latency (ms)"
    )

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


def fast_api(
        *,
        mlflow_tracking_uri: str,
        registered_model_name: str
    ) -> FastAPI:
    
    # Configure mlflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    model_uri = f"models:/{registered_model_name}@production"

    # Load prod model
    logger.info(f"Loading prod model from {model_uri}")
    
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{model_uri}': {e}") from e

    app = FastAPI(
        title="Text Sentiment Classification API",
        version="1.0.0",
        openapi_tags=[
            {
                "name": "Health",
                "description": "Service health and readiness checks"
            },
            {
                "name": "Prediction",
                "description": "Model inference endpoints"
            }
        ]
    )

    # Consistent error response schema
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.warning(f"HTTPException {exc.status_code} on {request.url.path}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "request_failed",
                "detail": str(exc.detail)
            }
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception on {request.url.path}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": str(exc)
            }
        )

    # Health
    @app.get("/health", tags=["Health"])
    def health():
        return {
            "status": "OK"
        }
    
    @app.get("/status", tags=["Health"])
    def status():
        return {
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
            500: {"model": ErrorResponse, "description": "Internal error"},
            503: {"model": ErrorResponse, "description": "Model not available (MLflow loading error)"}
        }
    )
    async def predict(req: PredictRequest):

        # Avoid empty text such as "    "
        text = req.text.strip()

        if not text:
            raise HTTPException(status_code=400, detail="Field 'text' is required")
        
        logger.info(f"Predict request received (chars={len(text)})")
        
        start = time.perf_counter()

        # MLflow signature expects a DataFrame with column 'title_text'
        try:
            pred = model.predict(pd.DataFrame({"title_text": [text]}))
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model prediction failed: {e}") from e
        
        latency_ms = int((time.perf_counter() - start) * 1000)
        
        logger.info(f"Predict success, sentiment is {str(pred[0])}, latency_ms is {latency_ms}")

        return {
            "sentiment": str(pred[0]),
            "model_uri": model_uri,
            "latency_ms": latency_ms
        }

    return app


# uvicorn serve:app --reload
# Swagger: http://localhost:8000/docs
# Health: http://localhost:8000/health
# Status: http://localhost:8000/status

load_dotenv()

app = fast_api(
    mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    registered_model_name="TF-IDF Logistic Regression")