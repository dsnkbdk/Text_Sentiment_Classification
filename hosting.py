import os
import time
import mlflow
import logging
from dotenv import load_dotenv
from mlflow import MlflowClient
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from fastapi import Body, FastAPI, Request, HTTPException
from typing import Literal, Optional, List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Schemas
SentimentLabel = Literal["negative", "neutral", "positive"]

class PredictRequest(BaseModel):
    # Single test
    input_text: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Single crypto news text for sentiment classification"
    )

    # Batch test
    input_texts: Optional[List[str]] = Field(
        default=None,
        description="Batch crypto news texts for sentiment classification"
    )

class PredictItem(BaseModel):
    sentiment: SentimentLabel = Field(
        ...,
        description="Predicted sentiment label"
    )
    probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Probabilities of each class"
    )

class PredictResponse(BaseModel):
    predictions: List[PredictItem] = Field(
        ...,
        description="Predictions for each input text"
    )
    model_uri: str = Field(
        ...,
        description="MLflow model URI used for this prediction (production alias)"
    )
    latency_ms: int = Field(
        ...,
        description="Inference latency (ms)"
    )
    meta: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Model metadata"
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
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{model_uri}': {e}") from e
    
    # Get model metadata
    meta = {
        "registered_model_name": registered_model_name,
        "alias": "production"
    }

    try:
        client = MlflowClient()
        mv = client.get_model_version_by_alias(registered_model_name, "production")
        meta.update({
            "model_version": mv.version,
            "run_id": mv.run_id
        })
    except Exception:
        pass

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
            "model_uri": model_uri,
            "meta": meta
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
    async def predict(
        req: PredictRequest = Body(
            ...,
            openapi_examples={
                "single": {
                    "summary": "Single test",
                    "value": {
                        "input_text": "Bitcoin rallies sharply after ETF approval, igniting strong investor enthusiasm."
                    }
                },
                "batch": {
                    "summary": "Batch test",
                    "value": {
                        "input_texts": [
                            "Ethereum trades sideways as investors await macroeconomic data.",
                            "Crypto prices plunge after exchange hack fears spark panic and heavy regulation."
                        ]
                    }
                }
            }
        )
    ):

        if req.input_text is not None:
            raw_text = [req.input_text]
        elif req.input_texts is not None:
            raw_text = req.input_texts
        else:
            raw_text = []

        input_text: List[str] = []
        for text in raw_text:
            if not isinstance(text, str):
                continue
            # Avoid empty text such as "    "
            string = text.strip()
            if string:
                input_text.append(string)

        if not input_text:
            raise HTTPException(
                status_code=400,
                detail="Field 'input_text' must be a string or 'input_texts' must be a list of strings"
            )
        
        logger.info(f"Predict request received (texts={len(input_text)})")
        
        start = time.perf_counter()

        # MLflow signature expects list/Series
        try:
            preds = model.predict(input_text)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model prediction failed: {e}") from e
        
        # Probabilities
        probas = None
        probas = None

        if hasattr(model, "predict_proba"):
            try:
                probas = model.predict_proba(input_text)
                classes = list(getattr(model, "classes_", []))
            except Exception:
                probas = None
                classes = None
        
        latency_ms = int((time.perf_counter() - start) * 1000)

        outcomes = []
        for i, pred in enumerate(preds):
            item = {
                "sentiment": str(pred),
                "probabilities": None
            }
            if probas is not None and classes:
                item["probabilities"] = {
                    classes[c]: probas[i][c] for c in range(len(classes))
                }
            outcomes.append(item)
        
        logger.info(f"Predict success (items={len(input_text)}), latency_ms={latency_ms}")

        return {
            "predictions": outcomes,
            "model_uri": model_uri,
            "latency_ms": latency_ms,
            "meta": meta
        }

    return app


# uvicorn serve:app --reload
# Swagger: http://localhost:8000/docs
# Health: http://localhost:8000/health
# Status: http://localhost:8000/status

load_dotenv()

app = fast_api(
    mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    registered_model_name="TFIDF_Logistic_Regression")