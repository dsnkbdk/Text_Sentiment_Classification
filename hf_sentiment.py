import torch
import logging
from transformers import pipeline
from typing import List, Dict, Any, Optional


from transformers import AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)

def sentiment_pipeline(model_name: str, device: int | None = None) -> None:
    
    # Select CPU or GPU
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    clf = pipeline(
        task="sentiment-analysis",
        model=model_name,
        revision="refs/pr/43",
        tokenizer=model_name,
        use_safetensors=True,
        device=device
    )

    logger.info(f"HF pipeline loaded: model={model_name}, device={device}")
    
    return clf

def predict_sentiment(
    clf,
    texts: list,
    batch_size: int,
    max_length: int
) -> dict:
    
    outputs = clf(
        inputs=texts,
        batch_size=batch_size,
        max_length=max_length,
        truncation=True,
        top_k=None
    )

    pred_labels = []
    pred_scores = []
    full_scores = []

    for item in outputs:
        scores = []
        
        for d in item:
            label = str(d["label"]).lower()
            scores.append({"label": label, "score": d["score"]})
        
        full_scores.append(scores)

        best = max(scores, key=lambda x: x["score"])
        logger.info(f"bestbestbest{best}")
        
        # If label mapping is required
        if best["label"] in {"label_0", "0"}:
            pred = "negative"
        elif best["label"] in {"label_1", "1"}:
            pred = "neutral"
        elif best["label"] in {"label_2", "2"}:
            pred = "positive"
        else:
            # If the labels are original
            pred = best["label"].replace("label_", "").strip()

        pred_labels.append(pred)
        pred_scores.append(best["score"])

    return {
        "pred_labels": pred_labels,
        "pred_scores": pred_scores,
        "full_scores": full_scores
    }
