import torch
import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

# Open-source LLM (Encoder-only)
def cardiffnlp_roberta(
        *,
        model: str,
        device: int | None = None
    ) -> None:
    
    # Select CPU or GPU
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    pipe = pipeline(
        task="sentiment-analysis",
        model=model,
        tokenizer=model,
        revision="refs/pr/43",
        device=device,
        use_safetensors=True
    )

    logger.info(f"Loading Hugging Face pipeline: model={model}, device={device}")

    return pipe

