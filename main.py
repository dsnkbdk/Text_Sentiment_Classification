import os
import json
import logging
import mimetypes
from openai import OpenAI
from dotenv import load_dotenv
from video_transcript import video_transcript
from object_detection import object_detection
from sentiment_analysis import sentiment_analysis
from question_answer import question_answer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def load_env() -> tuple[str, str]:
    """
    This function loads the `.env` file from the current working directory (if present),
    retrieves the OpenAI API key and video file path from the environment, and performs
    validation checks.
    
    Returns:
        tuple[str, str]: A tuple containing:
            - api_key (str): The OpenAI API key for authentication.
            - video_path (str): The path of the video file to be processed.

    Raises:
        RuntimeError: If `OPENAI_API_KEY` or `VIDEO_PATH` is missing.
        FileNotFoundError: If the specified video file does not exist.
        ValueError: If the file type is unsupported.
    """

    # Load .env
    load_dotenv()

    # Load and validate api key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Check `.env` file or environment variables")

    # Load and validate video path
    video_path = os.getenv("VIDEO_PATH")
    if not video_path:
        raise RuntimeError("Missing VIDEO_PATH. Check `.env` file or environment variables")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Check file type
    mime_type, _ = mimetypes.guess_type(video_path)
    if not mime_type or not mime_type.startswith("video/"):
        raise ValueError(f"Unsupported file type: {mime_type}")
    
    return api_key, video_path


def openai_pipeline(api_key: str, video_path: str) -> dict:
    """
    This function orchestrates the complete video parsing pipeline.
    Each stage is executed sequentially and logged for traceability.
    
    Args:
        api_key (str): The OpenAI API key for authentication.
        video_path (str): The path of the video file to be processed.
    
    Returns:
        dict: A dictionary with the following structure:
            {
                "Transcription": <str>,    # Video's complete Transcription
                "Objects": <list[str]>,    # As many Objects detected in video
                "Mode and sentiment": {    # The over all mode and sentiment of the video
                    "mode": <str>,
                    "sentiment": <str>,
                    "explanation": <str>
                },
                "Q&A pairs": [             # Convert video's transcript to list of QA pairs about the video
                    {"Q": <str>, "A": <str>},
                    ...
                ]
            }
    Raises:
        Exception: Propagates any unexpected error that occurs during execution.
    """
    
    # Initialise the client
    client = OpenAI(api_key=api_key)

    try:
        # Get the complete transcription
        transcription = video_transcript(client=client, video_path=video_path, model="whisper-1")
        logger.info("Transcription is complete")

        # Detect objects in the video
        objects = object_detection(client=client, video_path=video_path, model="gpt-4.1", sample_rate=0.5)
        logger.info("Object detection is complete")

        # Analyse the mode and sentiment of the video
        mode_sentiment = sentiment_analysis(client=client, transcription=transcription, model="gpt-4.1")
        logger.info("Mode and sentiment analysis are complete")

        # Generate Q&A pairs
        qa_pairs = question_answer(client=client, transcription=transcription, model="gpt-4.1")
        logger.info("Q&A pairs generation is complete")

    except Exception:
        logger.exception("Unexpected error occurred while parsing video")
        raise

    try:
        # Merge and format output
        merge_output = {
            "Transcription": transcription,
            "Objects": json.loads(objects).get("objects", []),
            "Mode and sentiment": json.loads(mode_sentiment),
            "Q&A pairs": json.loads(qa_pairs).get("QA_pairs", [])
        }

    except Exception:
        logger.exception("Unexpected error occurred while merging and formatting output")
        raise

    return merge_output


def main():
    """
    Main entry point for execution.
    The function serves as the top-level orchestration layer,
    all lower-level exceptions are propagated upward and logged here.
    
    Logging:
        - INFO: Prints the final formatted JSON output.
        - EXCEPTION: Captures full traceback information if a fatal error occurs.
    
    Raises:
        Exception: Any unexpected error that occurs during execution.
    
    Example:
        >>> if __name__ == "__main__":
        ...     main()
    """
    
    try:
        api_key, video_path = load_env()
        merge_output = openai_pipeline(api_key, video_path)
        
        # Format JSON output
        json_output = json.dumps(merge_output, indent=4, ensure_ascii=False).replace(',\n    "', ',\n\n    "')
        logger.info(f"\n{json_output}")

    except Exception:
        logger.exception("Fatal Error: Execution terminated unexpectedly.")




if __name__ == "__main__":
    main()