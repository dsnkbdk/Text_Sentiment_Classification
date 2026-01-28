# AI-Driven Video Understanding Application

## Overview

This repository is the **Goldenset Intern-to-Hire Project**, which implements a modular and automated AI pipeline for video understanding using the OpenAI API.  
It integrates speech transcription, object detection, sentiment analysis, and question-answer generation into a single streamlined process.

This Project requires candidates to:

- Use OpenAI API to process a `.mp4` video and output a structured JSON containing:
  - Full Transcription
  - Objects detected in the video
  - Overall mood and sentiment
  - A list of Q&A pairs generated from the video
- Add unit tests
- Add README.md
- Ensure one-click execution in [CodeSandbox.io](https://codesandbox.io/)

## Project Structure

```bash
.
├── .codesandbox/                   # CodeSandbox configuration
├── .devcontainer/                  # Development container configuration
├── .git/                           # Git version control folder
├── .github/                        # GitHub workflow and CI/CD configuration
├── .coveragerc                     # Coverage configuration for pytest-cov
├── .env                            # Environment variables (API key, video path)
├── .gitignore                      # Git ignore rules
├── AI_Intern_Project.mp4           # Input video file for processing
├── main.py                         # Main orchestration logic for running the full pipeline
├── object_detection.py             # Detects objects from video frames
├── question_answer.py              # Generates Q&A pairs from transcript
├── requirements.txt                # Python dependencies list
├── sentiment_analysis.py           # Analyses mood and sentiment from transcription
├── video_transcript.py             # Extracts and transcribes audio
├── pytest.ini                      # Pytest configuration file
├── README.md                       # README documentation
└── tests/                          # Unit tests folder
    ├── conftest.py                 # Pytest shared fixtures and setup
    ├── test_main.py                # Test file for main.py
    ├── test_object_detection.py    # Test file for object_detection.py
    ├── test_question_answer.py     # Test file for question_answer.py
    ├── test_sentiment_analysis.py  # Test file for sentiment_analysis.py
    └── test_video_transcript.py    # Test file for video_transcript.py
```

## Environment Configuration

### 1. API key

After registering an OpenAI account, create an API key by navigating to:

```bash
Settings
↓
API keys
↓
Create new secret key
```

Please note that the Secret Key is displayed only once at the time of creation.  
Make sure to store it safely. If it is lost or exposed, please create a new one immediately.

### 2. CodeSandbox Settings

CodeSandbox is a cloud development platform that provides a user-friendly graphical interface, enabling developers to quickly set up a development environment.  
For this project, please ensure the following configuration:

- **Visibility**

  Set it to **Unlisted (everyone with the link can view)**, which allows reviewers to reproduce and evaluate the work.

- **Project Setup**

  Navigating to:

  ```bash
  Set up my own Development Container
      ├── Python version
      │   └── 3.12-bullseye
      ├── Setup tasks
      │   └── Add command: pip install -r requirements.txt
      └── Environment variables
          ├── OPENAI_API_KEY=sk-proj-xxxxxxxxx
          └── VIDEO_PATH=AI_Intern_Project.mp4
  ```

### 3. Dependencies

| Package                  | Purpose                             |
| ------------------------ | ----------------------------------- |
| `moviepy`                | Extracts audio from video           |
| `openai`                 | Interfaces with OpenAI models       |
| `opencv-python-headless` | Frame sampling for object detection |
| `pytest`, `pytest-cov`   | Unit testing and coverage           |
| `python-dotenv`          | Loads `.env` environment variables  |

### 4. Configuration Files

Ensure the following configuration files exist in the root directory:

- `.coveragerc` — Defines coverage rules for unit testing.
- `.gitignore` — Specifies files and directories to be excluded from version control.
- `pytest.ini` — Contains Pytest configuration for logging, coverage, and test discovery.
- `requirements.txt` — Lists all Python dependencies used in the project.

## Running

### CodeSandbox

This project will be shared via the CodeSandbox link, then follow the instructions:

```bash
Open the link
↓
Fork (Create new fork)
↓
Set your own API key and video path
↓
The sandbox will restart and automatically install requirements.txt
↓
CodeSandbox
↓
Tasks
↓
Start
```

`main.py` will automatically execute, display progress and results (JSON) in the terminal.

### Output Example

You will see a structured JSON printed in the terminal, similar to:

```json
{
    "Transcription": "Cooking the perfect...",

    "Objects": [
        "1. Stove",
        "2. Frying pan",
        "3. Tongs",
        ...
    ],

    "Mode and sentiment": {
        "mode": "instructional",
        "sentiment": "positive",
        "explanation": "The video mode is..."
    },

    "Q&A pairs": [
        {
            "Q": "What is the...",
            "A": "A thick cut..."
        },
        {
            "Q": "How should you...",
            "A": "Season the steak..."
        },
        ...
    ]
}
```

### Customization

You can modify the pipeline by:

- Changing the model name (e.g., `gpt-5`, `gpt-4o`)
- Adjusting frame sampling rate in `object_detection.py`
- Changing or adding custom prompts

## Approaches and Solutions

### Transcription (`video_transcript.py`)

#### 1. Input Format

According to OpenAI's official documentation [Create transcription](https://platform.openai.com/docs/api-reference/audio/createTranscription),
the audio model supports direct transcription of `.mp4` files. Our experiments revealed several drawbacks to uploading video files directly:

- High token consumption
- Heavy network usage
- Long processing time (approximately 3 minutes)

To address this, we used the `moviepy` library to extract the audio track from the video and convert it into an `.mp3` file before sending it to the model.
The results showed that:

- Transcription quality: No noticeable difference
- Token usage: Significantly reduced
- Processing time: Reduced to roughly 10 seconds

This simple pre-processing step significantly improves the efficiency of the transcription workflow without compromising accuracy.

#### 2. Model Selection

The audio model options are `gpt-4o-mini-transcribe`, `gpt-4o-transcribe`, and `whisper-1`. We evaluated the transcription quality of all three models:

- gpt-4o-mini-transcribe

```
After a few minutes, take a peek and flip when it's at least 50% brown. I'm nice and begin basting the steak.
```

- gpt-4o-transcribe

```
After a few minutes, take a peek and flip when it's a deep golden brown. Immediately add a knob of butter, aromatics, and begin basting your steak.
```

- whisper-1

```
After a few minutes, take a peek and flip when it's a deep golden brown. Immediately add a knob of butter and some aromatics and begin basting the steak.
```

The results showed that only the `whisper-1` model produced a fully accurate transcription.
`gpt-4o-mini-transcribe` performed the worst, while `gpt-4o-transcribe` missed some minor details.

Therefore, we ultimately selected `whisper-1` model for transcription.

#### 3. Tuning

Although `whisper-1` performed well, we observed occasional hallucinations during repeated experiments: in some runs,
the model appended some extra sentences at the end of the transcription, such as “Thanks for watching”, which do not exist in the source audio. The possible causes include:

- Background music and environmental noise
- Approximately one second of trailing silence at the end
- Slight accent or pronunciation variations in the speech

To address this, we introduced the following parameters for model tuning:

```python
language="en"
prompt="Transcribe exactly what is spoken. Ignore any background music or noise that may be present."
```

After applying these settings, hallucinations no longer occur, and the model produces accurate and stable transcription.

### Object Detection (`object_detection.py`)

#### 1. Input Format

OpenAI's official documentation does not mention video analysis capabilities. Instead, it supports image analysis [Analyze images](https://platform.openai.com/docs/guides/images-vision?api-mode=responses&format=base64-encoded#analyze-images).

Therefore, we considered sampling the video and converting it into multiple frames (images) as input. OpenAI supports three ways to provide image inputs:

- **URL**
- **Base64 encoded image**
- **file ID**

The **Base64 encoded** method caught our attention. It allows us to convert each image into an encoded string, eliminating the need to store temporary image files. Instead, we simply create a list of strings representing the frames.

We implemented this approach using the `opencv` library and introduced a `sample_rate` parameter to control the sampling frequency. In theory, a higher sampling rate captures more frames, allowing for more object detection.

However, during the experiments, we observed that setting `sample_rate` to 1 (that is, extracting 60 images from the video) triggered an API error. OpenAI limits token consumption to approximately 30,000 per minute, while processing 60 images would exceed 45,000 tokens.

We recommend keeping `sample_rate = 0.5` (samples one frame every two seconds). This captures around 30 images from the video, sufficient for keeping detail while avoiding token overuse.

#### 2. Model Building

We directly followed the official examples to build the image analysis model, keeping `model="gpt-4.1"` unchanged. Because we found that `gpt-4.1` provides a good balance between accuracy and token consumption.

We focused on designing the `input content` to achieve comprehensive object detection and ensure standardised output. The image analysis API is actually implemented through the `Responses` Interface, which is currently the most advanced API. It supports three roles:

- developer — highest priority, typically used to define system-level behaviour or structured output formats
- user — flexible input role, representing user-provided content
- assistant — used for the model’s responses

We utilised the `developer` role to construct prompts for structured output.

For the `user` role, we implemented a `for` loop to sequentially append all sampled images into the `input content` list.

Additionally, we used keywords such as `"distinct"` and `"as possible as"` to encourage the model to detect **as many unique objects as possible** in each frame.

This approach allowed us to keep a consistent structure, maximise detection coverage, and fully leverage the strengths of `gpt-4.1` in visual understanding tasks.

#### 3. JSON Output

The model response supports JSON output, which requires us to enforce a well-defined JSON schema for consistent outputs. In our case, we expected the model to return a list of detected objects:

```json
{
    "Objects": [
        "1. Stove",
        "2. Frying pan",
        "3. Tongs",
        ...
    ]
}
```

To achieve this, we configured the JSON schema as follows:

- The field `objects` is defined with `"type": "array"`
- Each element within the array is defined with `"items": {"type": "string"}`

### Mode and Sentiment (`sentiment_analysis.py`)

#### 1. Input Format

As mentioned earlier, the current model response API does not yet support direct video analysis. Therefore, we used the transcription text as the model input.

#### 2. Model Building

We continued using the `gpt-4.1` model for this task, as it provides a reliable balance between reasoning quality and cost efficiency.

- The **developer** role defines the structured output prompt.
- The **user** role provides the task prompt and includes the transcription as part of the input.

#### 3. JSON Output

The expected output structure is as follows:

```json
{
  "Mode and sentiment": {
    "mode": "instructional",
    "sentiment": "positive",
    "explanation": "The video mode is..."
  }
}
```

To implement this structure, we defined the `"mode"`, `"sentiment"`, and `"explanation"` fields respectively, and defined `"type": "string"`.

### Q&A Pairs (`question_answer.py`)

#### 1. Model Building

Similar to `sentiment_analysis.py`, we continue using the transcription text as the model input.

When constructing the `input content`, we do not manually provide any questions. Instead, we use prompts to guide the model to automatically generate 5 to 10 Q&A pairs based on the given transcription text.

- The **developer** role includes an additional prompt that constrains the Q&A generation behaviour to prevent hallucinations or irrelevant content.
- The **user** role provides a concise instruction to generate Q&A pairs and includes the transcription as part of the input.

#### 2. JSON Output

The expected output structure is as follows:

```json
{
    "Q&A pairs": [
        {
            "Q": "What is the...",
            "A": "A thick cut..."
        },
        {
            "Q": "How should you...",
            "A": "Season the steak..."
        },
        ...
    ]
}
```

- The field `QA_pairs` is defined with `"type": "array"`
- Each element within the array is defined with `"items": {"type": "object"}`
- Each Q and A within the object is defined with `"Q": {"type": "string"}`, `"A": {"type": "string"}`

### Main (`main.py`)

This module functions as the orchestrator of the video understanding pipeline. It coordinates the execution of individual processing components.

#### 1. Environment Loading and Validation

The script loads environment variables using `dotenv`, retrieving `OPENAI_API_KEY` and `VIDEO_PATH`. It validates that:

- The API key is present.
- The video file exists.
- The file type is a supported video format.

Invalid or missing configurations trigger explicit exceptions before the pipeline starts.

#### 2. Pipeline Orchestration

The `openai_pipeline()` function defines a sequential workflow:

- Transcription – Obtains the complete text from the video via `video_transcript()`.
- Object Detection – Performs frame sampling and image-based detection through `object_detection()`.
- Mode and Sentiment – Uses `sentiment_analysis()` to infer the speech mode and tone.
- Q&A Generation – Calls `question_answer()` to generate context-based question–answer pairs.

Each stage logs progress and exceptions using the `logging` library for traceability.

#### 3. Error Management and Logging

All operations are wrapped in structured `try–except` blocks. Errors are logged with timestamps and stack traces to facilitate debugging.

#### 4. Design Considerations

This module focuses on integration and reliability. Each analytical component is isolated in its own module, allowing updates or configuration without modifying `main.py`.

## How much AI is used?

The core functionalities of this project are AI-driven, including **speech-to-text**, **object detection**, **sentiment analysis**, and **Q&A generation**. In contrast, orchestration, I/O, validation, preprocessing, testing, and infrastructure are implemented using conventional code.

**Breakdown by model**

- `Whisper-1` – Used for speech recognition to transcribe the entire video into text.

- `GPT-4.1` – Used for multiple reasoning and generation tasks:
  - Infers and integrates objects detected from video frames
  - Analyses the overall mode and sentiment from the transcript
  - Generates Q&A pairs based on text
  - Produces structured JSON outputs

## Unit Tests

All modules (`main`, `video_transcript`, `object_detection`, `sentiment_analysis`, `question_answer`) include unit tests under `tests/`.

Run tests in the terminal with:

```
pytest
```

Tests are designed to:

- Verify core functionality — confirm that each module returns the correct output under normal conditions.
- Validate error handling — ensure the system raises or catches exceptions properly for missing files, invalid input, or failed API calls.
- Mock external dependencies — simulate OpenAI API, cv2, and file operations to test logic without real network or file access.
- Check boundary conditions — test edge cases such as unsupported file types, invalid frame rates, or extreme sampling rates.
- Ensure pipeline integration — confirm the full workflow (from environment setup to JSON output) runs smoothly and handles failures gracefully.

## Limitations and Future Work

**Where are the places you could do better?**

Due to time constraints, this project only tested and evaluated all available models within the `video_transcript` module.

However, in all other modules, the `gpt-4.1` model was used. From an engineering perspective, it is necessary to evaluate alternative models to pursue better performance and lower token costs.

The `video_transcript` module currently only supports extracting the audio track from a video file and sending it to the API. With minor modifications, the module could automatically identify video or audio files. For audio files, no preprocessing is required, and they can be sent directly to the API for transcription.

The `object_detection` module currently uses a sampling rate of 0.5 (one frame every two seconds) to comply with OpenAI's per-minute token limit. A potential improvement would be to calculate the token cost per image and send images to the API in batches, ensuring that the token cost per batch does not exceed the limit. Each batch can be sent at a one-minute interval to avoid triggering errors.
