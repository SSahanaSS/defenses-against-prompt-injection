from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import uuid
from PIL import Image
import pytesseract
import os
import tempfile
import whisper
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re
import torch.nn.functional as F
from enum import Enum

# ---------------------------------------------------------
# DEVICE SETUP
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# LOAD WHISPER MODEL (FOR AUDIO)
# ---------------------------------------------------------
whisper_model = whisper.load_model("base")

# ---------------------------------------------------------
# LOAD DISTILBERT MODEL + TOKENIZER
# ---------------------------------------------------------
MODEL_PATH = r"D:\downloads\3credproj\model"  # adjust your path
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
model.eval()

# ---------------------------------------------------------
# ENUM FOR RISK LEVEL
# ---------------------------------------------------------
class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# ---------------------------------------------------------
# EMOJI REMOVAL
# ---------------------------------------------------------
EMOJI_PATTERN = re.compile("[\U0001F600-\U0001F64F"
                           "\U0001F300-\U0001F5FF"
                           "\U0001F680-\U0001F6FF"
                           "\U0001F1E0-\U0001F1FF"
                           "\U00002500-\U00002BEF"
                           "\U00002702-\U000027B0"
                           "\U000024C2-\U0001F251"
                           "\U0001f926-\U0001f937"
                           "\U00010000-\U0010ffff"
                           "\u2640-\u2642"
                           "\u2600-\u2B55"
                           "\u200d"
                           "\u23cf"
                           "\u23e9"
                           "\u231a"
                           "\ufe0f"
                           "\u3030]+")
def remove_emojis(text: str) -> str:
    return EMOJI_PATTERN.sub(r'', text or "")

# ---------------------------------------------------------
# HEURISTIC LAYER
# ---------------------------------------------------------
class SimplifiedHeuristicFilter:
    def __init__(self):
        self.injection_patterns = {
            "injection_markers": [
                r"###?\s*[^#\n]+###?",
                r"---+\s*[^-\n]+---+",
                r"\[INST\]|\[/INST\]",
                r"<\|.*?\|>",
                r"```\s*(prompt|instruction|system)",
            ],
            "obfuscation": [
                r"[a-zA-Z0-9+/]{20,}={0,2}",
                r"\\u[0-9a-fA-F]{4}",
                r"&#x?[0-9a-fA-F]+;",
                r"%[0-9a-fA-F]{2}",
                r"[^\x00-\x7F]+.*[^\x00-\x7F]+",
            ]
        }

    def scan_prompt(self, text: str):
        if not text or not isinstance(text, str):
            return {"passed": True, "risk": RiskLevel.LOW.value, "score": 0.0, "matched_patterns": {}}

        text_lower = text.lower()
        matched_patterns = {}
        total_score = 0.0

        for category, patterns in self.injection_patterns.items():
            matches = []
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                    matches.append(pattern)
                    total_score += 1.0
            if matches:
                matched_patterns[category] = matches

        if total_score >= 2.0:
            risk, passed = RiskLevel.HIGH, False
        elif total_score >= 1.0:
            risk, passed = RiskLevel.MEDIUM, False
        else:
            risk, passed = RiskLevel.LOW, True

        return {"passed": passed, "risk": risk.value, "score": round(total_score, 2), "matched_patterns": matched_patterns}

# ---------------------------------------------------------
# DISTILBERT PREDICTION LAYER
# ---------------------------------------------------------
def model_predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return pred.item(), round(conf.item() * 100, 2)

label_names = {0: "Safe", 1: "Unsafe"}

# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------
app = FastAPI(title="Prompt Injection Detector API (2-Layer)")

heuristic_filter = SimplifiedHeuristicFilter()

@app.post("/ingest")
async def ingest(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    metadata: Optional[str] = Form(None)
):
    request_id = str(uuid.uuid4())
    extracted_text = text

    # --- IMAGE OCR ---
    # --- IMAGE OCR ---
    if image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
            temp_img.write(await image.read())
            temp_img_path = temp_img.name
        extracted_text = pytesseract.image_to_string(Image.open(temp_img_path))
        os.remove(temp_img_path)
        print(f"🖼️ [OCR] Extracted text from image:\n{extracted_text}\n")  # <-- add this

    # --- AUDIO TRANSCRIPTION ---
    elif audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(await audio.read())
            temp_audio_path = temp_audio.name
        result = whisper_model.transcribe(temp_audio_path)
        extracted_text = result["text"].strip()
        os.remove(temp_audio_path)
        print(f"🎤 [Whisper] Transcribed text from audio:\n{extracted_text}\n")  # <-- add this


    # --- CLEAN EMOJIS ---
    cleaned_text = remove_emojis(extracted_text or "")

    # --- LAYER 1: HEURISTIC SCAN ---
    heuristic_result = heuristic_filter.scan_prompt(cleaned_text)

    # --- LAYER 2: DISTILBERT ---
    if heuristic_result['passed']:
        label, confidence = model_predict(cleaned_text)
        model_result = {
            "prediction": label_names[label],
            "confidence": confidence
        }
    else:
        model_result = {
            "prediction": "Unsafe (heuristic)",
            "confidence": None
        }

    # --- BUILD RESPONSE ---
    response = {
        "request_id": request_id,
        "metadata": metadata,
        "final_text": cleaned_text,
        "heuristic": heuristic_result,
        "model": model_result,
        "image_name": image.filename if image else None,
        "audio_name": audio.filename if audio else None
    }

    return response
