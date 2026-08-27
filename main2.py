# ===================================================================
# ✅ FULL 2-LAYER SECURITY PIPELINE (FASTAPI VERSION)
# Combines: Heuristic Layer + DistilBERT + Optional Gemini Reply
# ===================================================================
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
import google.generativeai as genai
from dotenv import load_dotenv
import time

# ---------------------------------------------------------
# LOAD API KEY USING genai.configure
# ---------------------------------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
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
    "\u3030"
    "]+"
)
def remove_emojis(text: str) -> str:
    return EMOJI_PATTERN.sub(r'', text or "")

# ---------------------------------------------------------
# HEURISTIC LAYER (COMPLETE VERSION)
# ---------------------------------------------------------
class SimplifiedHeuristicFilter:
    def __init__(self):
        # quick safe: clearly benign phrasing
        self.quick_safe_patterns = [
            r"\b\d{1,2}:\d{2}\b",
            r"shopping list:.*",
            r"reminder:.*",
            r"note:.*",
            r"meeting at.*",
            r"time:.; location:.",
            r"chapter \d+",
            r"backup your project.*",
        ]

        # whitelist for structured snippets
        self.whitelist_patterns = [
            r"(x\s*=\s*\d+; y\s*=\s*x\s*[\\+\-\/]\s\d+)",
            r"def\s+\w+\(\):",
            r"SQL:\s*SELECT\s+.*;",
            r"YAML:\s*\w+:\s*.*",
        ]

        # suspicious keyword patterns
        self.patterns = [
            r"send\s+money", r"transfer\s+\d+", r"wire\s+funds", r"bank\s+account",
            r"credit\s+card", r"otp", r"https?://\S+", r"bit\.ly/\S+",
            r"click\s+this\s+link", r"free\s+reward", r"urgent",
            r"immediately", r"before\s+midnight", r"else\s+your\s+account\s+will\s+be\s+locked",
            r"verify\s+your\s+account", r"one[-\s]?time\s+password", r"claim\s+your\s+prize",
            r"payment\s+required", r"wire\s+the\s+money", r"send\s+otp"
        ]

        # injection & obfuscation patterns
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
            return {"passed": True, "risk": RiskLevel.LOW.value, "score": 0.0,
                    "matched_patterns": {}, "quick_safe": False}

        text_lower = text.lower()
        matched_patterns = {}
        total_score = 0.0
        quick_safe = False

        # quick safe
        for pattern in self.quick_safe_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                quick_safe = True
                break

        # whitelist
        for pattern in self.whitelist_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                quick_safe = True

        # suspicious keywords
        for pattern in self.patterns:
            if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                matched_patterns.setdefault("suspicious", []).append(pattern)
                total_score += 1.0

        # injection & obfuscation
        for category, patterns in self.injection_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                    matched_patterns.setdefault(category, []).append(pattern)
                    total_score += 1.0

        # risk
        if total_score >= 2.0:
            risk = RiskLevel.HIGH
        elif total_score >= 1.0:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.LOW

        passed = (risk == RiskLevel.LOW)

        return {"passed": passed, "risk": risk.value, "score": round(total_score, 2),
                "matched_patterns": matched_patterns, "quick_safe": quick_safe}

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
# GEMINI CALL USING SDK
# ---------------------------------------------------------
def send_to_gemini(text: str):
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(text)
        return {"gemini_reply": response.text.strip()}
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------
app = FastAPI(title="Prompt Injection Detector API (Full Version)")

heuristic_filter = SimplifiedHeuristicFilter()

@app.post("/ingest")
async def ingest(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    metadata: Optional[str] = Form(None)
):
    start_total = time.time()
    request_id = str(uuid.uuid4())
    extracted_text = text

    # --- IMAGE OCR ---
    if image:
        t1 = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
            temp_img.write(await image.read())
            temp_img_path = temp_img.name
        extracted_text = pytesseract.image_to_string(Image.open(temp_img_path))
        os.remove(temp_img_path)
        ocr_time = round(time.time() - t1, 4)
    else:
        ocr_time = 0.0

    # --- AUDIO TRANSCRIPTION ---
    if audio:
        t1 = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(await audio.read())
            temp_audio_path = temp_audio.name
        result = whisper_model.transcribe(temp_audio_path)
        extracted_text = result["text"].strip()
        os.remove(temp_audio_path)
        asr_time = round(time.time() - t1, 4)
    else:
        asr_time = 0.0

    # --- CLEAN EMOJIS ---
    cleaned_text = remove_emojis(extracted_text or "")

    # --- LAYER 1: HEURISTIC SCAN ---
    t1 = time.time()
    heuristic_result = heuristic_filter.scan_prompt(cleaned_text)
    heuristic_time = round(time.time() - t1, 4)

    # --- LAYER 2: DISTILBERT + GEMINI ---
    safe_to_send = heuristic_result['passed']
    model_result = None
    gemini_response = None
    model_time = 0.0
    gemini_time = 0.0

    if safe_to_send:
        t1 = time.time()
        label, confidence = model_predict(cleaned_text)
        model_time = round(time.time() - t1, 4)
        model_result = {"prediction": label_names[label], "confidence": confidence, "time_sec": model_time}

        if label_names[label] == "Safe":
            t3 = time.time()
            gemini_response = send_to_gemini(cleaned_text)
            gemini_time = round(time.time() - t3, 4)
        else:
            safe_to_send = False
            gemini_response = {"error": "DistilBERT flagged as unsafe, not sent."}
    else:
        gemini_response = {"error": "Heuristic scan flagged as unsafe, not sent."}

    total_time = round(time.time() - start_total, 4)

    # --- RESPONSE ---
    return {
        "request_id": request_id,
        "metadata": metadata,
        "final_text": cleaned_text,
        "heuristic": {**heuristic_result, "time_sec": heuristic_time},
        "model": model_result,
        "ocr_time_sec": ocr_time,
        "asr_time_sec": asr_time,
        "gemini_time_sec": gemini_time,
        "total_time_sec": total_time,
        "safe_to_send": safe_to_send,
        "gemini_response": gemini_response,
        "image_name": image.filename if image else None,
        "audio_name": audio.filename if audio else None
    }
