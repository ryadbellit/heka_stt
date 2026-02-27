from util import LANGUAGE, MODEL_ID
from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor

model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID)
processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE)

