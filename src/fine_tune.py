"""
Fine-tuning Whisper with LoRA using PEFT
Expects: dataset_audio/sentence_X.wav files + dataset_audio/transcriptions.txt
         (one transcription per line, matching sentence_0.wav, sentence_1.wav, ...)
"""

import os
import re
import torch
from dataclasses import dataclass
from typing import Any, Dict, List

import soundfile as sf
from torch.utils.data import Dataset

from transformers import (
    AutoModelForSpeechSeq2Seq,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

# ── Config ────────────────────────────────────────────────────────────────────

LANGUAGE        = "english"
MODEL_ID        = "openai/whisper-medium"
AUDIO_DIR       = "dataset_audio"
TRANSCRIPTS_FILE = os.path.join(AUDIO_DIR, "transcriptions.txt")
OUTPUT_DIR      = "whisper_lora_finetuned"
SAMPLING_RATE   = 16_000

# Training hyper-parameters
TRAIN_SPLIT     = 0.9
BATCH_SIZE      = 4
GRAD_ACCUM      = 1
LEARNING_RATE   = 1e-4
NUM_EPOCHS      = 3
WARMUP_STEPS    = 50
FP16            = torch.cuda.is_available()

LORA_R          = 8
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# ── Dataset ───────────────────────────────────────────────────────────────────

def discover_files(audio_dir: str):
    """Return (wav_path, index) pairs sorted by index."""
    pattern = re.compile(r"sentence_(\d+)\.wav$", re.IGNORECASE)
    entries = []
    for fname in os.listdir(audio_dir):
        m = pattern.match(fname)
        if m:
            entries.append((os.path.join(audio_dir, fname), int(m.group(1))))
    entries.sort(key=lambda x: x[1])
    return entries


def load_transcriptions(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


class WhisperAudioDataset(Dataset):
    def __init__(self, audio_paths: List[str], transcriptions: List[str], processor):
        assert len(audio_paths) == len(transcriptions), (
            f"Mismatch: {len(audio_paths)} audio files vs "
            f"{len(transcriptions)} transcriptions"
        )
        self.audio_paths    = audio_paths
        self.transcriptions = transcriptions
        self.processor      = processor

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio, sr = sf.read(self.audio_paths[idx])
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        input_features = self.processor.feature_extractor(
            audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
        ).input_features[0]

        labels = self.processor.tokenizer(
            self.transcriptions[idx], return_tensors="pt"
        ).input_ids[0]

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        return {"input_features": input_features.to(dtype), "labels": labels}


# ── Data collator ─────────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = torch.stack([f["input_features"] for f in features])

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt", padding=True
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        return {"input_features": input_features.to(dtype), "labels": labels}


# ── Metric ────────────────────────────────────────────────────────────────────

wer_metric = evaluate.load("wer")

def compute_metrics(pred, processor):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading model and processor…")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)

    model.config.use_cache = False
    model.generation_config.language = LANGUAGE
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    print("Applying LoRA…")
    lora_config = LoraConfig(
        r               = LORA_R,
        lora_alpha      = LORA_ALPHA,
        lora_dropout    = LORA_DROPOUT,
        target_modules  = LORA_TARGET_MODULES,
        bias            = "none",
        task_type       = TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Discovering audio files…")
    entries = discover_files(AUDIO_DIR)
    if not entries:
        raise FileNotFoundError(f"No sentence_X.wav files found in '{AUDIO_DIR}'")

    transcriptions = load_transcriptions(TRANSCRIPTS_FILE)
    audio_paths = [e[0] for e in entries]

    split = int(len(audio_paths) * TRAIN_SPLIT)
    train_dataset = WhisperAudioDataset(audio_paths[:split],    transcriptions[:split],  processor)
    eval_dataset  = WhisperAudioDataset(audio_paths[split:],    transcriptions[split:],  processor)
    print(f"Train: {len(train_dataset)} samples | Eval: {len(eval_dataset)} samples")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = OUTPUT_DIR,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LEARNING_RATE,
        num_train_epochs            = NUM_EPOCHS,
        warmup_steps                = WARMUP_STEPS,
        fp16                        = False,
        bf16                        = torch.cuda.is_available(),
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        logging_steps               = 10,
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,
        predict_with_generate       = True,
        generation_max_length       = 225,
        report_to                   = "none",
        dataloader_pin_memory       = torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = eval_dataset,
        data_collator   = data_collator,
        compute_metrics = lambda pred: compute_metrics(pred, processor),
        processing_class = processor.feature_extractor,
    )

    print("Starting training…")
    trainer.train()

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Peak VRAM used:      {peak:.2f} GB")
        print(f"Total VRAM reserved: {reserved:.2f} GB")

    print(f"Saving LoRA adapter to '{OUTPUT_DIR}'…")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Done.")


# ── Inference helper (load & run after training) ──────────────────────────────

def transcribe(wav_path: str, adapter_dir: str = OUTPUT_DIR):
    """Quick inference with the fine-tuned LoRA adapter."""
    from peft import PeftModel

    processor = WhisperProcessor.from_pretrained(adapter_dir, language=LANGUAGE)
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if FP16 else torch.float32
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    inputs = processor.feature_extractor(
        audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(inputs)

    return processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]


if __name__ == "__main__":
    main()