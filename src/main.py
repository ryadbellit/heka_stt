from record_audio import record_session

import os
import torch
from datasets import Dataset, Audio, DatasetDict
from transformers import (
    WhisperFeatureExtractor, 
    WhisperTokenizer, 
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# --- 1. CONFIGURATION ---
MODEL_NAME = "openai/whisper-medium"
AUDIO_DIR = "dataset_audio"
LANGUAGE = "English"
TASK = "transcribe"

# --- 2. DYNAMIC DATASET CREATION ---
# Assuming you have a list of transcriptions corresponding to 001-060
# Replace the strings below with your actual text data
transcriptions = [
    "The first sentence transcript goes here.",
    "The second sentence transcript goes here.",
    # ... add all 60 sentences here ...
]

# Quick check: ensure you have 60 transcriptions
if len(transcriptions) < 60:
    print(f"Warning: You only provided {len(transcriptions)} transcriptions for 60 files.")

data_list = []
for i in range(1, 61):
    file_name = f"sentence_{str(i).zfill(3)}.wav"
    file_path = os.path.join(AUDIO_DIR, file_name)
    if os.path.exists(file_path):
        data_list.append({"audio": file_path, "sentence": transcriptions[i-1]})

raw_dataset = Dataset.from_list(data_list)
raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=16000))
ds = raw_dataset.train_test_split(test_size=0.1) # 54 train, 6 eval

# --- 3. PROCESSORS & TOKENIZERS ---
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
feature_extractor = processor.feature_extractor
tokenizer = processor.tokenizer

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

ds = ds.map(prepare_dataset, remove_columns=ds.column_names["train"], num_proc=1)

# --- 4. DATA COLLATOR ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# --- 5. LOAD MODEL WITH PEFT (LoRA) ---
# Using 8-bit quantization to keep VRAM low
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME, 
    quantization_config=bnb_config, 
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=32, 
    lora_alpha=64, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05, 
    bias="none"
)

model = get_peft_model(model, lora_config)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# --- 6. TRAINING ARGUMENTS ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-medium-lora-custom",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=10,
    max_steps=200, # Small dataset needs fewer steps
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=50,
    eval_steps=50,
    logging_steps=10,
    report_to=["none"], # Change to "tensorboard" if desired
    remove_unused_columns=False,
    label_names=["labels"],
)

# --- 7. START TRAINING ---
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

trainer.train()

# Save the adapter
model.save_pretrained("./whisper-medium-lora-final")