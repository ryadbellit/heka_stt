from micro import record_dataset

with open("phrases.txt", "r", encoding="utf-8") as f:
    phrases = [line.strip() for line in f if line.strip()]

record_dataset(
    prompts=phrases,
    duration=10,
    frequency=16000
)