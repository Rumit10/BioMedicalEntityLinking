from transformers import AutoTokenizer
import json
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-3b")
input_text = []
output_text = []
max_len = 0
total_len = 0
count = 0
high_count = 0
with open("dataset/processed_data/train_dev_corpus.jsonl") as f:
    for line in f:
        count += 1
        data = json.loads(line)
        input_encoded = tokenizer(data["input"], return_tensors="pt")
        length = len(input_encoded["input_ids"][0])
        max_len = max(max_len, length)
        total_len += length
        if length > 512:
            high_count += 1

print(max_len, total_len/count, high_count)
        