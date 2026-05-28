from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
import torch
import os

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading Model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

print("Loading RCOEM Dataset...")
dataset_path = "rcoem_dataset.json"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset {dataset_path} not found.")

dataset = load_dataset("json", data_files=dataset_path)
split = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
val_ds = split["test"]

def format_data(example):
    text = f"Instruction:\n{example['instruction']}\nResponse:\n{example['response']}\n"
    return {"text": text}

train_fmt = train_ds.map(format_data)
val_fmt = val_ds.map(format_data)

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tok_train = train_fmt.map(tokenize)
tok_val = val_fmt.map(tokenize)

training_args = TrainingArguments(
    output_dir="./rcoem_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=True,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tok_train,
    eval_dataset=tok_val,
    tokenizer=tokenizer,
    max_seq_length=512
)

print("Starting Fine-Tuning for RCOEM...")
trainer.train()

trainer.model.save_pretrained("./rcoem_model")
tokenizer.save_pretrained("./rcoem_model")
print("RCOEM Model Training Completed successfully!")
