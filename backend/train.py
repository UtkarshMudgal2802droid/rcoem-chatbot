from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model
)
import torch
import os

# -----------------------------------
# MODEL NAME
# -----------------------------------
model_name = "mistralai/Mistral-7B-v0.1"

# -----------------------------------
# QUANTIZATION CONFIG
# -----------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# -----------------------------------
# LOAD TOKENIZER
# -----------------------------------
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------------
# LOAD MODEL
# -----------------------------------
print("Loading Model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# -----------------------------------
# LORA CONFIG
# -----------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# -----------------------------------
# APPLY PEFT
# -----------------------------------
model = get_peft_model(model, lora_config)

# -----------------------------------
# LOAD DATASET
# -----------------------------------
print("Loading RCOEM Dataset...")
dataset_path = "rcoem_dataset.json"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset {dataset_path} not found.")

dataset = load_dataset("json", data_files=dataset_path)

# -----------------------------------
# FORMAT DATA
# -----------------------------------
def format_data(example):
    text = f"""
Instruction:
{example['instruction']}
Response:
{example['response']}
"""
    return {"text": text}

dataset = dataset.map(format_data)

# -----------------------------------
# TOKENIZE
# -----------------------------------
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize)

# -----------------------------------
# TRAINING ARGUMENTS
# -----------------------------------
training_args = TrainingArguments(
    output_dir="./rcoem_model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    fp16=True
)

# -----------------------------------
# TRAINER
# -----------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# -----------------------------------
# START TRAINING
# -----------------------------------
print("Starting Fine-Tuning for RCOEM...")
trainer.train()

# -----------------------------------
# SAVE MODEL
# -----------------------------------
trainer.model.save_pretrained("./rcoem_model")
tokenizer.save_pretrained("./rcoem_model")
print("RCOEM Model Training Completed successfully!")
