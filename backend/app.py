import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "./rcoem_model"
fallback_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

tokenizer = None
model = None

try:
    if os.path.exists(model_path):
        print(f"Loading RCOEM model from {model_path} onto {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto"
        )
    else:
        print(f"RCOEM model not found. Downloading fallback model ({fallback_model})...")
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        model = AutoModelForCausalLM.from_pretrained(
            fallback_model,
            torch_dtype=dtype,
            device_map="auto"
        )
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None

dataset = []
try:
    with open("rcoem_dataset.json", "r") as f:
        dataset = json.load(f)
except:
    pass

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_text(data: PromptRequest):
    prompt_lower = data.prompt.lower().strip()
    
    # Dataset fallback: exact or substring match (removed brittle keyword intercepts)
    for item in dataset:
        if item["instruction"].lower() in prompt_lower or prompt_lower in item["instruction"].lower():
            return {"response": item["response"]}
    
    # Normal AI Generation
    if not model or not tokenizer:
        return {"response": "System is running in mock mode. RCOEM model could not be loaded."}
    
    formatted_prompt = f"Instruction:\n{data.prompt}\nResponse:\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Response:\n" in response:
        response = response.split("Response:\n")[-1].strip()
    
    return {"response": response}
