# RCOEM AI ChatBot Project

This is an academic project to fine-tune an open-source LLM for Shri Ramdeobaba College of Engineering and Management (RCOEM). 

It demonstrates how to create a domain-specific conversational AI capable of answering queries regarding admissions, courses, placements, and campus facilities.

## Architecture
- **Base Model**: `mistralai/Mistral-7B-v0.1`
- **Fine-Tuning**: PEFT / LoRA
- **Quantization**: bitsandbytes (4-bit)
- **Backend**: FastAPI
- **Frontend**: React (Vite)

## Setup

### Backend
```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```
To train the model: `python train.py` (Requires GPU)
To run the server: `uvicorn app:app --reload`

### Frontend
```bash
cd frontend
npm install
npm run dev
```
