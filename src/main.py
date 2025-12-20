import os
import io
import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException 
from fastapi.concurrency import run_in_threadpool
import asyncio
from PIL import Image

from src.modeling_divedoc import get_model
from src.processing_divedoc import get_processor

constants = {}
model_lock = asyncio.Lock() 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Init & Load model/processor
    """
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("WARNING: HF_TOKEN not found!")

    try:
        constants["processor"] = get_processor(hf_token,img_height=2048, img_width=2048, img_lm_input_seq_length=4096)
        
        constants["model"] = get_model()
        
        constants["device"] = constants["model"].device
        
        print("[INFO] Model & Processor loaded. [INFO]")
        
    except Exception as e:
        print(f"Error Message : {e}")
        raise e

    yield # API running

    # when API is stopped
    constants.clear()
    print("[INFO] API stopped. [INFO]")


# --- Start the API ---
app = FastAPI(
    title="DIVE-Doc API",
    description="OCR-Free Document VQA API",
    version="1.0.0",
    lifespan=lifespan
)

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": "model" in constants}

@app.post("/ask")
async def ask_question(
    question: str = Form(...),    
    file: UploadFile = File(...)
):
    
    if "model" not in constants:
        raise HTTPException(status_code=503, detail="Model not found.")
    
    if "processor" not in constants:
        raise HTTPException(status_code=503, detail="Processor not found.")
    
    if "device" not in constants:
        raise HTTPException(status_code=503, detail="Device not found.")

    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Unsupported format, require JPG or PNG.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        model = constants["model"]
        processor = constants["processor"]
        device = constants["device"]

        inputs = processor(text=question, images=image, return_tensors="pt", padding=True).to(device)
        input_length = inputs["input_ids"].shape[-1]

        # authorize on thread at a time
        async with model_lock:
            def generate_answer():
                with torch.inference_mode():
                    return model.generate( **inputs, max_new_tokens=100,do_sample=False)

            output_ids = await run_in_threadpool(generate_answer)

        generated_ids = output_ids[0][input_length:]
        answer = processor.decode(generated_ids, skip_special_tokens=True)

        return {
            "question": question,
            "answer": answer,
            "filename": file.filename
        }

    except Exception as e:
        print(f"Error during model prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# To start the server using python main.py
if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)