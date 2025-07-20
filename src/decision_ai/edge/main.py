from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
import textract, requests, os

API_KEY = os.getenv("DECISION_AI_API_KEY", "supersecret")
PREDICT_URL = os.getenv("PREDICT_URL", "http://localhost:8000/predict")

app = FastAPI()
# CORS: permite que o front-end em localhost:3000 faça upload
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

def check_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid key")
    
def safe_decode(data: bytes) -> str:
    """
    Tenta UTF-8, depois ISO-8859-1; se falhar, ignora bytes inválidos.
    """
    for enc in ("utf-8", "iso-8859-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")

@app.post("/upload", dependencies=[Depends(check_key)])
async def upload(file: UploadFile = File(...)):
    with NamedTemporaryFile(delete=True) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        raw_bytes = textract.process(tmp.name)
        text = safe_decode(raw_bytes)
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Falha ao extrair texto: arquivo vazio ou formato não suportado.",
            )
    payload = [{
        "cv_text": text,
        "job_text": ""
    }]
    r = requests.post(PREDICT_URL, json=payload, headers={"X-API-Key": API_KEY})
    if r.status_code != 200:
        raise HTTPException(502, "Prediction failed")
    return r.json()[0]           # {"proba": 0.xx, "pred": 0/1}
