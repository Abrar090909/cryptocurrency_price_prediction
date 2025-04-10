
from fastapi import FastAPI
from api_model import main as run_model

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is live!"}

@app.get("/predict")
def predict():
    result = run_model()
    return {"prediction": result}
