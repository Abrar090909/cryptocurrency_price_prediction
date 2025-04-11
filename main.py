from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ⬇️ Add CORS here, right after app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # use ["https://your-bolt-app.com"] in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello from Render!"}

# Other routes follow here
