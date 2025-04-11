from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from api_model import CryptoTradingEnv, ActorCriticLSTM, get_crypto_data, train, calculate_sharpe

app = FastAPI()

# Add CORS for Bolt or local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
df = get_crypto_data()
df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
env = CryptoTradingEnv(df, window_size=10)
input_size = env.observation_space.shape[1]
model = ActorCriticLSTM(input_size=input_size, hidden_size=64, num_actions=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Request models
class TrainRequest(BaseModel):
    episodes: int

class PredictRequest(BaseModel):
    date: str

@app.get("/")
def root():
    return {"message": "Hello from Render!"}

@app.post("/train")
def train_model(req: TrainRequest):
    rewards = train(env, model, optimizer, episodes=req.episodes)
    return {"message": f"Training completed for {req.episodes} episodes."}

@app.post("/predict")
def predict_action(req: PredictRequest):
    date_str = req.date
    date_index = df[df['Date'] == date_str].index
    if len(date_index) == 0 or date_index[0] < 10:
        return {"error": "Invalid date or not enough data."}

    idx = date_index[0]
    state = df.iloc[idx - 10:idx].drop(columns=["Date"]).values
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        policy, _ = model(state_tensor)
        action = torch.argmax(torch.softmax(policy, dim=-1)).item()

    actions = {0: "BUY", 1: "SELL", 2: "HOLD"}
    return {"action": actions[action]}

@app.get("/metrics")
def get_metrics():
    values = env.get_portfolio_history()
    if len(values) < 2:
        return {"error": "Not enough data. Please train the model first."}

    returns = np.diff(values) / values[:-1]
    sharpe = calculate_sharpe(returns)

    win_trades = sum(1 for r in returns if r > 0)
    total_trades = len(returns)
    win_ratio = win_trades / total_trades if total_trades > 0 else 0

    initial_value = values[0]
    final_value = values[-1]
    net_profit = final_value - initial_value
    profit_percent = (net_profit / initial_value) * 100

    peak = values[0]
    max_drawdown = 0
    for val in values:
        if val > peak:
            peak = val
        drawdown = (peak - val) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return {
        "sharpe_ratio": round(sharpe, 3),
        "win_ratio": round(win_ratio * 100, 2),
        "profit_percent": round(profit_percent, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
    }
