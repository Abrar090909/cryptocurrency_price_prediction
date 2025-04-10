
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, clear_output

# ========================== LOAD DATA ==========================
def get_crypto_data(symbol="BTC-USD", start="2020-01-01", end="2024-12-31"):
    df = yf.download(symbol, start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df

# ========================== ENVIRONMENT ==========================
class CryptoTradingEnv(gym.Env):
    def __init__(self, df, window_size=10, initial_balance=1000):
        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, df.shape[1] - 1), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0  # 1 for holding, 0 for none
        self.asset_value = self.initial_balance
        self.current_step = self.window_size
        self.history = []  # Initialize history here
        return self._get_obs()

    def _get_obs(self):
        obs = self.df.iloc[self.current_step - self.window_size:self.current_step]
        return obs.drop(columns=["Date"]).values.astype(np.float32)

    def step(self, action):
        price = self.df.loc[self.current_step, 'Close']
        if isinstance(price, (pd.Series, np.ndarray)):
            price = price.item()
        prev_asset = self.asset_value
        if action == 0:  # Buy
            if self.position == 0:
                self.position = self.balance / price
                self.balance = 0
        elif action == 1:  # Sell
            if self.position > 0:
                self.balance = self.position * price
                self.position = 0
        self.asset_value = self.balance + self.position * price
        reward = self.asset_value - prev_asset
        self.history.append(self.asset_value)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_obs(), reward, done, {}

    def get_portfolio_history(self):
        return self.history

# ========================== A2C MODEL ==========================
class ActorCriticLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(ActorCriticLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        h_lstm = h_lstm[:, -1, :]
        policy = self.actor(h_lstm)
        value = self.critic(h_lstm)
        return policy, value

# ========================== SHARPE RATIO ==========================
def calculate_sharpe(returns):
    returns = np.array(returns)
    excess_returns = returns - np.mean(returns)
    std = np.std(returns) + 1e-6
    sharpe = (np.mean(returns) / std) * np.sqrt(252)
    return max(0.0, sharpe)

# ========================== TRAINING ==========================
def train(env, model, optimizer, gamma=0.99, episodes=50):
    all_rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        log_probs = []
        values = []
        rewards = []
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            policy_logits, value = model(state_tensor)
            probs = torch.softmax(policy_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            next_state, reward, done, _ = env.step(action.item())
            log_probs.append(dist.log_prob(action))
            values.append(value.squeeze(0))
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            state = next_state
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_reward = sum(rewards).item()
        all_rewards.append(total_reward)
        print(f"Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}")
    return all_rewards

# ========================== PREDICT ACTION FOR DATE ==========================
def predict_action_for_date(env, model, df, date_str, window_size):
    date_index = df[df['Date'] == date_str].index
    if len(date_index) == 0 or date_index[0] < window_size:
        print("Invalid date or not enough data before this date.")
        return
    idx = date_index[0]
    state = df.iloc[idx - window_size:idx].drop(columns=["Date"]).values
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        policy, _ = model(state_tensor)
        action = torch.argmax(torch.softmax(policy, dim=-1)).item()
    actions = {0: "BUY", 1: "SELL", 2: "HOLD"}
    print(f"Predicted action for {date_str}: {actions[action]}")

# ========================== MAIN ==========================
def main():
    global env, df, model
    df = get_crypto_data()
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    env = CryptoTradingEnv(df, window_size=10)
    input_size = env.observation_space.shape[1]
    model = ActorCriticLSTM(input_size=input_size, hidden_size=64, num_actions=3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Training started...")
    rewards = train(env, model, optimizer, episodes=30)
    print("Training complete.")
    plt.figure(figsize=(10, 4))
    plt.plot(env.get_portfolio_history())
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    values = env.get_portfolio_history()
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
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Win Ratio: {win_ratio * 100:.2f}%")
    print(f"Total Profit: ${net_profit:.2f} ({profit_percent:.2f}%)")
    print(f"Max Drawdown: {max_drawdown * 100:.2f}%")

date_input = widgets.Text(
    value='',
    placeholder='YYYY-MM-DD',
    description='Date:',
    disabled=False
)
submit_btn = widgets.Button(description="Predict")
display(date_input, submit_btn)

def on_button_clicked(b):
    clear_output(wait=True)
    display(date_input, submit_btn)
    date_str = date_input.value
    predict_action_for_date(env, model, df, date_str, window_size=10)

submit_btn.on_click(on_button_clicked)

if __name__ == "__main__":
    main()
