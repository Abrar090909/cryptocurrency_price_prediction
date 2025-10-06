# Cryptocurrency Price Movement Prediction using Deep Reinforcement Learning

This project implements an intelligent cryptocurrency trading bot that uses a Deep Reinforcement Learning (DRL) model to predict Bitcoin price movements and automate trading decisions (Buy, Sell, Hold). The agent is built using the Soft Actor-Critic (SAC) algorithm, a state-of-the-art DRL technique, to learn optimal trading strategies from historical market data.

## 📋 Table of Contents
- [About The Project](#about-the-project)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Scope](#future-scope)
- [License](#license)
- [Contact](#contact)

## 🤖 About The Project

Cryptocurrency markets, particularly Bitcoin, are known for their high volatility and unpredictability. This project addresses the challenges of manual trading by developing an autonomous agent that leverages AI to make data-driven decisions.

Unlike traditional models, this system uses a reinforcement learning approach where an agent interacts with a simulated trading environment. It learns to maximize its rewards (profits) over time, adapting its strategy to market fluctuations. The core of this project is the **Soft Actor-Critic (SAC)** model, which balances exploration of new strategies with exploitation of known profitable ones.

## ✨ Key Features

- **Automated Trading:** Autonomously executes Buy, Sell, and Hold decisions.
- **Adaptive Learning:** Continuously refines its trading strategy by learning from past successes and failures.
- **Risk Management:** Aims to optimize the risk-reward ratio by learning from a reward mechanism.
- **Data-Driven Predictions:** Bases decisions on historical price, volume, and market cap data.
- **High Accuracy:** Achieves high accuracy in predicting directional price movements (increase vs. decrease).

## 🏗️ Model Architecture

The system is built upon an **Actor-Critic** framework, a popular architecture in deep reinforcement learning.

-   **Actor Network:** This network acts as the policy-maker. It takes the current market state as input and decides on the best action to take (Buy, Sell, or Hold).
-   **Critic Network:** This network evaluates the action taken by the Actor. It estimates the expected future reward (Q-value) for a given state-action pair, providing feedback that helps the Actor improve its decisions.

The model is trained using the **Soft Actor-Critic (SAC)** algorithm from the Stable-Baselines3 library, which is known for its sample efficiency and stability.

## 📊 Dataset

The model was trained on historical Bitcoin data sourced from **Yahoo Finance**. The dataset includes the following key features for each day:
-   **Open:** Opening price
-   **High:** Highest price
-   **Low:** Lowest price
-   **Close:** Closing price
-   **Volume:** Total trading volume
-   **Marketcap:** Market capitalization

## 🛠️ Technologies Used

-   **Programming Language:** Python 3.8+
-   **DRL Framework:** PyTorch
-   **RL Library:** Stable-Baselines3
-   **RL Environment:** Gymnasium (formerly OpenAI Gym)
-   **Data Manipulation:** Pandas, NumPy
-   **Data Visualization:** Matplotlib, Seaborn
-   **Machine Learning:** Scikit-learn

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Make sure you have Python 3.8 or higher installed on your system.

### Installation

1.  Clone the repository:
    ```sh
    git clone [https://github.com/your_username/your_repository_name.git](https://github.com/your_username/your_repository_name.git)
    ```
2.  Navigate to the project directory:
    ```sh
    cd your_repository_name
    ```
3.  Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created containing all the necessary libraries like pandas, numpy, torch, stable-baselines3, gymnasium, etc.)*

## 📈 Usage

1.  Ensure you have the `coin_Bitcoin.csv` dataset in the project's root directory.
2.  Run the main Python script to start the training and evaluation process:
    ```sh
    python main.py
    ```
The script will load the data, preprocess it, train the SAC model, evaluate its performance, and save the trained model as `bitcoin_price_sac_model.zip`. It will also output the final trading decision based on the most recent data.

## 🏆 Results

The model demonstrates strong performance in predicting the direction of Bitcoin's price movements.

-   **Classification Accuracy:** **95.94%**
    *(This metric indicates how accurately the model predicts whether the price will increase or decrease the next day.)*

-   **Confusion Matrix:**
    ![Confusion Matrix](https://i.imgur.com/your-confusion-matrix-image-url.png)
    *(You would replace this with the actual image of your confusion matrix)*

-   **Final Trading Decision Example:**
    ```
    Trading Decision: Buy
    ```

## 🔮 Future Scope

This project serves as a strong foundation, and future work can include:

-   **Real-Time Integration:** Connect the model to a live cryptocurrency exchange API to execute real trades.
-   **Multi-Asset Portfolio:** Extend the model to manage a diversified portfolio of multiple cryptocurrencies.
-   **Alternative Data:** Incorporate sentiment analysis from news and social media to enhance predictions.
-   **Advanced Risk Management:** Implement more sophisticated risk controls like dynamic stop-loss and take-profit orders.
-   **Hyperparameter Tuning:** Further optimize the model's performance by fine-tuning the SAC algorithm's parameters.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact

Abrar Ahmed - myselfabrar.23@gmail.com

Project Link: [https://github.com/your_username/your_repository_name](https://github.com/your_username/your_repository_name)
