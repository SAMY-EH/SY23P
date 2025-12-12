# SY23P

# Sentiment-Augmented Reinforcement Learning for Regime-Aware Bitcoin Trading

## 📌 Overview

This project builds an **intelligent Bitcoin trading system** that combines:

1. **Sentiment Analysis (NLP)** from tweets/news
2. **Regime Classification** (haussier / neutre / baissier)
3. **Reinforcement Learning Agent** (PPO/SAC)
4. **Backtesting & Performance Dashboard**

The final goal is to produce a **trading model capable of adapting to market conditions**, leveraging both **textual sentiment** and **market time-series data**.

---

## 🧱 Project Architecture

```
/data
  /raw_market
  /raw_sentiment
  /processed

/src
  sentiment/
    preprocess.py
    sentiment_model.py
  regime/
    regime_model.py
  rl/
    env.py
    agent.py
    training.py
  utils/
    indicators.py
    sync.py

/notebooks
  EDA.ipynb
  SentimentTests.ipynb
  RegimeClassifier.ipynb

/models
  sentiment.pt
  regime_classifier.pt
  rl_agent/

/docs
  report.pdf
  presentation.pdf

README.md
```

---

## 🧩 Pipeline Summary

### **1) Data Collection & Synchronization**

* Market data (OHLCV, volume, optionally orderbooks)
* Tweets, Reddit, news headlines
* Time alignment every 5 minutes

### **2) Sentiment Module**

* Preprocessing texts
* Sentiment scoring with FinBERT / CryptoBERT
* Features: sentiment mean, std, momentum, volume

### **3) Regime Prediction**

* LSTM/TCN/Transformer classifier
* Predict probabilities: `P(up), P(neutral), P(down)`

### **4) Reinforcement Learning Agent**

* Env Gym-like
* PPO or SAC
* Action space: continuous position [-1, 1]
* Reward includes returns + transaction costs + risk penalty

### **5) Backtesting & Evaluation**

* Portfolio evolution
* Sharpe, Sortino, drawdown
* Comparison vs Buy&Hold and basic strategies

---

## 🗂 Technologies

* Python
* PyTorch / TensorFlow
* Transformers (HuggingFace)
* Stable Baselines 3
* Pandas / NumPy
* Matplotlib / Plotly for visualization
* Jupyter Notebooks

---

## 🚀 Getting Started

### 1. Clone the repository

```
git clone <repo-url>
cd <repo>
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Download datasets

(Add instructions once dataset sources are confirmed)

---

## 📊 Deliverables

* Full trading pipeline (NLP → Regime → RL)
* Interactive dashboard (Streamlit or Power BI)
* Research-style PDF report
* Slides for presentation

---

## 👤 Contributors

* **Person A**: NLP + Regime Modeling
* **Person B**: RL + Backtesting

---

## 📄 License

MIT License (or change if needed).
