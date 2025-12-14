# 🌬️ Wind Energy Forecasting using ARIMA & LSTM

## 📌 Project Overview

Accurate wind energy forecasting is a critical challenge in renewable energy systems due to the highly **non-linear, stochastic, and time-dependent** nature of wind patterns. This project presents a **hybrid forecasting framework** that combines traditional statistical models with deep learning techniques to improve prediction accuracy and reliability.

The system leverages:

* **ARIMA** for capturing linear temporal patterns
* **LSTM (Long Short-Term Memory)** networks for learning complex non-linear dependencies
* A **Hybrid ARIMA + LSTM model** that predicts residual errors from ARIMA using LSTM

This approach significantly enhances forecasting performance compared to standalone models.

---

## 🎯 Objectives

* Analyze wind speed data using exploratory data analysis (EDA)
* Build and evaluate a statistical time-series model (ARIMA)
* Design a deep learning-based forecasting model (LSTM)
* Develop a hybrid ARIMA–LSTM model for superior accuracy
* Compare model performance using standard evaluation metrics

---

## 🧠 Methodology

### 1️⃣ Exploratory Data Analysis (EDA)

* Time-series visualization
* Trend and seasonality analysis
* Data cleaning and resampling
* Stationarity checks

### 2️⃣ ARIMA Model

* Captures linear trends and seasonality
* Suitable for short-term forecasting
* Generates baseline predictions

### 3️⃣ LSTM Model

* Deep learning model specialized for sequence prediction
* Learns long-term dependencies in wind patterns
* Handles non-linear relationships effectively

### 4️⃣ Hybrid ARIMA + LSTM Model

* ARIMA predicts base values
* Residual errors are computed
* LSTM learns residual patterns
* Final prediction = ARIMA forecast + LSTM residual forecast

This hybrid strategy combines **statistical stability with deep learning flexibility**.

---

## 📂 Project Structure

```
Wind_energy_prediction/
│
├── Data/
│   └── T1.csv
│
├── Models/
│   ├── arima_model.pkl
│   ├── lstm_model.h5
│   ├── residual_lstm_model.h5
│   ├── scaler_X.pkl
│   ├── scaler_y.pkl
│   └── scaler_residuals.pkl
│
├── Notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_ARIMA_MODEL.ipynb
│   ├── 03_LSTM_MODELS.ipynb
│   ├── 04_HYBRID_MODEL.ipynb
│   └── 05_Evaluation.ipynb
│
├── Results/
│   ├── processed_hourly.csv
│   ├── arima_forecast.npy
│   ├── arima_future_forecast.npy
│   ├── hybrid_forecast.npy
│   ├── hybrid_test_actual.npy
│   ├── hybrid_forecast.png
│   └── evaluation_metrics.csv
│
└── README.md
```

---

## 📊 Evaluation Metrics

The models are evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

The **Hybrid ARIMA + LSTM model** demonstrates superior performance by reducing forecast error and improving generalization.

---

## 🚀 Key Highlights

* End-to-end time series forecasting pipeline
* Combines classical and deep learning models
* Modular and scalable design
* Industry-relevant renewable energy use case
* Resume and interview-ready project

---

## 🛠️ Technologies Used

* **Python**
* **Pandas, NumPy, Matplotlib**
* **Statsmodels (ARIMA)**
* **TensorFlow / Keras (LSTM)**
* **Scikit-learn**

---

## 🔮 Future Enhancements

* Add weather features (temperature, pressure)
* Deploy using Streamlit or Flask
* Extend to multi-step and probabilistic forecasting
* Integrate real-time wind sensor data

---

## 👨‍💻 Author

**Devraj D. Korgaonkar**
B.Tech CSE (AI & ML)
Guru Nanak Institutions Technical Campus

---

⭐ If you find this project useful, feel free to star the repository!
