# Aviator Predictor AI

An AI-powered prediction tool for the Aviator crash game. It uses **OCR screen reading**, **Markov chains**, **LSTM/GRU neural networks**, and **XGBoost** in an ensemble approach to predict the next multiplier value.

## Features

- **Screen OCR** — Reads multiplier values directly from the screen using Tesseract OCR
- **Markov Chain Analysis** — Time-based statistical predictions and streak analysis
- **LSTM & GRU Neural Networks** — Deep learning models trained on historical data
- **XGBoost Ensemble** — Optional gradient boosting model for improved accuracy
- **Anomaly Detection** — IsolationForest-based outlier detection
- **Self-Improving** — Tracks prediction accuracy and retrains automatically
- **Tkinter GUI** — User-friendly interface with real-time statistics and charts

## Requirements

- Python 3.8+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed on your system

## Installation

1. **Install Python dependencies:**

   ```bash
   pip install numpy pandas scikit-learn tensorflow xgboost opencv-python pillow pytesseract matplotlib
   ```

2. **Install Tesseract OCR:**

   - Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - Default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
   - If installed elsewhere, update `TESSERACT_PATH` in `Aviator_AI.py`

## Usage

```bash
python Aviator_AI.py
```

1. Click **"Bölge Ayarla"** (Set Region) to select the screen area where the multiplier is displayed.
2. Click **"Başlat"** (Start) to begin reading and predicting.
3. The program captures multipliers via OCR, logs them to CSV, and generates predictions for the next round.
4. Use **"İstatistikleri Göster"** and **"Grafik Penceresi"** buttons to view stats and charts.

## How Predictions Work

The system uses an **ensemble** of four approaches:

| Model | Description |
|-------|-------------|
| **Statistical** | Markov chain + hourly patterns + streak analysis |
| **LSTM** | Long Short-Term Memory neural network |
| **GRU** | Gated Recurrent Unit neural network |
| **XGBoost** | Gradient boosted decision trees (optional) |

**Accuracy evaluation:**
- If prediction ≥ actual → error must be < 0.50 to count as correct
- If prediction < actual → error must be < 1.50 to count as correct

The model requires at least **20 data points** before it starts generating predictions.

## Project Structure

```
Aviator/
├── Aviator_AI.py            # Main application
├── setup_instructions.txt   # Setup guide (Turkish)
└── README.md                # This file
```

**Generated at runtime:**
```
~/Desktop/AviatorData_v2/
├── training_data_v2.csv     # Collected data & predictions
├── aviator_carpanlar_v2.txt # Multiplier history log
├── aviator_v2_log.txt       # Application logs
├── aviator_lstm_model_v2.h5 # Trained LSTM model
├── aviator_gru_model_v2.h5  # Trained GRU model
├── aviator_xgb_model_v2.json# Trained XGBoost model
└── screenshots_v2/          # OCR screenshots
```

## Disclaimer

This tool is for **educational and experimental purposes only**. Crash games are inherently random (RNG-based), and no prediction model can guarantee results. Use responsibly.

## License

This project is proprietary. All rights reserved.
