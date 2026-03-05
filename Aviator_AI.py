#!/usr/bin/env python
# -- coding: utf-8 --
"""
aviator_predictor_v2.py

VERSIYON: v2
-------------------------------------------------------------------------------
Bu proje, önceki "aviator_predictor_PRO.py" kodunun çok daha geliştirilmiş ve
detaylandırılmış halidir. Aşağıdaki kod mümkün olduğunca +1000 satır olacak
şekilde tasarlanmıştır. Bu denli uzun tutulma amacı, normalde farklı modüllere
bölünebilecek bir projeyi tek dosyada detaylı dokümantasyon ve ek işlevlerle
tutabilmektir.

Ne Yapıyor?
-----------
Bu program, "Aviator" adlı bir oyunda (veya benzer RNG tabanlı çarpan
oyunlarında) otomatik olarak ekrandan OCR ile çarpan değerini okur, bu çarpanları
CSV dosyasına kaydeder. Ayrıca Markov zinciri tabanlı, saat-bazlı istatistiksel
yaklaşımlarla ve LSTM/GRU nöral ağlarıyla tahmin üretir. Ek olarak XGBoost veya
RandomForest gibi bir ek model de ensemble'a dahil edilebilir. Tahmin sonuçlarını
da CSV dosyasına işler, "programın tahmini - gerçek" farkını hesaplayarak kendini
"doğru" veya "yanlış" kabul edecek şekilde eğitir.

Geniş Özellikler:
-----------------
1) Tesseract OCR ile ekrandan çarpan okumak
2) Tkinter GUI
3) Anomali tespiti (IsolationForest)
4) Büyük çarpan Winsorizing (ör. 20x üstüne 20 kabul)
5) Markov + saat bazlı + streak analizi
6) LSTM ve GRU ağlarının her ikisini de eğitme ve tahmin
7) Opsiyonel: XGBoost ek modeli
8) Ensemble (Statistical + LSTM + GRU + XGBoost)
9) Online (adım adım) ve toplu (belirli veri biriktiğinde) eğitim
10) İki farklı threshold'a göre (0.50 / 1.50) doğru-yanlış değerlendirmesi
11) Güven oranı (Confidence) hesaplaması - gizleme seçeneği
12) Geniş ve gereksiz uzun dokümantasyon (istem nedeniyle)...

Kurulum:
--------
- Python 3.7+
- pip install --upgrade:
  * numpy
  * pandas
  * scikit-learn
  * tensorflow (veya tensorflow-cpu)
  * xgboost (opsiyonel, fakat ekledik)
  * opencv-python
  * pillow
  * pytesseract
  * matplotlib
  * python-tk (bazı sistemlerde tkinter önceden yüklü)

Kullanım:
---------
1) Tesseract konumunu TESSERACT_PATH ile belirt.
2) Kod çalışırken bir GUI açılır. Başlat deyince ekrandaki (x1,y1,x2,y2) bölgesini
   tarayıp OCR ile çarpan bulmaya çalışır.
3) Her yakaladığı çarpanı CSV'ye yazar ve bir SONRAKİ tahmin oluşturur (ilk turda
   tahmin yok).
4) Bir dahaki veriye "bir önceki tahmini" yazar; aradaki farkı (tahmin - gerçek)
   hesaplayıp "doğru mu yanlış mı" işaretler. 
   * Doğru/yanlış kuralı:
     - Eğer tahmin >= gerçek => fark = tahmin - gerçek >= 0.50 ise yanlış
                                                  < 0.50 ise doğru
     - Eğer tahmin <  gerçek => fark = gerçek - tahmin >= 1.50 ise yanlış
                                                  < 1.50 ise doğru
5) LSTM ve GRU modellerine veri gider. Ensemble'da da nihai tahmin üretilir.
6) GUI üzerinden "Bölge Ayarla" ile OCR okuma alanını değiştirebilirsin.
7) "İstatistikleri Göster", "Grafik Penceresi" gibi butonlar da var.

Notlar:
-------
- Bu kadar uzun kod normalde best practice değil. Yüzlerce satırı birleştirdik
  çünkü talep "mümkün olduğunca uzun" oldu. 
- Aşağıda göreceğin her metot ve fonksiyona bolca dokümantasyon ekliyoruz.
- Gerçek kullanımda "predictor" modülünü, "gui" modülünü, "models" modülünü,
  "data" modülünü vs. ayırmak daha sağlıklı.

-------------------------------------------------------------------------------
"""


import os
import sys
import time
import logging
from datetime import datetime
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd

# OCR
from PIL import ImageGrab, Image, ImageTk
import pytesseract
import cv2

# ML / AI
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# XGBoost (opsiyonel, ensemble'de kullanılacak)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Matplotlib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from collections import defaultdict

################################################################################
# SATIR SAYACI EKLİYORUZ - KODUN +1000 SATIR OLMASINI KOLAYLAŞTIRMAK İÇİN       #
################################################################################

# Satır sayacı için basit bir metod
def line_counter_decorator(func):
    """Decorator for counting lines of code execution. This is purely to
    artificially inflate the code size and to demonstrate that we can surpass
    1000 lines for this script. Real usage wouldn't do this.
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result
    return wrapper

################################################################################
# 0) BAZI SABITLER VE GLOBAL DEĞERLER                                          #
################################################################################

@line_counter_decorator
def configure_globals():
    """Küresel sabitlerin konfigürasyonunu yapan fonksiyon. Bu fonksiyon
    normalde doğrudan kod içinde verilebilir ama biz satır sayısını kabartmak
    adına ekstra fonksiyon oluşturuyoruz.
    """
    # Masaüstü dizinini tespit
    global DESKTOP_PATH
    DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")

    # Veri klasörü
    global DATA_FOLDER
    DATA_FOLDER = os.path.join(DESKTOP_PATH, "AviatorData_v2")
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # CSV dosyası
    global TRAIN_CSV
    TRAIN_CSV = os.path.join(DATA_FOLDER, "training_data_v2.csv")

    # History
    global HISTORY_FILE
    HISTORY_FILE = os.path.join(DATA_FOLDER, "aviator_carpanlar_v2.txt")

    # Model dosyaları
    global MODEL_FILE_LSTM
    MODEL_FILE_LSTM = os.path.join(DATA_FOLDER, "aviator_lstm_model_v2.h5")

    global MODEL_FILE_GRU
    MODEL_FILE_GRU = os.path.join(DATA_FOLDER, "aviator_gru_model_v2.h5")

    global MODEL_FILE_XGB
    MODEL_FILE_XGB = os.path.join(DATA_FOLDER, "aviator_xgb_model_v2.json")  # XGBoost

    # Log
    global LOG_FILE
    LOG_FILE = os.path.join(DATA_FOLDER, "aviator_v2_log.txt")

    # Screenshot folder
    global SCREENSHOT_FOLDER
    SCREENSHOT_FOLDER = os.path.join(DATA_FOLDER, "screenshots_v2")
    os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

    # Tesseract path
    global TESSERACT_PATH
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    # Logging basic config
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Varsayılan OCR bölgesi
    global DEFAULT_X1, DEFAULT_Y1, DEFAULT_X2, DEFAULT_Y2
    DEFAULT_X1, DEFAULT_Y1, DEFAULT_X2, DEFAULT_Y2 = 960, 520, 1350, 620

    # Anomali ve diğer parametreler
    global MIN_DATA_THRESHOLD
    MIN_DATA_THRESHOLD = 30  # LSTM/GRU/ensemble eğitimi için minimum veri sayısı

    global RETRAIN_THRESHOLD
    RETRAIN_THRESHOLD = 10  # Yeni veri geldiğinde, bu sayıya ulaşırsa toplu retrain

    global BATCH_SIZE_ONLINE
    BATCH_SIZE_ONLINE = 4   # Online eğitim batch büyüklüğü

    global WAIT_AFTER_READ
    WAIT_AFTER_READ = 4     # OCR okuduktan sonra bekleme

    # Eşikler (fark)
    # 1) Tahmin >= Gerçek => 0.50
    # 2) Tahmin < Gerçek => 1.50
    global THRESHOLD_HIGH
    THRESHOLD_HIGH = 0.50

    global THRESHOLD_LOW
    THRESHOLD_LOW = 1.50

    # Ensemble için
    global ENSEMBLE_WEIGHT_STAT
    global ENSEMBLE_WEIGHT_LSTM
    global ENSEMBLE_WEIGHT_GRU
    global ENSEMBLE_WEIGHT_XGB

    # Toplam = 1.0 olacak şekilde. Bu sadece bir örnek.
    # Dilersen toplayınca 1.0 olmasına da gerek yok, normalize edebilirsin.
    ENSEMBLE_WEIGHT_STAT = 0.25
    ENSEMBLE_WEIGHT_LSTM = 0.25
    ENSEMBLE_WEIGHT_GRU  = 0.25
    ENSEMBLE_WEIGHT_XGB  = 0.25

    # Güven oranını göster/gösterme ayarı
    global SHOW_CONFIDENCE
    SHOW_CONFIDENCE = True

    # Scale objesi
    global USE_SCALER
    USE_SCALER = True  # Veriyi LSTM/GRU/XGB'e vermeden önce MinMaxScaler kullan

@line_counter_decorator
def ensure_tesseract():
    """Tesseract OCR'nin varlığını kontrol eder, yoksa hata fırlatır."""
    if not os.path.exists(TESSERACT_PATH):
        raise FileNotFoundError("Tesseract OCR bulunamadı veya yolu hatalı.")


################################################################################
# 1) UTILS                                                                     #
################################################################################

@line_counter_decorator
def get_color_category(multiplier: float) -> str:
    """
    multiplier <= 1.99 -> Mavi
    multiplier <= 9.99 -> Mor
    else -> Pembe
    """
    if multiplier <= 1.99:
        return "Mavi"
    elif multiplier <= 9.99:
        return "Mor"
    else:
        return "Pembe"


@line_counter_decorator
def color_to_onehot(c: str):
    """Renklere one-hot vektör döndür."""
    if c == "Mavi":
        return [1,0,0]
    elif c == "Mor":
        return [0,1,0]
    else:
        return [0,0,1]


@line_counter_decorator
def onehot_to_color(arr):
    """One-hot'u tekrar string renge dön."""
    idx = np.argmax(arr)
    if idx == 0:
        return "Mavi"
    elif idx == 1:
        return "Mor"
    else:
        return "Pembe"


@line_counter_decorator
def check_correctness(prediction, actual):
    """
    Tahmin (prediction) ile Gerçek (actual) arasındaki farkı değerlendirip
    doğru/yanlış döndürür.

    Kural:
        if pred >= actual:
            fark = pred - actual
            if fark > THRESHOLD_HIGH (0.50):
                yanlis
            else:
                dogru
        else:
            fark = actual - pred
            if fark > THRESHOLD_LOW (1.50):
                yanlis
            else:
                dogru

    Geri döndür: (is_correct, absolute_diff)
    """
    global THRESHOLD_HIGH, THRESHOLD_LOW
    if prediction >= actual:
        diff = prediction - actual
        if diff > THRESHOLD_HIGH:
            return (False, diff)
        else:
            return (True, diff)
    else:
        diff = actual - prediction
        if diff > THRESHOLD_LOW:
            return (False, diff)
        else:
            return (True, diff)


################################################################################
# 2) ANOMALİ TESPİTİ                                                           #
################################################################################

class AnomalyDetector:
    """
    IsolationForest tabanlı basit bir anomali tespit sistemi.
    """
    @line_counter_decorator
    def _init_(self, contamination=0.05, random_state=42):
        self.model = None
        self.contamination = contamination
        self.random_state = random_state

    @line_counter_decorator
    def train(self, X):
        """
        X: shape = (n_samples, n_features)
        X genelde [[multiplier], [multiplier], ...] şeklinde olabilir.
        """
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.model.fit(X)

    @line_counter_decorator
    def predict(self, X):
        """
        1 => normal
        -1 => anomali
        """
        if not self.model:
            return np.ones(len(X), dtype=int)
        return self.model.predict(X)


################################################################################
# 3) LSTM MODEL                                                                 #
################################################################################

class AviatorLSTM:
    """
    LSTM tabanlı model.
    """
    @line_counter_decorator
    def _init_(self, seq_len=5, lr=0.001, epochs=10, batch_size=8):
        self.model = None
        self.sequence_length = seq_len
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    @line_counter_decorator
    def build_model(self, input_dim):
        inp = keras.Input(shape=(self.sequence_length, input_dim), name="lstm_input")
        x = layers.LSTM(32, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(inp)
        x = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        out_color = layers.Dense(3, activation='softmax', name='color_out')(x)
        out_mult  = layers.Dense(1, activation='linear', name='mult_out')(x)

        model = keras.Model(inputs=inp, outputs=[out_color, out_mult], name="AviatorLSTMModel")
        model.compile(
            optimizer=Adam(self.lr),
            loss={'color_out': 'categorical_crossentropy','mult_out':'mse'},
            loss_weights={'color_out':1.0, 'mult_out':1.0}
        )
        self.model = model

    @line_counter_decorator
    def train_model(self, X, y_color, y_mult):
        if not self.model:
            return
        self.model.fit(
            X,
            {'color_out': y_color, 'mult_out': y_mult},
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1
        )

    @line_counter_decorator
    def train_on_batch(self, X, y_color, y_mult):
        if not self.model:
            return
        loss = self.model.train_on_batch(X, {'color_out': y_color, 'mult_out': y_mult})
        return loss

    @line_counter_decorator
    def predict(self, X):
        if not self.model:
            return None, None
        preds = self.model.predict(X)
        color_probs = preds[0]
        mult_vals = preds[1]
        return color_probs, mult_vals

    @line_counter_decorator
    def load(self, path):
        self.model = keras.models.load_model(path)

    @line_counter_decorator
    def save(self, path):
        if self.model:
            self.model.save(path)


################################################################################
# 4) GRU MODEL                                                                  #
################################################################################

class AviatorGRU:
    """
    GRU tabanlı model. Yapısı LSTM'e oldukça benziyor ama GRU hücreleri kullanıyor.
    """
    @line_counter_decorator
    def _init_(self, seq_len=5, lr=0.001, epochs=10, batch_size=8):
        self.model = None
        self.sequence_length = seq_len
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    @line_counter_decorator
    def build_model(self, input_dim):
        inp = keras.Input(shape=(self.sequence_length, input_dim), name="gru_input")
        x = layers.GRU(32, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(inp)
        x = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        out_color = layers.Dense(3, activation='softmax', name='color_out')(x)
        out_mult  = layers.Dense(1, activation='linear', name='mult_out')(x)

        model = keras.Model(inputs=inp, outputs=[out_color, out_mult], name="AviatorGRUModel")
        model.compile(
            optimizer=Adam(self.lr),
            loss={'color_out': 'categorical_crossentropy','mult_out':'mse'},
            loss_weights={'color_out':1.0, 'mult_out':1.0}
        )
        self.model = model

    @line_counter_decorator
    def train_model(self, X, y_color, y_mult):
        if not self.model:
            return
        self.model.fit(
            X,
            {'color_out': y_color, 'mult_out': y_mult},
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1
        )

    @line_counter_decorator
    def train_on_batch(self, X, y_color, y_mult):
        if not self.model:
            return
        loss = self.model.train_on_batch(X, {'color_out': y_color, 'mult_out': y_mult})
        return loss

    @line_counter_decorator
    def predict(self, X):
        if not self.model:
            return None, None
        preds = self.model.predict(X)
        color_probs = preds[0]
        mult_vals = preds[1]
        return color_probs, mult_vals

    @line_counter_decorator
    def load(self, path):
        self.model = keras.models.load_model(path)

    @line_counter_decorator
    def save(self, path):
        if self.model:
            self.model.save(path)


################################################################################
# 5) STATISTICAL PREDICTOR (Markov + Saat Bazlı + Streak Analizi)              #
################################################################################

class StatisticalPredictor:
    @line_counter_decorator
    def _init_(self, df: pd.DataFrame):
        self.df = df.copy().sort_values("timestamp").reset_index(drop=True)

    @line_counter_decorator
    def calculate_color_transitions(self):
        transitions = defaultdict(lambda: defaultdict(int))
        colors = self.df["color"].values
        for i in range(len(colors)-1):
            c_now = colors[i]
            c_next = colors[i+1]
            transitions[c_now][c_next] += 1
        for c in transitions:
            total = sum(transitions[c].values())
            for n in transitions[c]:
                transitions[c][n] /= total
        return transitions

    @line_counter_decorator
    def predict_color(self, hour, prev_color, streak_count):
        """
        Markov + hour + streak => final color
        """
        if len(self.df) < 5:
            return "Mavi"
        trans = self.calculate_color_transitions()
        markov_c = self.markov_color(prev_color, trans)
        hour_c   = self.hourly_color(hour)
        streak_c = self.streak_break(prev_color, streak_count)
        # Basit bir majority vote
        candidates = [markov_c, hour_c, streak_c]
        final_c = max(set(candidates), key=candidates.count)
        return final_c

    @line_counter_decorator
    def markov_color(self, prev_color, transitions):
        if prev_color not in transitions or len(transitions[prev_color]) == 0:
            return "Mavi"
        return max(transitions[prev_color].items(), key=lambda x:x[1])[0]

    @line_counter_decorator
    def hourly_color(self, hour):
        subset = self.df[self.df["hour"] == hour]
        if len(subset) < 3:
            if len(self.df) < 3:
                return "Mavi"
            return self.df["color"].mode()[0]
        return subset["color"].mode()[0]

    @line_counter_decorator
    def streak_break(self, color, streak_count):
        if streak_count >= 3:
            freq = self.df["color"].value_counts()
            if len(freq) == 0:
                return color
            mc = freq.index[0]
            return "Mavi" if mc == color else mc
        return color

    @line_counter_decorator
    def predict_multiplier(self, hour, prev_color):
        if len(self.df) < 5:
            return 1.0
        last_20 = self.df.tail(20)
        rolling_mean_20 = last_20["multiplier"].mean()

        subset = self.df[self.df["hour"] == hour]
        if len(subset) >= 5:
            hour_mean = subset["multiplier"].mean()
        else:
            hour_mean = rolling_mean_20

        color_factor_map = {"Mavi":1.0, "Mor":1.05, "Pembe":1.15}
        c_factor = color_factor_map.get(prev_color, 1.0)

        combined = (rolling_mean_20*0.5 + hour_mean*0.5)* c_factor
        return float(max(1.0, min(50.0, combined)))


################################################################################
# 6) XGBOOST PREDICTOR (OPSİYONEL)                                             #
################################################################################

class XGBPredictor:
    """
    XGBoost regressoru hem multiplier'ı tahmin etmek için,
    hem de opsiyonel bir 'color' sınıflandırması için (multi-class) kullanılabilir.
    Fakat bu örnekte sadece multiplier tahminine odaklanacağız. 
    Renk tahmininde scikit-learn bazlı XGBClassifier da kullanabilirdik.
    """
    @line_counter_decorator
    def _init_(self):
        if not HAS_XGBOOST:
            self.model = None
            self.enabled = False
        else:
            self.enabled = True
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )

    @line_counter_decorator
    def fit(self, X, y):
        if not self.enabled:
            return
        self.model.fit(X, y)

    @line_counter_decorator
    def predict(self, X):
        if not self.enabled:
            return None
        return self.model.predict(X)

    @line_counter_decorator
    def save_model(self, path):
        if self.enabled and self.model:
            self.model.save_model(path)

    @line_counter_decorator
    def load_model(self, path):
        if self.enabled and os.path.exists(path):
            self.model = xgb.XGBRegressor()
            self.model.load_model(path)


################################################################################
# 7) ANA LEARNER / CORE LOGIC (ENSEMBLE)                                       #
################################################################################

class AviatorV2Learner:
    """
    Burada her şeyi birleştiriyoruz:
      - CSV read/write
      - Anomaly detection
      - Winsorizing
      - Markov + hour-based + streak (StatisticalPredictor)
      - LSTM
      - GRU
      - XGBoost (opsiyonel)
      - Ensemble tahmin
      - Threshold'a dayalı doğru/yanlış
      - Confidence vs. gizlenebilir
    """
    @line_counter_decorator
    def _init_(self):
        # Aşağıdaki tabloya ek kolonlar ekleyebiliriz
        self.df = pd.DataFrame(columns=[
            'timestamp','multiplier','color','hour','minute','second',
            'prev_multiplier','prev_color','streak_count','anomaly_flag',
            'predicted_color','predicted_multiplier','confidence',
            'is_correct','diff','scaler_used'  # ek kolonlar
        ])
        self.new_data_count = 0

        # next_prediction: bir sonraki veri eklendiğinde, hangi tahmini oraya koyacağız
        self.next_prediction = None

        # Anomali
        self.anomaly_detector = AnomalyDetector()

        # Predictors
        self.stat_predictor = None

        self.lstm = AviatorLSTM(seq_len=5, lr=0.001, epochs=3, batch_size=4)
        self.has_lstm = False

        self.gru  = AviatorGRU(seq_len=5, lr=0.001, epochs=3, batch_size=4)
        self.has_gru = False

        self.xgb_model = XGBPredictor()
        self.has_xgb = self.xgb_model.enabled

        # Scaler
        self.scaler = MinMaxScaler()

        # Data loading
        self.load_data()

        # Build all
        self.build_all()

    @line_counter_decorator
    def load_data(self):
        if os.path.exists(TRAIN_CSV):
            try:
                self.df = pd.read_csv(TRAIN_CSV)
            except:
                pass

        # Kolon eksikse ekle
        needed_cols = [
            'second','anomaly_flag','predicted_color','predicted_multiplier',
            'confidence','is_correct','diff','scaler_used'
        ]
        for col in needed_cols:
            if col not in self.df.columns:
                if col in ['predicted_color']:
                    self.df[col] = "N/A"
                else:
                    self.df[col] = 0

        # History file
        if os.path.exists(HISTORY_FILE):
            self.load_history_file()

        # Duplicate satır varsa
        self.df = self.df.drop_duplicates(subset=['timestamp','multiplier'], keep='last')
        self.save_data()

    @line_counter_decorator
    def load_history_file(self):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            if "✅ Çarpan:" in line:
                parts = line.strip().split(" - ")
                mult_str = parts[0].split(": ")[1].replace("x","")
                val = float(mult_str)
                t_ = parts[1].split(": ")[1]
                c_ = parts[2].split(": ")[1]
                hh,mm,ss = [int(x) for x in t_.split(":")]
                if not self.is_data_exists(val, t_):
                    self.add_new_data(val, c_, hh, mm, ss, from_history=True)

    @line_counter_decorator
    def is_data_exists(self, multiplier, time_):
        if len(self.df) == 0:
            return False
        match = self.df[
            (self.df["multiplier"] == multiplier)
            & (self.df["timestamp"].str.contains(time_))
        ]
        return len(match) > 0

    @line_counter_decorator
    def save_data(self):
        self.df.to_csv(TRAIN_CSV, index=False)

    @line_counter_decorator
    def build_all(self):
        """
        Anomali tespiti, winsorize, stat predictor, LSTM, GRU, XGB...
        """
        # 1) Anomali
        if len(self.df) > 10:
            X_ = self.df[["multiplier"]].values
            self.anomaly_detector.train(X_)
            preds = self.anomaly_detector.predict(X_)
            # IsolationForest -1 => anomali, 1 => normal
            # Biz df'ye 1/0 yazarız
            self.df["anomaly_flag"] = [1 if p==1 else 0 for p in preds]

        # 2) Winsorize => 20
        self.df["multiplier"] = np.minimum(self.df["multiplier"].values, 20.0)

        # 3) Stat predictor
        normal_df = self.df[self.df["anomaly_flag"]==1]
        if len(normal_df) < 5:
            normal_df = self.df
        if len(normal_df) >= 5:
            self.stat_predictor = StatisticalPredictor(normal_df)

        # 4) LSTM build
        self.build_lstm_model()

        # 5) GRU build
        self.build_gru_model()

        # 6) XGB build
        self.build_xgb_model()

    @line_counter_decorator
    def build_lstm_model(self):
        normal_df = self.df[self.df["anomaly_flag"]==1]
        if len(normal_df) < MIN_DATA_THRESHOLD:
            self.has_lstm = False
            return
        X, yc, ym = self.create_lstm_dataset(normal_df, is_lstm=True)
        if X is None or len(X)==0:
            self.has_lstm = False
            return
        input_dim = X.shape[-1]
        if not self.lstm.model:
            self.lstm.build_model(input_dim)
        self.lstm.train_model(X, yc, ym)
        self.lstm.save(MODEL_FILE_LSTM)
        self.has_lstm = True

    @line_counter_decorator
    def build_gru_model(self):
        normal_df = self.df[self.df["anomaly_flag"]==1]
        if len(normal_df) < MIN_DATA_THRESHOLD:
            self.has_gru = False
            return
        X, yc, ym = self.create_lstm_dataset(normal_df, is_lstm=False)  # same fn
        if X is None or len(X)==0:
            self.has_gru = False
            return
        input_dim = X.shape[-1]
        if not self.gru.model:
            self.gru.build_model(input_dim)
        self.gru.train_model(X, yc, ym)
        self.gru.save(MODEL_FILE_GRU)
        self.has_gru = True

    @line_counter_decorator
    def build_xgb_model(self):
        """
        XGB ile multiplier tahmini (renk tahminini şimdilik eklemiyor).
        """
        if not self.has_xgb:
            return
        normal_df = self.df[self.df["anomaly_flag"]==1]
        if len(normal_df) < MIN_DATA_THRESHOLD:
            return
        # XGB'e basit feature set: hour, minute, second, prev_multiplier, color->onehot, streak_count
        feats, labels = self.create_xgb_dataset(normal_df)
        if len(feats)==0:
            return
        self.xgb_model.fit(feats, labels)
        self.xgb_model.save_model(MODEL_FILE_XGB)

    @line_counter_decorator
    def create_lstm_dataset(self, df: pd.DataFrame, is_lstm=True):
        """
        Hem LSTM hem GRU dataset'i aynı mantık:
        seq_len = 5, her satır (hour, minute, second, multiplier, color onehot)
        Y => color onehot, multiplier
        """
        # 1) sort
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 2) seq_len
        seq_len = self.lstm.sequence_length if is_lstm else self.gru.sequence_length

        # 3) X, Y
        X_list = []
        Yc_list = []
        Ym_list = []

        # Hazırlık
        # scaler? 
        use_scaler = USE_SCALER
        # Bazı numeric kolonlar: hour, minute, second, multiplier
        # color -> onehot
        # sonra istenirse streak_count'u da ekleyebilirsin

        # Her satıra bir index
        for i in range(seq_len, len(df)):
            window = df.iloc[i-seq_len:i]
            cur = df.iloc[i]

            seq_data = []
            for j in range(seq_len):
                row_j = window.iloc[j]
                h = row_j["hour"]
                m = row_j["minute"]
                s = row_j["second"]
                pm = row_j["multiplier"]
                oh = color_to_onehot(row_j["color"])
                sc = row_j["streak_count"]  # ekledim
                # data_row = [h,m,s,pm]+ oh
                data_row = [h, m, s, pm, sc] + oh  # bir miktar daha feature
                seq_data.append(data_row)

            c_oh_cur = color_to_onehot(cur["color"])
            mm_cur   = cur["multiplier"]

            X_list.append(seq_data)
            Yc_list.append(c_oh_cur)
            Ym_list.append([mm_cur])

        if len(X_list)==0:
            return None, None, None

        X_ = np.array(X_list, dtype=np.float32)
        Yc_ = np.array(Yc_list, dtype=np.float32)
        Ym_ = np.array(Ym_list, dtype=np.float32)

        # scale X_ shape: (N, seq_len, feature_dim)
        # feature_dim = 4 + 3 + 1 => 8 (hour, minute, second, multiplier, streak, color onehot(3))
        if use_scaler:
            # 2D scale => (N*seq_len, feature_dim)
            orig_shape = X_.shape
            X_2d = X_.reshape(-1, orig_shape[2])  # (N*seq_len, feature_dim)
            self.scaler.partial_fit(X_2d)  # scale yap
            X_2d_scaled = self.scaler.transform(X_2d)
            X_scaled = X_2d_scaled.reshape(orig_shape)
            return X_scaled, Yc_, Ym_
        else:
            return X_, Yc_, Ym_

    @line_counter_decorator
    def create_xgb_dataset(self, df: pd.DataFrame):
        """
        XGB'e basit feature set: hour, minute, second, prev_multiplier, color->onehot, streak_count
        Label: multiplier
        """
        df = df.sort_values("timestamp").reset_index(drop=True)
        # Filtre
        # min 1
        if len(df)<2:
            return None, None

        feats = []
        labels= []
        for i in range(1, len(df)):
            row = df.iloc[i]
            h = row["hour"]
            m = row["minute"]
            s = row["second"]
            pm = row["prev_multiplier"]
            sc = row["streak_count"]
            c_oh = color_to_onehot(row["prev_color"])  # bir önceki color

            # label
            y_ = row["multiplier"]
            x_ = [h,m,s, pm, sc] + c_oh
            feats.append(x_)
            labels.append(y_)

        feats = np.array(feats, dtype=np.float32)
        labels= np.array(labels, dtype=np.float32)

        if USE_SCALER:
            self.scaler.partial_fit(feats)
            feats = self.scaler.transform(feats)

        return feats, labels


    @line_counter_decorator
    def online_train_lstm(self, new_rows):
        if not self.has_lstm or not self.lstm.model:
            return
        normal_part = new_rows[new_rows["anomaly_flag"]==1]
        seq_len = self.lstm.sequence_length
        if len(normal_part) < seq_len:
            return

        # Tüm normal_part üzerinde dataset oluştur
        # Fakat ufak bir hile: Sadece son eklenen veriler seq oluşturmak
        # normal_part -> tek satır vs. => min seq_len

        # Basit yoldan, ufak df = son seq_len + 1 satır
        # Yine de bu sonda online train mantığını temsilen yazıyoruz
        # Bazen tam data gerekecek
        # minimal approach: df = self.df.tail(seq_len+10) 
        # buraya basit olsun diye:
        sub_df = self.df.tail(seq_len+50)
        X_, Yc_, Ym_ = self.create_lstm_dataset(sub_df, is_lstm=True)
        if X_ is None:
            return
        idx=0
        while idx<len(X_):
            end_ = min(idx+BATCH_SIZE_ONLINE, len(X_))
            Xb = X_[idx:end_]
            Ycb= Yc_[idx:end_]
            Ymb= Ym_[idx:end_]
            self.lstm.train_on_batch(Xb, Ycb, Ymb)
            idx = end_
        self.lstm.save(MODEL_FILE_LSTM)

    @line_counter_decorator
    def online_train_gru(self, new_rows):
        if not self.has_gru or not self.gru.model:
            return
        normal_part = new_rows[new_rows["anomaly_flag"]==1]
        seq_len = self.gru.sequence_length
        if len(normal_part) < seq_len:
            return
        sub_df = self.df.tail(seq_len+50)
        X_, Yc_, Ym_ = self.create_lstm_dataset(sub_df, is_lstm=False)
        if X_ is None:
            return
        idx=0
        while idx<len(X_):
            end_ = min(idx+BATCH_SIZE_ONLINE, len(X_))
            Xb = X_[idx:end_]
            Ycb= Yc_[idx:end_]
            Ymb= Ym_[idx:end_]
            self.gru.train_on_batch(Xb, Ycb, Ymb)
            idx = end_
        self.gru.save(MODEL_FILE_GRU)

    @line_counter_decorator
    def online_train_xgb(self, new_rows):
        if not self.has_xgb:
            return
        normal_part = new_rows[new_rows["anomaly_flag"]==1]
        if len(normal_part) < 2:
            return
        sub_df = self.df.tail(100)  # son 100 satır
        feats, labels = self.create_xgb_dataset(sub_df)
        if feats is None or len(feats)==0:
            return
        # XGBoost incremental training is a bit tricky
        # Yine de xgb_model fit çabalayabiliriz
        self.xgb_model.fit(feats, labels)
        self.xgb_model.save_model(MODEL_FILE_XGB)


    @line_counter_decorator
    def add_new_data(self, multiplier, color, hour, minute, second=0, from_history=False):
        """
        Asıl veri ekleme fonksiyonu. 
        1) Bir önceki satırın tahminini koy (self.next_prediction)
        2) Doğru/yanlış hesapla
        3) Kaydet, online eğit, vs.
        4) Yeni tahmini self.next_prediction olarak oluştur
        """
        if len(self.df) > 0:
            prev_multiplier = self.df.iloc[-1]["multiplier"]
            prev_color = self.df.iloc[-1]["color"]
        else:
            prev_multiplier = multiplier
            prev_color = color

        streak_count = self.get_streak_count(color)

        if from_history:
            date_str = datetime.now().strftime("%Y-%m-%d")
            timestamp = f"{date_str} {hour:02d}:{minute:02d}:{second:02d}"
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # clip
        multiplier = min(multiplier, 20.0)

        # anomali
        anomaly_flag = 1
        if self.anomaly_detector.model:
            X_single = np.array([[multiplier]], dtype=np.float32)
            pr_ = self.anomaly_detector.predict(X_single)
            # -1 => anomali, 1 => normal
            anomaly_flag = 1 if pr_[0] == 1 else 0

        # predicted
        if self.next_prediction is not None:
            predicted_color = self.next_prediction["color"]
            predicted_multiplier = self.next_prediction["multiplier"]
            predicted_conf = self.next_prediction["confidence"]
        else:
            predicted_color = "N/A"
            predicted_multiplier = 0.0
            predicted_conf = 0.0

        # correctness
        is_correct_bool, diff_val = check_correctness(predicted_multiplier, multiplier)
        is_correct = 1 if is_correct_bool else 0

        # row
        row_dict = {
            'timestamp': timestamp,
            'multiplier': multiplier,
            'color': color,
            'hour': hour,
            'minute': minute,
            'second': second,
            'prev_multiplier': prev_multiplier,
            'prev_color': prev_color,
            'streak_count': streak_count,
            'anomaly_flag': anomaly_flag,
            'predicted_color': predicted_color,
            'predicted_multiplier': round(predicted_multiplier,2),
            'confidence': round(predicted_conf,1) if SHOW_CONFIDENCE else 0.0,
            'is_correct': is_correct,
            'diff': round(diff_val,3),
            'scaler_used': 1 if USE_SCALER else 0,
        }

        new_row_df = pd.DataFrame([row_dict])
        self.df = pd.concat([self.df, new_row_df], ignore_index=True)

        if not from_history:
            self.save_data()
            self.new_data_count += 1

            if self.new_data_count >= RETRAIN_THRESHOLD:
                self.build_all()
                self.new_data_count = 0
            else:
                self.online_train_lstm(new_row_df)
                self.online_train_gru(new_row_df)
                self.online_train_xgb(new_row_df)

        # Yeni next_prediction
        self.next_prediction = self._predict_next(hour, minute, second, multiplier, color)

        return self.next_prediction


    @line_counter_decorator
    def _predict_next(self, hour, minute, second, prev_mult, prev_color):
        """
        Tüm analizlere, desenlere, serilere bakarak ensemble tahmin hesapla:
         1) Statistical tahmin (renk + multiplier)
         2) LSTM tahmini
         3) GRU tahmini
         4) XGB tahmini
         5) combine => ensemble
        """
        if not self.stat_predictor:
            return None

        # STATS
        scount = self.get_streak_count(prev_color)
        color_stat = self.stat_predictor.predict_color(hour, prev_color, scount)
        mult_stat  = self.stat_predictor.predict_multiplier(hour, prev_color)

        # LSTM
        color_lstm, mult_lstm = None, None
        if self.has_lstm and self.lstm.model:
            seq = self.build_single_sequence(hour, minute, second, prev_mult, prev_color, is_lstm=True)
            if seq is not None:
                c_probs, m_vals = self.lstm.predict(seq)
                if c_probs is not None:
                    color_lstm = onehot_to_color(c_probs[0])
                    mult_lstm  = float(m_vals[0][0])

        # GRU
        color_gru, mult_gru = None, None
        if self.has_gru and self.gru.model:
            seqg = self.build_single_sequence(hour, minute, second, prev_mult, prev_color, is_lstm=False)
            if seqg is not None:
                c_probs_g, m_vals_g = self.gru.predict(seqg)
                if c_probs_g is not None:
                    color_gru = onehot_to_color(c_probs_g[0])
                    mult_gru  = float(m_vals_g[0][0])

        # XGB
        mult_xgb = None
        if self.has_xgb and self.xgb_model.enabled:
            # xgb feature: hour, minute, second, prev_multiplier, prev_color->onehot, streak
            # Tek sample
            c_oh = color_to_onehot(prev_color)
            feat_xgb = np.array([[hour, minute, second, prev_mult, scount] + c_oh], dtype=np.float32)
            if USE_SCALER:
                feat_xgb = self.scaler.transform(feat_xgb)
            y_pred = self.xgb_model.predict(feat_xgb)
            if y_pred is not None:
                mult_xgb = float(y_pred[0])

        # ENSEMBLE:
        # Renk: majority vote => stat, lstm, gru
        color_list = [color_stat]
        if color_lstm:
            color_list.append(color_lstm)
        if color_gru:
            color_list.append(color_gru)
        final_color = max(set(color_list), key=color_list.count)

        # Mult:
        # stat, lstm, gru, xgb
        # ENSEMBLE_WEIGHT_STAT, ENSEMBLE_WEIGHT_LSTM, ENSEMBLE_WEIGHT_GRU, ENSEMBLE_WEIGHT_XGB
        global ENSEMBLE_WEIGHT_STAT, ENSEMBLE_WEIGHT_LSTM, ENSEMBLE_WEIGHT_GRU, ENSEMBLE_WEIGHT_XGB
        w_stat = ENSEMBLE_WEIGHT_STAT
        w_lstm = ENSEMBLE_WEIGHT_LSTM
        w_gru  = ENSEMBLE_WEIGHT_GRU
        w_xgb  = ENSEMBLE_WEIGHT_XGB

        ms = []
        ws = []

        ms.append(mult_stat)
        ws.append(w_stat)

        if mult_lstm is not None:
            ms.append(mult_lstm)
            ws.append(w_lstm)

        if mult_gru is not None:
            ms.append(mult_gru)
            ws.append(w_gru)

        if mult_xgb is not None:
            ms.append(mult_xgb)
            ws.append(w_xgb)

        # Weighted average
        total_w = sum(ws)
        if total_w <= 0:
            # fallback
            final_mult = mult_stat
        else:
            weighted_sum = 0
            for mm, ww in zip(ms, ws):
                weighted_sum += mm*ww
            final_mult = weighted_sum / total_w

        # clip
        if final_mult < 1.0:
            final_mult = 1.0
        if final_mult > 100.0:
            final_mult = 100.0

        # confidence
        conf = self.calc_confidence(final_color, final_mult, hour)

        return {
            "color": final_color,
            "multiplier": round(final_mult,2),
            "confidence": round(conf,1) if SHOW_CONFIDENCE else 0.0
        }

    @line_counter_decorator
    def build_single_sequence(self, hour, minute, second, prev_mult, prev_color, is_lstm=True):
        """
        Tek bir veri girdisi için LSTM/GRU input sekansı oluşturur.
        Yani df.tail(seq_len-1) + bu satır.
        """
        seq_len = self.lstm.sequence_length if is_lstm else self.gru.sequence_length
        if len(self.df)< seq_len:
            return None
        tail_df = self.df.tail(seq_len-1)
        if len(tail_df) < (seq_len-1):
            return None

        seq_data = []
        for i in range(len(tail_df)):
            r_ = tail_df.iloc[i]
            h_ = r_["hour"]
            m_ = r_["minute"]
            s_ = r_["second"]
            mul_ = r_["multiplier"]
            c_oh = color_to_onehot(r_["color"])
            sc_ = r_["streak_count"]
            data_row = [h_, m_, s_, mul_, sc_] + c_oh
            seq_data.append(data_row)

        # current row
        c_oh_cur = color_to_onehot(prev_color)
        seq_data.append([hour, minute, second, prev_mult, self.get_streak_count(prev_color)] + c_oh_cur)

        X_ = np.array([seq_data], dtype=np.float32)
        if USE_SCALER:
            orig_shape = X_.shape
            X_2d = X_.reshape(-1, orig_shape[2])
            # partial_fit => caution, ama deneme
            X_2d_scaled = self.scaler.transform(X_2d)
            X_scaled = X_2d_scaled.reshape(orig_shape)
            return X_scaled
        else:
            return X_

    @line_counter_decorator
    def calc_confidence(self, color, multiplier, hour):
        """
        Oldukça basit bir confidence hesaplaması. 
        Dilersen "toplam is_correct yüzdesi" veya "hour bazlı analiz" vs. eklersin.
        """
        if not SHOW_CONFIDENCE:
            return 0.0

        # 1) Tüm verideki doğru tahmin yüzdesi
        if len(self.df)>0:
            correctness_ratio = self.df["is_correct"].mean()
            if pd.isna(correctness_ratio):
                correctness_ratio = 0.5
        else:
            correctness_ratio = 0.5

        base_conf = correctness_ratio*100.0  # 0-1 => 0-100

        # 2) hour spesifik bir bonus
        hour_df = self.df[self.df["hour"]==hour]
        if len(hour_df)>=5:
            local_correct = hour_df["is_correct"].mean()
            if pd.isna(local_correct):
                local_correct = 0.5
            hour_bonus = (local_correct - 0.5)*20.0
        else:
            hour_bonus = 0

        # 3) degrade / clamp
        conf = base_conf + hour_bonus
        if conf<0:
            conf=0
        if conf>95:
            conf=95
        return conf

    @line_counter_decorator
    def get_streak_count(self, color):
        """
        Son satırlardan itibaren, aynı renk üst üste kaç kez gelmiş?
        """
        streak=0
        for i in range(len(self.df)-1, -1, -1):
            if self.df.iloc[i]["color"] == color:
                streak+=1
            else:
                break
        return streak

    @line_counter_decorator
    def get_history_stats(self):
        if len(self.df)==0:
            return None
        total_records = len(self.df)
        avg_ = self.df["multiplier"].mean()
        mx_  = self.df["multiplier"].max()
        mn_  = self.df["multiplier"].min()
        color_dist = self.df["color"].value_counts().to_dict()

        color_streaks = {"Mavi":0,"Mor":0,"Pembe":0}
        for c in color_streaks:
            color_streaks[c] = self.max_streak(c)

        overall_acc = self.df["is_correct"].mean()*100.0

        stats = {
            "total_records": total_records,
            "avg_multiplier": avg_,
            "max_multiplier": mx_,
            "min_multiplier": mn_,
            "color_distribution": color_dist,
            "color_streaks": color_streaks,
            "overall_accuracy": overall_acc
        }
        return stats

    @line_counter_decorator
    def max_streak(self, color):
        max_s=0
        cur_s=0
        for i in range(len(self.df)):
            if self.df.iloc[i]["color"] == color:
                cur_s+=1
                max_s = max(max_s, cur_s)
            else:
                cur_s=0
        return max_s


################################################################################
# 8) GUI                                                                       #
################################################################################

class AviatorV2GUI:
    """
    Tkinter arayüzü.
    """
    @line_counter_decorator
    def __init__(self):
        ensure_tesseract()
        self.learner = AviatorV2Learner()
        self.root = tk.Tk()
        self.root.title("Aviator Predictor (v2)")
        self.root.resizable(True, True)

        self.read_set = set()
        self.running = False
        self.x1, self.y1, self.x2, self.y2 = DEFAULT_X1, DEFAULT_Y1, DEFAULT_X2, DEFAULT_Y2

        self.success_count = 0
        self.fail_count = 0

        self.setup_gui()

    @line_counter_decorator
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Aviator Predictor v2", font=("Arial",16,"bold")).pack(pady=5)
        ttk.Label(main_frame, text=f"Veri Klasörü: {DATA_FOLDER}", font=("Arial",8)).pack()

        self.status_var = tk.StringVar(value="Durum: Durdu")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, foreground="red")
        self.status_label.pack(pady=5)

        stats_box = ttk.LabelFrame(main_frame, text="Durum", padding=10)
        stats_box.pack(fill=tk.X, pady=5)

        self.success_var = tk.StringVar(value="Veri Alındı: 0")
        self.fail_var    = tk.StringVar(value="Kontrol Ediliyor: 0")
        ttk.Label(stats_box, textvariable=self.success_var, foreground="green").pack()
        ttk.Label(stats_box, textvariable=self.fail_var,    foreground="red").pack()

        # Tahmin Paneli
        pred_box = ttk.LabelFrame(main_frame, text="Sonraki Tahmin", padding=10)
        pred_box.pack(fill=tk.X, pady=5)

        self.pred_color_var = tk.StringVar(value="Renk: -")
        self.pred_mult_var  = tk.StringVar(value="Çarpan: -")
        self.pred_conf_var  = tk.StringVar(value="Güven: -%")

        ttk.Label(pred_box, textvariable=self.pred_color_var).pack()
        ttk.Label(pred_box, textvariable=self.pred_mult_var).pack()
        ttk.Label(pred_box, textvariable=self.pred_conf_var).pack()

        # Listbox
        list_frame = ttk.LabelFrame(main_frame, text="Tespit Edilen Çarpanlar", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.list_box = tk.Listbox(list_frame, font=("Arial",10), selectmode=tk.SINGLE)
        self.list_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.list_box.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.list_box.config(yscrollcommand=scroll.set)

        # Alt butonlar
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(btn_frame, text="Başlat", command=self.start).pack(side=tk.LEFT, padx=5, expand=True)
        ttk.Button(btn_frame, text="Durdur", command=self.stop).pack(side=tk.LEFT, padx=5, expand=True)
        ttk.Button(btn_frame, text="Temizle", command=self.clear_list).pack(side=tk.LEFT, padx=5, expand=True)
        ttk.Button(btn_frame, text="Çıkış", command=self.exit_app).pack(side=tk.LEFT, padx=5, expand=True)

        ttk.Button(main_frame, text="Bölge Ayarla", command=self.set_region).pack(pady=5)
        ttk.Button(main_frame, text="İstatistikleri Göster", command=self.show_stats).pack(pady=5)
        ttk.Button(main_frame, text="Grafik Penceresi", command=self.show_graph_window).pack(pady=5)
        ttk.Button(main_frame, text="Veri Klasörünü Aç", command=self.open_data_folder).pack(pady=5)

    @line_counter_decorator
    def preprocess_image(self, pil_img: Image.Image):
        """
        OCR için resmi ön işlem.
        """
        np_img = np.array(pil_img.convert("RGB"))
        hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)

        lower1 = np.array([0,70,50], dtype=np.uint8)
        upper1 = np.array([10,255,255], dtype=np.uint8)
        lower2 = np.array([170,70,50], dtype=np.uint8)
        upper2 = np.array([180,255,255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((2,2), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        _, final = cv2.threshold(red_mask, 127,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return Image.fromarray(final)

    @line_counter_decorator
    def read_multiplier(self):
        """
        OCR ile ekrandan multiplier okumak.
        Ekran: (self.x1, self.y1, self.x2, self.y2)
        """
        try:
            sshot = ImageGrab.grab(bbox=(self.x1,self.y1,self.x2,self.y2))
            raw_path = os.path.join(SCREENSHOT_FOLDER, "raw_v2.png")
            sshot.save(raw_path)

            proc_img = self.preprocess_image(sshot)
            proc_path = os.path.join(SCREENSHOT_FOLDER, "processed_v2.png")
            proc_img.save(proc_path)

            for psm in [7,8,6,13]:
                config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=.0123456789xX"
                text = pytesseract.image_to_string(proc_img, config=config).strip()
                if 'x' in text.lower():
                    txt_ = text.lower().split('x')[0].strip()
                    try:
                        val = float(txt_)
                        if 1.0 <= val <= 100.0:
                            now_str = datetime.now().strftime("%H:%M:%S")
                            h,m,s = [int(x) for x in now_str.split(":")]
                            c_ = get_color_category(val)

                            msg = f"Çarpan: {val:.2f}x - Saat: {now_str} - Renk: {c_}"
                            if msg not in self.read_set:
                                self.read_set.add(msg)
                                self.success_count += 1

                                # add_new_data
                                next_pred = self.learner.add_new_data(val, c_, h, m, s)
                                self.list_box.insert(0, msg)

                                if next_pred:
                                    self.update_prediction_display(next_pred)

                                with open(HISTORY_FILE, "a", encoding='utf-8') as ff:
                                    ff.write("✅ "+msg+"\n")

                                time.sleep(WAIT_AFTER_READ)
                                return True
                    except ValueError:
                        pass

            self.fail_count += 1
            return False

        except Exception as e:
            logging.error(f"OCR hata: {str(e)}")
            messagebox.showerror("Hata", f"Ekran okuma hatası:\n{str(e)}")
            return False

    @line_counter_decorator
    def main_loop(self):
        """
        OCR okuma döngüsü. 
        """
        while self.running:
            self.read_multiplier()
            self.update_stats()
            time.sleep(0.3)

    @line_counter_decorator
    def start(self):
        if not self.running:
            self.running = True
            self.status_var.set("Durum: Çalışıyor")
            self.status_label.config(foreground="green")
            t = threading.Thread(target=self.main_loop, daemon=True)
            t.start()

    @line_counter_decorator
    def stop(self):
        self.running = False
        self.status_var.set("Durum: Durdu")
        self.status_label.config(foreground="red")

    @line_counter_decorator
    def update_stats(self):
        self.success_var.set(f"Veri Alındı: {self.success_count}")
        self.fail_var.set(f"Kontrol Ediliyor: {self.fail_count}")

    @line_counter_decorator
    def update_prediction_display(self, pred):
        if not pred:
            return
        self.pred_color_var.set(f"Renk: {pred['color']}")
        self.pred_mult_var.set(f"Çarpan: {pred['multiplier']}x")
        self.pred_conf_var.set(f"Güven: %{pred['confidence']}")

    @line_counter_decorator
    def clear_list(self):
        self.list_box.delete(0, tk.END)
        self.read_set.clear()

    @line_counter_decorator
    def exit_app(self):
        if self.running:
            self.stop()
        self.root.quit()

    @line_counter_decorator
    def set_region(self):
        def on_mouse_drag(event):
            nonlocal x2_, y2_
            x2_, y2_ = event.x, event.y
            canvas.coords(rect_id, x1_,y1_, x2_,y2_)

        def on_mouse_release(event):
            nonlocal x2_, y2_
            x2_, y2_ = event.x, event.y
            self.root.attributes("-alpha", 1.0)
            region_win.destroy()
            self.x1, self.y1 = min(x1_, x2_), min(y1_, y2_)
            self.x2, self.y2 = max(x1_, x2_), max(y1_, y2_)

        def on_mouse_click(event):
            nonlocal x1_, y1_
            x1_, y1_ = event.x, event.y
            canvas.bind("<B1-Motion>", on_mouse_drag)
            canvas.bind("<ButtonRelease-1>", on_mouse_release)

        x1_,y1_,x2_,y2_ = 0,0,0,0
        self.root.attributes("-alpha", 0.5)
        region_win = tk.Toplevel(self.root)
        region_win.attributes("-fullscreen", True)
        region_win.attributes("-topmost", True)

        sshot = ImageGrab.grab()
        sw = region_win.winfo_screenwidth()
        sh = region_win.winfo_screenheight()
        sshot = sshot.resize((sw,sh))
        photo = ImageTk.PhotoImage(sshot)

        canvas = tk.Canvas(region_win, cursor="crosshair")
        canvas.create_image(0,0, image=photo, anchor=tk.NW)
        canvas.pack(fill=tk.BOTH, expand=True)

        rect_id = canvas.create_rectangle(0,0,0,0, outline="red", width=2)
        canvas.bind("<Button-1>", on_mouse_click)
        region_win.mainloop()

    @line_counter_decorator
    def show_stats(self):
        stats = self.learner.get_history_stats()
        if not stats:
            messagebox.showinfo("İstatistikler", "Henüz veri yok!")
            return

        win = tk.Toplevel(self.root)
        win.title("Geçmiş Veri Analizi (v2)")
        win.geometry("600x600")

        ttk.Label(win, text="Genel İstatistikler", font=("Arial",12,"bold")).pack(pady=10)

        t = stats["total_records"]
        avg_ = stats["avg_multiplier"]
        mx_  = stats["max_multiplier"]
        mn_  = stats["min_multiplier"]
        dist = stats["color_distribution"]
        c_streak = stats["color_streaks"]
        overall_acc = stats["overall_accuracy"]

        msg = f"""
Toplam Kayıt: {t}
Ortalama Çarpan: {avg_:.2f}x
Maksimum Çarpan: {mx_:.2f}x
Minimum Çarpan: {mn_:.2f}x

Renk Dağılımı:
 Mavi: {dist.get('Mavi',0)}
 Mor: {dist.get('Mor',0)}
 Pembe: {dist.get('Pembe',0)}

En Uzun Seriler:
 Mavi: {c_streak['Mavi']}
 Mor: {c_streak['Mor']}
 Pembe: {c_streak['Pembe']}

Toplam Doğru Tahmin Oranı: {overall_acc:.2f}%
"""

        ttk.Label(win, text=msg, justify=tk.LEFT).pack(padx=10, pady=10)
        ttk.Button(win, text="Kapat", command=win.destroy).pack(pady=10)

    @line_counter_decorator
    def show_graph_window(self):
        data_df = self.learner.df.copy()
        if len(data_df) < 5:
            messagebox.showinfo("Grafik","Yeterli veri yok!")
            return

        win = tk.Toplevel(self.root)
        win.title("Grafik Penceresi (v2)")
        win.geometry("800x600")

        fig = plt.Figure(figsize=(8,4), dpi=100)
        ax = fig.add_subplot(111)
        sub_df = data_df.tail(50)
        x_ = np.arange(len(sub_df))
        y_ = sub_df["multiplier"].values

        ax.plot(x_, y_, label="Gerçek Çarpan", color='blue', linewidth=2)
        ax.set_title("Son 50 Tur Çarpan (v2)")
        ax.set_xlabel("Index")
        ax.set_ylabel("Çarpan")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ttk.Button(win, text="Kapat", command=win.destroy).pack(pady=5)

    @line_counter_decorator
    def open_data_folder(self):
        os.startfile(DATA_FOLDER)

    @line_counter_decorator
    def run(self):
        self.root.mainloop()

################################################################################
# 9) MAIN                                                                      #
################################################################################

@line_counter_decorator
def main():
    # Konfigürasyon
    configure_globals()
    # GUI
    app = AviatorV2GUI()
    app.run()

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Kritik hata: {str(e)}")
        # messagebox nispeten GUIye dokunur, bazen main thread değilse sorun çıkar.
        # Yine de deneyebiliriz:
        import traceback
        traceback_msg = traceback.format_exc()
        print("Kritik hata:", traceback_msg)
