import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import iqr, skew
from scipy.fft import fft
from tensorflow.keras.layers import LeakyReLU


# 設定隨機種子確保結果可複現
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# FFT 特徵的函數
def fft_features(values, prefix):
    fft_vals = np.abs(fft(values))
    return {
        f"{prefix}_fft_mean": np.mean(fft_vals),
        f"{prefix}_fft_std": np.std(fft_vals),
        f"{prefix}_fft_max": np.max(fft_vals)
    }

# Zero Crossing 計算函數
def zero_crossing_count(values):
    return np.sum((values[:-1] * values[1:]) < 0)

# 主要特徵萃取函數
def extract_features(df, with_labels=True):
    feature_list = []

    for uid, group in df.groupby("unique_id"):
        feature = {"unique_id": uid}
        
        if with_labels:
            labels = group.iloc[0][["gender", "hold racket handed", "play years", "level"]]
            feature.update(labels)

        feature["shot_length"] = len(group) / 85 
        
        # 計算合成量
        acc_mag = np.sqrt(group["Ax"]**2 + group["Ay"]**2 + group["Az"]**2)
        gyro_mag = np.sqrt(group["Gx"]**2 + group["Gy"]**2 + group["Gz"]**2)

        # 加入合成量統計特徵
        for name, values in zip(["acc_mag", "gyro_mag"], [acc_mag, gyro_mag]):
            feature[f"{name}_mean"] = np.mean(values)
            feature[f"{name}_std"] = np.std(values)
            feature[f"{name}_rms"] = np.sqrt(np.mean(values**2))
            feature[f"{name}_iqr"] = iqr(values)
            feature[f"{name}_zero_crossings"] = zero_crossing_count(values - np.mean(values))
        
        # 單變數特徵
        for col in ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]:
            values = group[col].values
            feature[f"{col}_mean"] = np.mean(values)
            feature[f"{col}_std"] = np.std(values)
            feature[f"{col}_median"] = np.median(values) 
            feature[f"{col}_iqr"] = iqr(values)
            feature[f"{col}_skew"] = skew(values)
            feature[f"{col}_rms"] = np.sqrt(np.mean(values**2))
            feature[f"{col}_abs_change_mean"] = np.mean(np.abs(np.diff(values)))
            feature[f"{col}_abs_change_median"] = np.median(np.abs(np.diff(values)))
            

            # 加入 FFT 特徵
            feature.update(fft_features(values, col))

        # 雙變數互動特徵（協方差、相關性）
        sensor_pairs = [("Az", "Gz"), ("Ay", "Gy"), ("Ax", "Gx")]
        for col1, col2 in sensor_pairs:
            values1 = group[col1].values
            values2 = group[col2].values
            feature[f"{col1}_{col2}_corr"] = np.corrcoef(values1, values2)[0, 1]
            feature[f"{col1}_{col2}_cov"] = np.cov(values1, values2)[0, 1]

        # 額外的時間特徵
        for col in ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]:
            values = group[col].values
            # 峰值特徵
            feature[f"{col}_peak_count"] = np.sum(np.diff(np.sign(np.diff(values))) < 0)
            # 頻率帶能量
            fft_values = np.abs(fft(values))
            n = len(fft_values)
            feature[f"{col}_energy_high"] = np.sum(fft_values[n//2:]**2)
            # 熵和複雜度測量
            hist, _ = np.histogram(values, bins=20)
            probs = hist / np.sum(hist)
            feature[f"{col}_entropy"] = -np.sum(probs * np.log2(probs + 1e-10))

        # 爆發力指標
        feature["power_index"] = np.max(acc_mag) / np.mean(acc_mag)
        

        feature_list.append(feature)

    return pd.DataFrame(feature_list)

# 讀入資料
print("讀取訓練資料...")
df_train = pd.read_csv("train_data.csv")
features_df = extract_features(df_train, with_labels=True)

# 編碼標籤欄位
features_df['gender'] = features_df['gender'].apply(lambda x: 0 if x == 2 else 1)
features_df['hold racket handed'] = features_df['hold racket handed'].apply(lambda x: 0 if x == 2 else 1)
features_df['level'] = features_df['level'] - 2

print("訓練資料：")
print(features_df.shape)

# 測試資料  
print("\n讀取測試資料...")
df_test = pd.read_csv("test_data.csv")
test_features_df = extract_features(df_test, with_labels=False)

print("\n測試資料：")
print(test_features_df.shape)

# 準備資料
X = features_df.drop(columns=["unique_id", "gender", "hold racket handed", "play years", "level"])
y_gender = features_df["gender"] 
y_hand = features_df["hold racket handed"]
y_years = features_df["play years"]
y_level = features_df["level"]

# 將標籤轉換為 one-hot 編碼 (針對多分類問題)
y_years_cat = to_categorical(y_years)
y_level_cat = to_categorical(y_level)

# 切分資料集
X_train, X_val, y_train_gender, y_val_gender = train_test_split(X, y_gender, test_size=0.1, random_state=42)
_, _, y_train_hand, y_val_hand = train_test_split(X, y_hand, test_size=0.1, random_state=42)
_, _, y_train_years, y_val_years = train_test_split(X, y_years_cat, test_size=0.1, random_state=42)
_, _, y_train_level, y_val_level = train_test_split(X, y_level_cat, test_size=0.1, random_state=42)

# 標準化特徵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 準備測試資料
X_test_final = test_features_df.drop(columns=["unique_id"])
X_test_scaled = scaler.transform(X_test_final)


# 創建一個共用的特徵提取模型1
def create_base_model_1(input_shape):
    inputs = Input(shape=(input_shape,))

    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    return Model(inputs=inputs, outputs=x)

# 創建一個共用的特徵提取模型2
def create_base_model_2(input_shape):
    inputs = Input(shape=(input_shape,))

    x = Dense(64, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    return Model(inputs=inputs, outputs=x)

# 創建預測性別的模型 (二元分類)
def create_gender_model(input_shape):
    base_model = create_base_model_1(input_shape)
    inputs = Input(shape=(input_shape,))
    x = base_model(inputs)
    outputs = Dense(1, activation='sigmoid', name='gender_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# 創建預測持拍手的模型 (二元分類)
def create_hand_model(input_shape):
    base_model = create_base_model_1(input_shape)
    inputs = Input(shape=(input_shape,))
    x = base_model(inputs)
    outputs = Dense(1, activation='sigmoid', name='hand_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# 創建預測球齡的模型 (三分類)
def create_years_model(input_shape, num_classes=3):
    base_model = create_base_model_2(input_shape)
    inputs = Input(shape=(input_shape,))
    x = base_model(inputs)
    outputs = Dense(num_classes, activation='softmax', name='years_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 創建預測等級的模型 (四分類)
def create_level_model(input_shape, num_classes=4):
    base_model = create_base_model_2(input_shape)
    inputs = Input(shape=(input_shape,))
    x = base_model(inputs)
    outputs = Dense(num_classes, activation='softmax', name='level_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model


# 訓練性別模型
print("\n訓練性別模型...")
gender_model = create_gender_model(X_train_scaled.shape[1])
history_gender = gender_model.fit(
    X_train_scaled, y_train_gender,
    validation_data=(X_val_scaled, y_val_gender),
    epochs=17,
    batch_size=64,
    verbose=1
)

# 評估性別模型
y_val_gender_pred = (gender_model.predict(X_val_scaled) > 0.5).astype(int).flatten()
gender_accuracy = accuracy_score(y_val_gender, y_val_gender_pred)
gender_precision = precision_score(y_val_gender, y_val_gender_pred)
gender_recall = recall_score(y_val_gender, y_val_gender_pred)
gender_f1 = f1_score(y_val_gender, y_val_gender_pred)
gender_auc = roc_auc_score(y_val_gender, gender_model.predict(X_val_scaled).flatten())

print(f"性別預測 ROC AUC：{gender_auc:.4f}")
print(f"性別預測 Accuracy：{gender_accuracy:.4f}")
print(f"性別預測 Precision：{gender_precision:.4f}")
print(f"性別預測 Recall：{gender_recall:.4f}")
print(f"性別預測 F1-Score：{gender_f1:.4f}\n")



# 訓練持拍手模型
print("訓練持拍手模型...")
hand_model = create_hand_model(X_train_scaled.shape[1])
history_hand = hand_model.fit(
    X_train_scaled, y_train_hand,
    validation_data=(X_val_scaled, y_val_hand),
    epochs=17,
    batch_size=64,
    verbose=1
)

# 評估持拍手模型
y_val_hand_pred = (hand_model.predict(X_val_scaled) > 0.5).astype(int).flatten()
hand_accuracy = accuracy_score(y_val_hand, y_val_hand_pred)
hand_precision = precision_score(y_val_hand, y_val_hand_pred)
hand_recall = recall_score(y_val_hand, y_val_hand_pred)
hand_f1 = f1_score(y_val_hand, y_val_hand_pred)
hand_auc = roc_auc_score(y_val_hand, hand_model.predict(X_val_scaled).flatten())

print(f"持拍手預測 ROC AUC：{hand_auc:.4f}")
print(f"持拍手預測 Accuracy：{hand_accuracy:.4f}")
print(f"持拍手預測 Precision：{hand_precision:.4f}")
print(f"持拍手預測 Recall：{hand_recall:.4f}")
print(f"持拍手預測 F1-Score：{hand_f1:.4f}\n")


# 訓練球齡模型
print("訓練球齡模型...")
years_model = create_years_model(X_train_scaled.shape[1], y_train_years.shape[1])
history_years = years_model.fit(
    X_train_scaled, y_train_years,
    validation_data=(X_val_scaled, y_val_years),
    epochs=1,
    batch_size=32,
    verbose=1
)

# 評估球齡模型
y_val_years_pred = np.argmax(years_model.predict(X_val_scaled), axis=1)
y_val_years_true = np.argmax(y_val_years, axis=1)
years_accuracy = accuracy_score(y_val_years_true, y_val_years_pred)
years_precision = precision_score(y_val_years_true, y_val_years_pred, average='weighted')
years_recall = recall_score(y_val_years_true, y_val_years_pred, average='weighted')
years_f1 = f1_score(y_val_years_true, y_val_years_pred, average='weighted')

print(f"球齡預測 Accuracy：{years_accuracy:.4f}")
print(f"球齡預測 Precision：{years_precision:.4f}")
print(f"球齡預測 Recall：{years_recall:.4f}")
print(f"球齡預測 F1-Score：{years_f1:.4f}\n")


print("訓練等級模型...")
level_model = create_level_model(X_train_scaled.shape[1], y_train_level.shape[1])
history_level = level_model.fit(
    X_train_scaled, y_train_level,
    validation_data=(X_val_scaled, y_val_level),
    epochs=1,
    batch_size=32,
    verbose=1
)


# 評估等級模型
y_val_level_pred = np.argmax(level_model.predict(X_val_scaled), axis=1)
y_val_level_true = np.argmax(y_val_level, axis=1)
level_accuracy = accuracy_score(y_val_level_true, y_val_level_pred)
level_precision = precision_score(y_val_level_true, y_val_level_pred, average='weighted')
level_recall = recall_score(y_val_level_true, y_val_level_pred, average='weighted')
level_f1 = f1_score(y_val_level_true, y_val_level_pred, average='weighted')

print(f"等級預測 Accuracy：{level_accuracy:.4f}")
print(f"等級預測 Precision：{level_precision:.4f}")
print(f"等級預測 Recall：{level_recall:.4f}")
print(f"等級預測 F1-Score：{level_f1:.4f}\n")



# 對測試資料進行預測
print("對測試資料進行預測...")

# 性別預測
gender_probs = gender_model.predict(X_test_scaled).flatten()
gender_probs = np.round(gender_probs, 4)

# 持拍手預測
hand_probs = hand_model.predict(X_test_scaled).flatten()
hand_probs = np.round(hand_probs, 4)

# 球齡預測 (3分類)
years_probs = years_model.predict(X_test_scaled)
years_df = pd.DataFrame(
    np.round(years_probs, 4), columns=["play years_0", "play years_1", "play years_2"]
)

# 等級預測 (4分類)
level_probs = level_model.predict(X_test_scaled)
level_df = pd.DataFrame(
    np.round(level_probs, 4), columns=["level_2", "level_3", "level_4", "level_5"]
)

# 組合結果
output_df = pd.DataFrame({
    "unique_id": test_features_df["unique_id"],
    "gender": gender_probs,
    "hold racket handed": hand_probs
})

output_df = pd.concat([output_df, years_df, level_df], axis=1)

# 顯示結果前幾筆
print("預測結果前幾筆:")
print(output_df.head())



print("---------------------------")
print("| 指標      | 深度學習 性別 | 持拍手  | 球齡    | 等級    |")
print(f"| Accuracy  | {gender_accuracy:.4f}  | {hand_accuracy:.4f} | {years_accuracy:.4f} | {level_accuracy:.4f} |")
print(f"| Precision | {gender_precision:.4f}  | {hand_precision:.4f} | {years_precision:.4f} | {level_precision:.4f} |")
print(f"| Recall    | {gender_recall:.4f}  | {hand_recall:.4f} | {years_recall:.4f} | {level_recall:.4f} |")
print(f"| F1-Score  | {gender_f1:.4f}  | {hand_f1:.4f} | {years_f1:.4f} | {level_f1:.4f} |")
if gender_auc is not None:
    print(f"| AUC       | {gender_auc:.4f}  | {hand_auc:.4f} | N/A      | N/A      |")
print("---------------------------")

# 匯出為 CSV 檔案
output_df.to_csv("submission.csv", index=False, float_format="%.4f")
print("預測結果已儲存至 submission.csv")