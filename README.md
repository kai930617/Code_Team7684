# AI CUP 2025 春季賽：桌球智慧球拍資料的精準分析競賽

## 競賽程式碼/TEAM_7684/桌球智慧球拍資料的精準分析競賽

---
## 專案說明

```bash 
├── 39_Test_Dataset/            # testdata
│   ├── test_data/
│   └── test_info.csv
├── 39_Training_Dataset/        # traindata
│   ├── train_data/
│   └── train_info.csv
├── DL_model.py                 # 主模型訓練與預測
├── data_arrange.py             # 資料處理程式（包含切割與格式整理）
├── test_data.csv               # 處理後的測試資料
├── train_data.csv              # 處理後的訓練資料
├── submission.csv              # 最終提交預測檔案
├── README.md                   # 專案說明文件（本檔案）
├── 前分析/
│   ├── XGBOOST.ipynb           # 前分析使用模型與結果
│   ├── all_player_data/        # 前分析數據資料夾(儲存前處理2數據)
│   ├── converted_data/         # 前分析數據資料夾(儲存前處理3數據)
│   ├── merge_data/             # 前分析數據資料夾(儲存前處理1數據)
│   ├── 前分析特徵處理.ipynb     # 前分析前處理程式
│   ├── 孫穎莎(RSH)_data_file/  # 前分析使用自製資料集
│   └── 資訊統計.ipynb          # 前分析敘述統計
```

## 本專案使用 **Python 3.8** ，以下列出主要使用套件

**使用套件說明**

| 套件名稱         | 版本需求     |                              
|------------------|--------------|
| **TensorFlow**   | ≥ 2.0        | 
| **NumPy**        | ≥ 1.21       | 
| **Pandas**       | ≥ 1.3        | 
| **Scikit-learn** | ≥ 1.0        |
| **SciPy**        | ≥ 1.7        |
| **XGBoost**      | ≥ 1.5        |

## 程式碼執行流程

### 前分析

1. 依序執行 **前分析特徵處理.ipynb** 的chunk，將 **孫穎莎(RSH)_data_file/** 內資料集依序做處理會產生 **merge_data/** 、 **all_player_data/** 及 **merge_data/** (最終存放前處理的資料夾)
2. 執行**資訊統計.ipynb** 僅是查看前處理後的敘述統計(可跳過)
3. 依序執行 **XGBOOST.ipynb** 的chunk，訓練前分析使用之模型及產生分析圖表  

### 競賽預測主程式

1. 執行 **data_arrange.py** 的chunk，將 **39_Training_Dataset/**、**39_Test_Dataset/** 內資料做前處理會產生，**train_data.csv**、**test_data.csv**兩個資料夾
2. 執行 **DL_model.py** 為預測主程式，包含特徵前處理、訓練模型、預測部分
3. 產生 **submission.csv** 為最終提交預測檔案


