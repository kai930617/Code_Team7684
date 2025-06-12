# AI CUP 2025 春季賽：桌球智慧球拍資料的精準分析競賽

## 競賽程式碼/TEAM_7684/桌球智慧球拍資料的精準分析競賽

---
## 專案說明

### 檔案所需結構樹狀圖
```bash
### 本專案內不提供
├── 39_Test_Dataset/            # testdata(主辦方提供，未經允許本專案內無附上)
│   ├── test_data/
│   └── test_info.csv
├── 39_Training_Dataset/        # traindata(主辦方提供，未經允許本專案內無附上)
│   ├── train_data/
│   └── train_info.csv
├── test_data.csv               # 處理後會產生的測試資料
├── train_data.csv              # 處理後的產生的訓練資料
### 本專案內提供
├── README.md                   # 專案說明文件（本檔案）
├── 前分析/
│   ├── XGBOOST.ipynb           # 前分析使用模型與結果
│   ├── all_player_data/        # 前分析數據資料夾(儲存前處理2數據)
│   ├── converted_data/         # 前分析數據資料夾(儲存前處理3數據)
│   ├── merge_data/             # 前分析數據資料夾(儲存前處理1數據)
│   ├── 前分析特徵處理.ipynb     # 前分析前處理程式
│   ├── 孫穎莎(RSH)_data_file/  # 前分析使用自製資料集
│   └── 資訊統計.ipynb          # 前分析敘述統計
├── data_arrange.py             # 資料前處理程式（包含整合與格式整理）
├── DL_model.py                 # 主模型訓練與預測
├── submission.csv              # 最終提交預測檔案

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
| **matplotlib**  | ≥ 3.2        |

## 程式碼執行流程

### 前分析(不影響預測，僅做競賽前分析)

1. 依序執行 **前分析特徵處理.ipynb** 的chunk，將 **孫穎莎(RSH)_data_file/** 內資料集依序做處理會依序將結果存放到 **merge_data/** 、**all_player_data/** 及 **merge_data/** (最終存放前處理的資料夾)
3. 執行**資訊統計.ipynb** 僅是查看前處理後的敘述統計(可跳過)
4. 依序執行 **XGBOOST.ipynb** 的chunk，訓練前分析使用之模型

### 競賽預測主程式

1. 執行 **data_arrange.py** 的chunk，將 **39_Training_Dataset/**、**39_Test_Dataset/** 內資料做前處理，會產生**train_data.csv**、**test_data.csv**兩個資料夾為處理後的資料
2. 執行 **DL_model.py** 為預測主程式，包含特徵工程、訓練模型、預測部分
3. 產生 **submission.csv** 為最終提交預測檔案


