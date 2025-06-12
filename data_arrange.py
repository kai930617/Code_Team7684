import os
import pandas as pd

def process_data(info_path, data_path, output_file):
    # 讀取資料資訊檔案
    info = pd.read_csv(info_path)
    total_files = len(info)
    print(f"開始處理 {total_files} 個檔案...")

    # 初始化存放所有受試者數據的列表
    data = []

    # 逐個讀取 unique_id 對應的數據檔案
    for i, unique_id in enumerate(info["unique_id"], 1):
        file_path = os.path.join(data_path, f"{unique_id}.txt")
        
        # 顯示進度
        print(f"處理中... ({i}/{total_files}) - {unique_id}.txt", end='\r')
        
        if os.path.exists(file_path):
            # 讀取所有數據，並指定欄位名稱
            df = pd.read_csv(file_path, header=None, sep=r'\s+')
            df.columns = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]  # 設定欄位名稱
            
            # 移除所有感測數據為零的行
            df = df[(df != 0).any(axis=1)]  # 只保留有非零數據的行
            
            # 取得對應的資訊
            info_row = info[info["unique_id"] == unique_id].iloc[0]
            
            # 將每行數據與資訊結合
            for _, row in df.iterrows():
                combined_row = list(info_row) + row.tolist()  # 組合資訊資料和數據行
                data.append(combined_row)

    print()  # 換行
    
    # 轉為 DataFrame
    columns = info.columns.tolist() + ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]  # 設定欄位名稱
    data_df = pd.DataFrame(data, columns=columns)

    # 取得檔案的絕對路徑
    absolute_path = os.path.abspath(output_file)
    
    # 輸出到 CSV 檔案
    data_df.to_csv(output_file, index=False)
    
    # 檢查檔案是否成功建立
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # 轉換為 MB
        print(f"成功輸出 {len(data_df)} 行數據")
        print(f"儲存位置: {absolute_path}")
        print(f"檔案大小: {file_size:.2f} MB")
    else:
        print(f"檔案建立失敗: {absolute_path}")


# 設定資料夾路徑
train_info_path = "39_Training_Dataset/train_info.csv"
train_data_path = "39_Training_Dataset/train_data/"
test_info_path = "39_Test_Dataset/test_info.csv"
test_data_path = "39_Test_Dataset/test_data/"

print("=== 開始處理訓練資料 ===")
# 處理並輸出訓練資料
process_data(train_info_path, train_data_path, "train_data.csv")

print("\n=== 開始處理測試資料 ===")
# 處理並輸出測試資料
process_data(test_info_path, test_data_path, "test_data.csv")

print("\n=== 所有處理完成！ ===")
print(f"當前工作目錄: {os.getcwd()}")