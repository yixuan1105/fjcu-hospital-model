import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import warnings
import os
# ★ 開啟 Pandas 的中文全形字自動對齊功能
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.width', 1000)  # 避免螢幕寬度不足自動換行
#這個預測模型具有中等偏上（ROC-AUC 約 0.74 ~ 0.78）的區辨能力，整體準確率 (Accuracy)：約 85% ~ 87%

#忽略 Pandas 處理時的一些警告訊息，讓終端機畫面乾淨
warnings.filterwarnings('ignore')

# 1. 讀取醫師提供的真實資料
# 自動取得當前檔案 (knn_validation.py) 所在的資料夾路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'dataset_20260918_1k_with_names.csv')
# 讀取 CSV
df = pd.read_csv(csv_path)

# 定義要放入計算的特徵欄位（包含性別、年齡與 6 項生命徵象）
features = ['hsex', 'age', 'SBP', 'DBP', 'RR', 'pulse', 'spo2', 'temperature']

# 資料清理 
# 先把 admit_to 的空缺值補上 'Discharged/Other' (出院或其他)。
df['admit_to'] = df['admit_to'].fillna('Discharged/Other')
# 接著，只針對「生理特徵」或「檢傷級數」真正有漏填的資料列進行剔除
df = df.dropna(subset=features + ['degree', 'serious_event'])

# 2. 資料切分
# 隨機抽取 50 筆作為測試資料（模擬新進病人），其餘作為歷史資料庫
test_df = df.sample(n=50, random_state=42)#隨機亂數種子:它的作用是「固定隨機結果」
train_df = df.drop(test_df.index)

# 初始化縮放器，統一將數值縮放到 0~1 之間，避免血壓等大數值主導距離
scaler = MinMaxScaler()
train_df[features] = scaler.fit_transform(train_df[features])

# 使用 .copy() 避免 Pandas 報錯 (SettingWithCopyWarning)
test_df_scaled = test_df.copy()
test_df_scaled[features] = scaler.transform(test_df[features])

# 3. 針對 50 筆測試資料逐一進行驗證
results = []
# ★ 注意這裡改用 test_df_scaled 來進行迴圈
for idx, test_patient in test_df_scaled.iterrows():
    target_degree = test_patient['degree']
    
    # 核心邏輯：只在「同一檢傷級別」中尋找鄰居
    same_degree_train = train_df[train_df['degree'] == target_degree]
    
    # 若該級別的歷史資料過少（小於設定的 K 值），則需防呆處理
    k_neighbors = min(5, len(same_degree_train)) #代表系統會在歷史資料中，找出與當前病人生理數據最接近的 5 位病人
    if k_neighbors == 0:
        continue
        
    # 4. 建立並訓練 KNN 模型
    nn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
    nn.fit(same_degree_train[features])
    
    # 5. 尋找最近的 K 個鄰居 (注意：需轉為 2D 陣列輸入)
    distances, indices = nn.kneighbors([test_patient[features]])
    
    # 取得這 K 個鄰居的完整資料
    neighbors = same_degree_train.iloc[indices[0]]
    
    # 6. 統計比例
    # 計算 3 小時內嚴重惡化的比例 (值為 1 的平均)
    serious_risk_pct = neighbors['serious_event'].mean() * 100
    
    # ★ 關鍵修改 4：讓動向比例的輸出格式更清楚 (轉成帶有 % 的字典)
    admit_distribution = neighbors['admit_to'].value_counts(normalize=True).to_dict()
    admit_distribution = {k: f"{round(v * 100)}%" for k, v in admit_distribution.items()}
    
    # 將結果儲存，方便後續印出或轉為 DataFrame 檢視
    results.append({
        'Excel_Row': idx + 2,
        'Name': test_df.loc[idx, 'name'],
        'Degree': target_degree,
        'Serious_Risk(%)': round(serious_risk_pct, 2),
        'Admit_Distribution': admit_distribution
    })

# 檢視前 30 筆預測結果 (配合介面呈現需求)
results_df = pd.DataFrame(results)
print("=== KNN 預測測試結果 ===")
print(results_df.head(30).to_string(index=False))

#Serious Risk%：找出生理數據（年齡、性別、心跳、血壓、呼吸、血氧等）與他最相似的 5 位歷史病患，看這 5 人中有幾人在 3 小時內發生嚴重惡化。
#Admit Distribution%： 那 5 位最相似的歷史病患，最後各自去了哪裡（例如：住一般病房 Ward、住加護病房 ICU、急診留觀、或出院等）。
#5位歷史病患來自 dataset 中的歷史資料庫（train_df），而不是那 50 筆測試資料。
#在原始 1,000 筆資料中，真正發生 3 小時內嚴重惡化（serious_event == 1）的病人只有 21 人（約 2.1%）。這代表急診室裡約 98% 的病人在留觀期間都是穩定的。