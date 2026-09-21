import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import warnings

# ★ 關鍵修改 1：忽略 Pandas 處理時的一些警告訊息，讓終端機畫面乾淨
warnings.filterwarnings('ignore')

# 1. 讀取醫師提供的真實資料
df = pd.read_csv('dataset_20260918_1k.csv')

# 定義要放入計算的特徵欄位（包含性別、年齡與 6 項生命徵象）
features = ['hsex', 'age', 'SBP', 'DBP', 'RR', 'pulse', 'spo2', 'temperature']

# ★ 關鍵修改 2：資料清理 (非常重要！)
# 真實資料中，某些病人可能沒量血氧或呼吸，這些 NaN(空缺值) 會導致 KNN 演算法直接崩潰。
# 所以我們要把這些特徵或目標欄位有缺漏的資料列先剔除。
df = df.dropna(subset=features + ['degree', 'serious_event', 'admit_to'])

# 2. 資料切分
# 隨機抽取 50 筆作為測試資料（模擬新進病人），其餘作為歷史資料庫
test_df = df.sample(n=50, random_state=42)
train_df = df.drop(test_df.index)

# 初始化縮放器，統一將數值縮放到 0~1 之間，避免血壓等大數值主導距離
scaler = MinMaxScaler()
train_df[features] = scaler.fit_transform(train_df[features])

# ★ 關鍵修改 3：使用 .copy() 避免 Pandas 報錯 (SettingWithCopyWarning)
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
        'Test_Patient_ID': idx,
        'Degree': target_degree,
        'Serious_Risk(%)': round(serious_risk_pct, 2),
        'Admit_Distribution': admit_distribution
    })

# 檢視前 10 筆預測結果 (讓結果更整齊)
results_df = pd.DataFrame(results)
print("=== KNN 預測測試結果 ===")
print(results_df.to_string(index=False))

#Serious Risk%：如3小時內嚴重惡化風險為20%。系統在三級歷史病患中，找出生理數據與他最相似的 5 位前人，其中有 1 位（1/5 = 20%） 在 3 小時內出現了休克、急救等嚴重惡化事件。
#Admit Distribution%： 那 5 位最相似的歷史病患中，如有 4 位（80%） 住進一般病房（Ward），1 位（20%） 住進加護病房（ICU）。
#輸出的50筆資料的每一位會跟他最相似的五位(挑出距離最近、生理狀況最接近的 5 位)比對