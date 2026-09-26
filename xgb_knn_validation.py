import os # 操作系統檔案路徑管理
import warnings # 控制警告訊息的顯示
import numpy as np # 高效的高維度數學運算工具
import pandas as pd # 資料表處理工具
from xgboost import XGBClassifier # 改用 XGBoost 分類器
from sklearn.neighbors import NearestNeighbors # K-近鄰演算法
from sklearn.preprocessing import MinMaxScaler # 資料最大最小標準化工具
# ★ 開啟 Pandas 的中文全形字自動對齊功能
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.width', 1000)  # 避免螢幕寬度不足自動換行

# 忽略警告訊息
warnings.filterwarnings('ignore')

# 1. 讀取與清理資料
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'dataset_20260918_1k_with_names.csv')
df = pd.read_csv(csv_path)

# 特徵欄位
features = ['hsex', 'age', 'degree', 'SBP', 'DBP', 'RR', 'pulse', 'spo2', 'temperature']

# 資料清理：填補未住院病人的去向
df['admit_to'] = df['admit_to'].fillna('Discharged/Other')
df = df.dropna(subset=features + ['serious_event'])

# 2. 切分訓練集與測試集 (取 50 筆模擬新病人)
test_df = df.sample(n=50, random_state=42)
train_df = df.drop(test_df.index)

# 3. 計算正負樣本比例，設定 scale_pos_weight 以處理急診不平衡資料
neg_count = (train_df['serious_event'] == 0).sum()
pos_count = (train_df['serious_event'] == 1).sum()
scale_pos_weight_value = neg_count / pos_count  # 負樣本/正樣本比例（約 46.5）

# 訓練 XGBoost 模型用於「病情惡化風險預測」
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,                         # 樹不宜太深，避免在小數據集上過擬合
    learning_rate=0.05,                  # 學習率設為 0.05 進行平滑更新
    scale_pos_weight=scale_pos_weight_value, # 關鍵：針對極少數重症給予更高的懲罰權重
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(train_df[features], train_df['serious_event'])

# 預測測試集的惡化機率 (轉換為百分比 %)
test_xgb_probs = xgb_model.predict_proba(test_df[features])[:, 1] * 100

# 4. 核心二：建立 KNN 模型用於「尋找 Top 5 相似歷史個案」
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df[features])
test_scaled = scaler.transform(test_df[features])

knn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn_model.fit(train_scaled)

# 5. 彙整 50 筆測試資料的綜合預測結果
results = []
for i, (idx, test_patient) in enumerate(test_df.iterrows()):
    # 取得 KNN 最近的 5 個鄰居
    distances, indices = knn_model.kneighbors([test_scaled[i]])
    neighbors = train_df.iloc[indices[0]]
    
    # 計算鄰居的去向分布
    admit_dist = neighbors['admit_to'].value_counts(normalize=True).to_dict()
    admit_dist = {k: f"{round(v * 100)}%" for k, v in admit_dist.items()}
    
    results.append({
        'Excel_Row': idx + 2,
        'Name': test_df.loc[idx, 'name'],
        'Degree': test_patient['degree'],
        'Serious_Risk(%)': round(test_xgb_probs[i], 2), # XGBoost 預測之惡化機率
        'Admit_Distribution': admit_dist                  # KNN 找出的相似個案動向
    })

# 6. 輸出前 30 筆測試結果給前端/介面顯示
results_df = pd.DataFrame(results)
print("=== XGBoost + KNN 預測結果 ===")
print(results_df.head(30).to_string(index=False))