# Titanic 乘客資料管理系統（RESTful API + 機器學習存活預測）

以 Flask 打造的 Titanic 乘客資料管理系統，包含：

- 乘客資料的 CRUD（新增／查詢／修改／刪除），前端用 Ajax（`fetch`）呼叫後端 RESTful API
- 一鍵訓練機器學習模型（自動比較多種演算法、調整超參數，選出最佳模型並存檔）
- 用訓練好的模型預測乘客是否存活（支援單筆輸入，或上傳 CSV 批次預測）

作業練習題目來源：[telunyang/restful_api_ajax](https://github.com/telunyang/restful_api_ajax)

---

## 功能總覽

| 類別           | 功能                                                                                                                       |
| -------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 資料管理       | 乘客列表（分頁、姓名搜尋）、新增、編輯、刪除                                                                               |
| 模型訓練       | 一鍵訓練，自動用 GridSearchCV 調整超參數，比較邏輯迴歸與隨機森林，選出最佳模型                                             |
| 訓練狀態       | 背景執行緒訓練，前端輪詢顯示「訓練中 / 已完成」，並顯示最佳超參數與準確率                                                  |
| 模型預測       | 單筆表單輸入預測存活機率；上傳 CSV 批次預測                                                                                |
| 資料分析視覺化 | 用 Chart.js 呈現人口統計、社會階級、家庭結構、登船港口、特徵工程（缺失值、頭銜萃取）、特徵相關性與模型特徵重要性等互動圖表 |

---

## 專案結構

```
titanic_restful_project/
├── app.py               # Flask 主程式：頁面路由 + RESTful API + ML API + 分析 API
├── ml_model.py           # 機器學習邏輯：前處理、GridSearchCV 訓練、存檔、預測
├── analysis.py            # 資料分析邏輯：各種 EDA 統計計算（人口統計/階級/家庭/港口/特徵工程/相關性）
├── init_db.py             # 讀取 titanic.csv，建立 SQLite 資料庫 my_db.db
├── titanic.csv             # 原始資料集
├── sample_passengers_for_prediction #多筆假資料測試集
├── requirements.txt         # 套件需求
├── templates/
│   ├── index.html             # 乘客列表首頁
│   ├── new.html                 # 新增乘客
│   ├── edit.html                 # 編輯乘客
│   ├── train.html                 # 訓練模型頁面
│   ├── predict.html                # 預測存活頁面
│   └── analysis.html                # 資料分析視覺化儀表板
├── models/                # 訓練完成後自動產生（模型檔、訓練資訊）
│   ├── titanic_model.joblib
│   └── model_info.json
└── my_db.db               # 執行 init_db.py 後自動產生（不進版控）
```

---

## 安裝與執行

### 1. 安裝套件

```bash
pip install -r requirements.txt
```

### 2. 初始化資料庫

用 `titanic.csv` 建立 SQLite 資料庫（`my_db.db`）：

```bash
python init_db.py
```

### 3. 啟動伺服器

```bash
python app.py
```

啟動後開啟瀏覽器： http://127.0.0.1:5000

### 4. 操作流程

1. 首頁可以瀏覽 / 搜尋 / 新增 / 編輯 / 刪除乘客資料
2. 點選「訓練模型」，按下「開始訓練模型」→ 等待背景訓練完成（約數十秒），畫面會自動顯示最佳模型與超參數
3. 點選「預測存活」，可以：
   - 填單筆乘客資料表單，取得存活與否＋存活機率
   - 上傳 CSV（欄位需包含 `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked`）批次預測
4. 點選「資料分析」，可以看到依目前資料庫即時計算的探索性資料分析（EDA）圖表：人口統計、社會階級、家庭結構、登船港口、特徵工程、特徵相關性；如果已經訓練過模型，還會顯示模型的特徵重要性

---

## RESTful API 文件

### 乘客資料 CRUD

| Method | Endpoint                                    | 說明                               |
| ------ | ------------------------------------------- | ---------------------------------- |
| GET    | `/api/passengers?page=&per_page=&search=` | 取得乘客列表（分頁、可依姓名搜尋） |
| GET    | `/api/passengers/<id>`                    | 取得單一乘客                       |
| POST   | `/api/passengers`                         | 新增乘客（JSON body）              |
| PUT    | `/api/passengers/<id>`                    | 修改乘客（JSON body）              |
| DELETE | `/api/passengers/<id>`                    | 刪除乘客                           |

### 機器學習

| Method | Endpoint            | 說明                                                                         |
| ------ | ------------------- | ---------------------------------------------------------------------------- |
| POST   | `/api/ml/train`   | 觸發一次背景訓練（202 表示已開始；訓練中再次呼叫回 409）                     |
| GET    | `/api/ml/status`  | 查詢目前訓練狀態、最佳超參數與準確率                                         |
| POST   | `/api/ml/predict` | 預測。JSON body = 單筆；`multipart/form-data`（欄位名 `file`）= CSV 批次 |

### 資料分析

| Method | Endpoint                  | 說明                                                                            |
| ------ | ------------------------- | ------------------------------------------------------------------------------- |
| GET    | `/api/analysis/summary` | 一次回傳所有 EDA 統計結果（人口統計/階級/家庭/港口/特徵工程/相關性/特徵重要性） |

**單筆預測範例：**

```bash
curl -X POST http://127.0.0.1:5000/api/ml/predict \
  -H "Content-Type: application/json" \
  -d '{"Pclass":1,"Sex":"female","Age":29,"SibSp":0,"Parch":0,"Fare":100,"Embarked":"S"}'
```

回傳：

```json
{
  "message": "ok",
  "item": {
    "Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0, "Parch": 0,
    "Fare": 100, "Embarked": "S",
    "Survived_pred": 1,
    "Survival_probability": 0.9556
  }
}
```

**CSV 批次預測範例：**

```bash
curl -X POST http://127.0.0.1:5000/api/ml/predict \
  -F "file=@sample_passengers.csv"
```

---

## 機器學習模型說明

### 特徵欄位

`Pclass, Sex, Age, SibSp, Parch, Fare, Embarked`（`Name`、`Ticket`、`Cabin`、`PassengerId` 不用於訓練）

### 資料前處理

- 數值欄位（`Pclass, Age, SibSp, Parch, Fare`）：中位數補缺值 → 標準化（`StandardScaler`）
- 類別欄位（`Sex, Embarked`）：眾數補缺值 → One-Hot Encoding

### 候選模型與超參數搜尋範圍

| 模型                       | 調整的超參數                                                                              |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| `LogisticRegression`     | `C`: [0.01, 0.1, 1, 10]、`solver`: [liblinear, lbfgs]                                 |
| `RandomForestClassifier` | `n_estimators`: [100, 200]、`max_depth`: [None, 5, 10]、`min_samples_split`: [2, 5] |

兩個模型都用 `GridSearchCV`（5-fold 交叉驗證）找出各自的最佳超參數，再用「訓練時完全沒看過」的測試集（20% 資料）評估，選出測試集準確率較高的當作最終模型，並用該組最佳超參數在**全部資料**上重新訓練一次，存到 `models/titanic_model.joblib`。

### 訓練資料來源

訓練時是即時從 SQLite 的 `titanic` 資料表讀取（而不是直接讀 `titanic.csv`），所以如果在網頁上新增/修改/刪除過乘客資料，重新訓練會反映最新的資料庫內容。

---

---

## 資料分析方法（EDA）

`analysis.py` 針對目前資料庫（可能已經被 CRUD 操作修改過）即時計算以下統計，全部透過 `/api/analysis/summary` 一次回傳：

| 面向       | 計算內容                                                                                                                            |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 人口統計   | 性別存活率、年齡分組（兒童/青少年/青年/中年/老年）存活率                                                                            |
| 社會階級   | 艙等存活率、各艙等平均票價、票價四分位區間存活率                                                                                    |
| 家庭結構   | 依同行人數（`SibSp+Parch+1`）分組存活率、找出存活率最高的家庭規模、獨自旅行 vs 攜家帶眷比較                                       |
| 登船港口   | 各登船港口存活率                                                                                                                    |
| 特徵工程   | 各欄位缺失值統計；用正規表示式從`Name` 萃取頭銜（Mr/Mrs/Miss/Master/Rare）並比較存活率；是否有艙位紀錄（`Cabin`）與存活率的關係 |
| 特徵相關性 | `Survived, Pclass, Age, SibSp, Parch, Fare, FamilySize` 的皮爾森相關係數矩陣                                                      |
| 模型解釋   | 如果已訓練過模型，顯示該模型判斷「哪些特徵最重要」（RandomForest 用 Gini 重要性，LogisticRegression 用標準化係數絕對值）            |

前端頁面（`templates/analysis.html`）用 [Chart.js](https://www.chartjs.org/)（CDN 載入）畫互動式長條圖，每個圖表都附上依實際數據動態產生的重點結論文字。

---

## 已知限制

- 範例中的模型與超參數搜尋範圍是示範用途，數量不多，實際準確率沒有特別調優
- 訓練狀態存在記憶體中的全域變數，僅適合單一 process 開發環境使用；正式多人多程序環境需要改用資料庫或 Redis 等外部儲存
- `app.py` 用 `debug=True` 啟動，正式上線前應關閉

---
