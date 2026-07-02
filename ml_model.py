"""
ml_model.py

負責 Titanic 存活預測的機器學習邏輯：
1. 資料前處理（缺值填補、類別編碼、數值標準化）
2. 用 GridSearchCV 對兩種模型（LogisticRegression、RandomForest）調整超參數，
   並比較兩者，選出表現最好的當作最終模型。
3. 把最終模型（含前處理）用 joblib 存檔，之後可以直接載入來預測。
4. 提供單筆 / 批次（DataFrame）預測的函式。

這個檔案刻意跟 app.py（Flask 路由）分開，
好處是：ML 邏輯獨立好測試，之後想換模型、加演算法也只需要改這裡。
"""

import os
import json
from datetime import datetime

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# 1. 欄位設定
# ============================================================

# 拿來預測的特徵欄位（PassengerId / Name / Ticket / Cabin 對預測沒有直接幫助，先不用）
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "Survived"

# 模型與訓練資訊要存放的位置
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.joblib")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")


# ============================================================
# 2. 前處理器
# ============================================================

def build_preprocessor():
    """
    建立資料前處理的 ColumnTransformer。

    數值欄位（Pclass, Age, SibSp, Parch, Fare）：
        - 用中位數（median）補缺值（例如 Age 有缺值）
        - 標準化（StandardScaler），對 LogisticRegression 比較友善

    類別欄位（Sex, Embarked）：
        - 用眾數（most_frequent）補缺值（例如 Embarked 有 2 筆缺值）
        - One-Hot Encoding 轉成數值
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return preprocessor


# ============================================================
# 3. 候選模型與超參數搜尋範圍
# ============================================================
# 這裡示範兩種模型、各自 2 個超參數，你也可以再加其他模型或參數。

def get_candidates():
    """
    回傳一個 dict，每個 key 是模型名稱，value 是 (pipeline, param_grid)。
    GridSearchCV 會對每一組參數用交叉驗證（cv）算出平均準確率，
    最後選出「最佳參數組合」。
    """
    candidates = {}

    # ---- 候選 1：邏輯迴歸 ----
    logreg_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    logreg_param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],          # 正則化強度
        "classifier__solver": ["liblinear", "lbfgs"],  # 優化演算法
    }
    candidates["LogisticRegression"] = (logreg_pipeline, logreg_param_grid)

    # ---- 候選 2：隨機森林 ----
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])
    rf_param_grid = {
        "classifier__n_estimators": [100, 200],       # 樹的數量
        "classifier__max_depth": [None, 5, 10],        # 樹的最大深度
        "classifier__min_samples_split": [2, 5],        # 節點至少要有幾筆資料才能再切
    }
    candidates["RandomForest"] = (rf_pipeline, rf_param_grid)

    return candidates


# ============================================================
# 4. 訓練 + 超參數搜尋 + 選最佳模型
# ============================================================

def train_and_select_best(df, progress_callback=None):
    """
    輸入：df（從資料庫讀出來的 titanic 資料，DataFrame）
    流程：
        1. 切出 X（特徵）、y（標籤 Survived）
        2. 切出 train / test（test 用來最後驗證，不參與 GridSearch）
        3. 對每個候選模型跑 GridSearchCV（cv=5），得到最佳超參數
        4. 用 test set 評估每個候選模型「調完參數後」的準確率
        5. 選 test 準確率最高的當作最終模型
        6. 用「最佳超參數」在『全部資料』上重新訓練一次，
           這樣正式上線預測時，模型有看過最多資料
    回傳：一個 dict，裡面有最終模型 pipeline、以及所有訓練資訊
    """

    def report(msg):
        if progress_callback:
            progress_callback(msg)

    report("準備資料中...")
    df = df.copy()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = get_candidates()
    results = {}  # 存每個模型的搜尋結果，方便前端顯示比較

    for name, (pipeline, param_grid) in candidates.items():
        report(f"正在對 {name} 進行超參數搜尋（GridSearchCV）...")

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        # 用 test set（GridSearch 完全沒看過的資料）驗證最佳模型的表現
        test_pred = grid_search.best_estimator_.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_pred)

        results[name] = {
            "best_params": grid_search.best_params_,
            "cv_accuracy": grid_search.best_score_,
            "test_accuracy": test_accuracy,
        }
        report(f"{name} 完成，測試集準確率 = {test_accuracy:.4f}")

    # 選出 test_accuracy 最高的模型
    best_model_name = max(results, key=lambda name: results[name]["test_accuracy"])
    best_params = results[best_model_name]["best_params"]

    report(f"最佳模型是 {best_model_name}，用最佳超參數在全部資料上重新訓練最終模型...")

    # 用「最佳超參數」建立一個全新的 pipeline，並用「全部資料」重新 fit，
    # 這樣正式預測用的模型，是吃過最多資料訓練出來的版本。
    final_pipeline, _ = candidates[best_model_name]
    final_pipeline.set_params(**best_params)
    final_pipeline.fit(X, y)

    report("訓練完成")

    return {
        "pipeline": final_pipeline,
        "best_model_name": best_model_name,
        "best_params": best_params,
        "cv_accuracy": results[best_model_name]["cv_accuracy"],
        "test_accuracy": results[best_model_name]["test_accuracy"],
        "all_candidates_results": results,  # 兩個模型的完整比較結果
        "n_samples": len(df),
    }


# ============================================================
# 5. 模型存檔 / 讀檔
# ============================================================

def save_model(train_result):
    """
    把訓練結果存到 models/ 資料夾：
        - titanic_model.joblib：模型本身（含前處理 pipeline）
        - model_info.json：這次訓練的資訊（最佳參數、準確率、時間等），
          方便網頁重新整理後還能顯示上一次的訓練結果。
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(train_result["pipeline"], MODEL_PATH)

    info = {
        "best_model_name": train_result["best_model_name"],
        "best_params": train_result["best_params"],
        "cv_accuracy": train_result["cv_accuracy"],
        "test_accuracy": train_result["test_accuracy"],
        "all_candidates_results": train_result["all_candidates_results"],
        "n_samples": train_result["n_samples"],
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": MODEL_PATH,
    }

    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    return info


def load_model_info():
    """讀取上一次訓練的資訊（如果存在的話），伺服器重啟後也能顯示。"""
    if not os.path.exists(MODEL_INFO_PATH):
        return None
    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model():
    """讀取已存檔的模型（joblib），拿來做預測用。"""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


# ============================================================
# 6. 預測
# ============================================================

def predict_dataframe(pipeline, df):
    """
    輸入一個 DataFrame（至少要有 FEATURE_COLUMNS 這幾個欄位），
    回傳同一個 DataFrame，多加兩欄：
        - Survived_pred：預測是否存活（0 / 1）
        - Survival_probability：預測存活的機率（0~1）
    """
    df = df.copy()

    # 確保欄位齊全，缺的欄位補 NaN（前處理的 imputer 會處理缺值）
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    X = df[FEATURE_COLUMNS]

    proba = pipeline.predict_proba(X)[:, 1]  # 存活（Survived=1）的機率
    pred = (proba >= 0.5).astype(int)

    df["Survived_pred"] = pred
    df["Survival_probability"] = proba.round(4)

    return df
