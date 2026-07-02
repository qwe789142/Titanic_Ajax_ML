"""
analysis.py

負責 Titanic 資料集的探索性資料分析（EDA），把 pandas 計算出來的統計結果
整理成前端 Chart.js 可以直接使用的 JSON 格式。

刻意跟 app.py（路由）、ml_model.py（訓練/預測）分開，
維持「資料庫 -> 統計計算 -> 訓練 / 預測」三個模組各自獨立的架構。
"""

import re

import pandas as pd
import numpy as np


# ============================================================
# 小工具
# ============================================================

def _group_survival(df, group_series, order=None, min_count=1):
    """
    共用的「依類別分組計算存活率」工具。

    輸入：
        df: 原始 DataFrame（要有 Survived 欄位）
        group_series: 分組用的 Series（例如性別、艙等、年齡層…）
        order: 想要的分類順序（例如 ["女性", "男性"]），沒給就依名稱排序
        min_count: 至少要有幾筆資料才會被列入結果（避免樣本數太少的雜訊）

    輸出：
        { "labels": [...], "counts": [...], "survived_counts": [...], "survival_rates": [...] }
    """
    tmp = df.assign(_group=group_series)
    grouped = tmp.groupby("_group", observed=True)["Survived"].agg(["count", "sum"]).reset_index()
    grouped["rate"] = (grouped["sum"] / grouped["count"] * 100).round(1)
    grouped = grouped[grouped["count"] >= min_count]

    if order:
        grouped["_group"] = pd.Categorical(grouped["_group"], categories=order, ordered=True)
        grouped = grouped.sort_values("_group")
    else:
        grouped = grouped.sort_values("_group")

    return {
        "labels": grouped["_group"].astype(str).tolist(),
        "counts": grouped["count"].astype(int).tolist(),
        "survived_counts": grouped["sum"].astype(int).tolist(),
        "survival_rates": grouped["rate"].tolist(),
    }


def _overall_rate(df):
    total = len(df)
    survived = int(df["Survived"].sum())
    rate = round(survived / total * 100, 1) if total else 0
    return total, survived, rate


# ============================================================
# 1. 總覽 KPI
# ============================================================

def compute_overview(df):
    total, survived, rate = _overall_rate(df)
    return {
        "total": total,
        "survived": survived,
        "died": total - survived,
        "survival_rate": rate,
    }


# ============================================================
# 2. 人口統計：性別 / 年齡
# ============================================================

SEX_LABELS = {"female": "女性", "male": "男性"}


def compute_gender_analysis(df):
    sex_display = df["Sex"].map(SEX_LABELS).fillna(df["Sex"])
    result = _group_survival(df, sex_display, order=["女性", "男性"])

    insight = None
    if "女性" in result["labels"] and "男性" in result["labels"]:
        female_rate = result["survival_rates"][result["labels"].index("女性")]
        male_rate = result["survival_rates"][result["labels"].index("男性")]
        insight = (
            f"女性存活率為 {female_rate}%，男性存活率為 {male_rate}%，"
            f"相差 {round(female_rate - male_rate, 1)} 個百分點，"
            f"與「女士與兒童優先」的救援原則相符。"
        )
    return {**result, "insight": insight}


def _age_group(age):
    if pd.isna(age):
        return "年齡未知"
    if age <= 12:
        return "兒童 (0-12)"
    if age <= 18:
        return "青少年 (13-18)"
    if age <= 35:
        return "青年 (19-35)"
    if age <= 60:
        return "中年 (36-60)"
    return "老年 (60+)"


AGE_GROUP_ORDER = ["兒童 (0-12)", "青少年 (13-18)", "青年 (19-35)", "中年 (36-60)", "老年 (60+)", "年齡未知"]


def compute_age_analysis(df):
    age_group_series = df["Age"].apply(_age_group)
    result = _group_survival(df, age_group_series, order=AGE_GROUP_ORDER)

    insight = None
    known = [(l, r) for l, r in zip(result["labels"], result["survival_rates"]) if l != "年齡未知"]
    if known:
        best_group, best_rate = max(known, key=lambda x: x[1])
        insight = f"「{best_group}」的存活率最高，達 {best_rate}%。"

    return {**result, "insight": insight}


# ============================================================
# 3. 社會階級與資源分配：艙等 / 票價
# ============================================================

PCLASS_LABELS = {1: "頭等艙 (1)", 2: "二等艙 (2)", 3: "三等艙 (3)"}


def compute_class_analysis(df):
    pclass_display = df["Pclass"].map(PCLASS_LABELS).fillna(df["Pclass"].astype(str))
    result = _group_survival(df, pclass_display, order=list(PCLASS_LABELS.values()))

    insight = None
    if len(result["labels"]) >= 2:
        best_idx = int(np.argmax(result["survival_rates"]))
        worst_idx = int(np.argmin(result["survival_rates"]))
        insight = (
            f"{result['labels'][best_idx]} 存活率最高（{result['survival_rates'][best_idx]}%），"
            f"{result['labels'][worst_idx]} 存活率最低（{result['survival_rates'][worst_idx]}%），"
            f"顯示艙等（社會階級）與獲救機會有明顯關聯。"
        )

    # 附帶：各艙等的平均票價，佐證「艙等」與「票價」的關係
    fare_by_class = df.groupby(df["Pclass"].map(PCLASS_LABELS))["Fare"].mean().round(1)
    fare_by_class = fare_by_class.reindex(list(PCLASS_LABELS.values())).dropna()

    return {
        **result,
        "insight": insight,
        "avg_fare_labels": fare_by_class.index.tolist(),
        "avg_fare_values": fare_by_class.values.tolist(),
    }


def compute_fare_analysis(df):
    fare = df["Fare"].fillna(df["Fare"].median())
    try:
        bins = pd.qcut(fare, 4, duplicates="drop")
    except ValueError:
        # 資料量太少或票價都一樣時，qcut 可能會失敗，改用固定區間
        bins = pd.cut(fare, bins=[-0.01, 10, 25, 50, fare.max() + 1])

    # 把區間轉成好讀的字串標籤，例如 "$0 - $8"
    labels = bins.apply(lambda iv: f"${max(iv.left, 0):.0f} - ${iv.right:.0f}")
    order = sorted(labels.unique(), key=lambda s: float(s.split(" - ")[0].replace("$", "")))

    result = _group_survival(df, labels, order=order)

    insight = None
    if result["labels"]:
        best_idx = int(np.argmax(result["survival_rates"]))
        insight = f"票價區間「{result['labels'][best_idx]}」的存活率最高（{result['survival_rates'][best_idx]}%），票價越高的乘客，通常艙等也越高、存活率也越高。"

    return {**result, "insight": insight}


# ============================================================
# 4. 家庭結構與同行人數
# ============================================================

def _family_size_label(size):
    if size >= 8:
        return "8+"
    return str(size)


def compute_family_analysis(df):
    family_size = df["SibSp"] + df["Parch"] + 1
    labels = family_size.apply(_family_size_label)
    order = [str(i) for i in range(1, 8)] + ["8+"]

    result = _group_survival(df, labels, order=order)

    # 找出「有足夠樣本數（>=5 人）」中存活率最高的家庭規模，避免單一極端值誤導結論
    candidates = [
        (l, r, c) for l, r, c in zip(result["labels"], result["survival_rates"], result["counts"])
        if c >= 5
    ]
    insight = None
    if candidates:
        best_size, best_rate, best_count = max(candidates, key=lambda x: x[1])
        insight = f"同行人數為 {best_size} 人時存活率最高，達 {best_rate}%（共 {best_count} 筆樣本）。獨自旅行（1 人）或人數過多的大家庭，存活率通常較低。"

    # 附帶：獨自旅行 vs 攜帶家人 的簡單對照
    alone_display = family_size.apply(lambda s: "獨自旅行" if s == 1 else "攜帶家人")
    alone_result = _group_survival(df, alone_display, order=["獨自旅行", "攜帶家人"])

    return {
        **result,
        "insight": insight,
        "alone_vs_family": alone_result,
    }


# ============================================================
# 5. 登船地點
# ============================================================

EMBARKED_LABELS = {"S": "南安普敦 (S)", "C": "瑟堡 (C)", "Q": "皇后鎮 (Q)"}


def compute_embarked_analysis(df):
    embarked_display = df["Embarked"].map(EMBARKED_LABELS).fillna("未知")
    order = list(EMBARKED_LABELS.values()) + ["未知"]
    result = _group_survival(df, embarked_display, order=order)

    insight = None
    known = [(l, r) for l, r in zip(result["labels"], result["survival_rates"]) if l != "未知"]
    if known:
        best_group, best_rate = max(known, key=lambda x: x[1])
        insight = f"從「{best_group}」登船的乘客存活率最高，達 {best_rate}%。這可能與該港口上船旅客的艙等組成有關。"

    return {**result, "insight": insight}


# ============================================================
# 6. 特徵工程：缺失值 / 頭銜萃取 / 是否有艙位紀錄
# ============================================================

def compute_missing_value_summary(df):
    total = len(df)
    columns = ["Age", "Cabin", "Embarked"]
    items = []
    for col in columns:
        if col not in df.columns:
            continue
        missing = int(df[col].isna().sum())
        items.append({
            "column": col,
            "missing_count": missing,
            "missing_percent": round(missing / total * 100, 1) if total else 0,
        })
    return items


# 從姓名萃取頭銜，並把少見的頭銜合併成「Rare」，是 Titanic 資料集最經典的特徵工程範例
_TITLE_MAP = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "Countess": "Rare", "the Countess": "Rare", "Capt": "Rare",
    "Col": "Rare", "Don": "Rare", "Dona": "Rare", "Dr": "Rare", "Major": "Rare",
    "Rev": "Rare", "Sir": "Rare", "Jonkheer": "Rare",
}
_MAIN_TITLES = ["Mr", "Mrs", "Miss", "Master"]


def extract_title(name):
    match = re.search(r",\s*([^.]*)\.", str(name))
    if not match:
        return "Rare"
    raw_title = match.group(1).strip()
    if raw_title in _MAIN_TITLES:
        return raw_title
    return _TITLE_MAP.get(raw_title, "Rare")


def compute_title_analysis(df):
    titles = df["Name"].apply(extract_title)
    order = _MAIN_TITLES + ["Rare"]
    result = _group_survival(df, titles, order=order, min_count=1)

    insight = (
        "從姓名中萃取出的頭銜（Mr / Mrs / Miss / Master / 其他）能反映性別與年齡（例如 Master 代表男孩），"
        "存活率的差異也明顯比單純用性別分類更細緻。"
    )
    return {**result, "insight": insight}


def compute_cabin_analysis(df):
    has_cabin = df["Cabin"].notna().map({True: "有艙位紀錄", False: "無艙位紀錄"})
    result = _group_survival(df, has_cabin, order=["有艙位紀錄", "無艙位紀錄"])

    insight = None
    if "有艙位紀錄" in result["labels"] and "無艙位紀錄" in result["labels"]:
        with_rate = result["survival_rates"][result["labels"].index("有艙位紀錄")]
        without_rate = result["survival_rates"][result["labels"].index("無艙位紀錄")]
        insight = (
            f"有艙位紀錄的乘客存活率為 {with_rate}%，明顯高於沒有紀錄的 {without_rate}%。"
            f"這很可能不是「艙位本身」造成的，而是頭等艙乘客的艙位紀錄本來就比較完整（資料缺失本身也是一種訊號）。"
        )
    return {**result, "insight": insight}


# ============================================================
# 7. 數值特徵相關性
# ============================================================

def compute_correlation_matrix(df):
    numeric_df = pd.DataFrame({
        "Survived": df["Survived"],
        "Pclass": df["Pclass"],
        "Age": df["Age"],
        "SibSp": df["SibSp"],
        "Parch": df["Parch"],
        "Fare": df["Fare"],
        "FamilySize": df["SibSp"] + df["Parch"] + 1,
    })
    corr = numeric_df.corr(numeric_only=True).round(2)
    return {
        "columns": corr.columns.tolist(),
        "matrix": corr.values.tolist(),
    }


# ============================================================
# 8. 模型特徵重要性（如果已經訓練過模型）
# ============================================================

def compute_feature_importance(pipeline):
    """
    從已訓練好的 pipeline 取出「哪些特徵對預測結果影響最大」。
    - RandomForest: 用 feature_importances_（Gini 重要性）
    - LogisticRegression: 用係數絕對值（標準化過的特徵，數值可互相比較大小）
    回傳 None 表示還沒有訓練好的模型。
    """
    if pipeline is None:
        return None

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        feature_names = preprocessor.get_feature_names_out()
        # 把 "num__Age"、"cat__Sex_female" 這種前綴去掉，顯示比較好讀
        display_names = [name.split("__", 1)[-1] for name in feature_names]

        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
            method = "RandomForest 的 Gini 重要性"
        elif hasattr(classifier, "coef_"):
            importances = np.abs(classifier.coef_[0])
            method = "LogisticRegression 的標準化係數絕對值"
        else:
            return None

        order = np.argsort(importances)[::-1]
        return {
            "labels": [display_names[i] for i in order],
            "importances": [round(float(importances[i]), 4) for i in order],
            "method": method,
        }
    except Exception:
        return None


# ============================================================
# 9. 整合：一次算出所有分析結果
# ============================================================

def compute_full_analysis(df, pipeline=None):
    df = df.copy()

    return {
        "overview": compute_overview(df),
        "gender": compute_gender_analysis(df),
        "age": compute_age_analysis(df),
        "pclass": compute_class_analysis(df),
        "fare": compute_fare_analysis(df),
        "family": compute_family_analysis(df),
        "embarked": compute_embarked_analysis(df),
        "missing_values": compute_missing_value_summary(df),
        "title": compute_title_analysis(df),
        "cabin": compute_cabin_analysis(df),
        "correlation": compute_correlation_matrix(df),
        "feature_importance": compute_feature_importance(pipeline),
    }
