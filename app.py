import sqlite3
import threading
import pandas as pd
from flask import Flask, jsonify, request, render_template

import ml_model  # 我們自己寫的機器學習模組（訓練 / 存檔 / 預測）
import analysis  # 我們自己寫的資料分析模組（EDA 統計計算）

app = Flask(__name__)

# ============================================================
# 1. 全域讀取資料庫
# ============================================================

DATABASE = "my_db.db"

# 這裡我們直接在全域讀取資料庫，這樣在每個 route 就可以直接使用 db 來存取資料庫了。
db = sqlite3.connect(DATABASE, check_same_thread=False)

# 讓我們在讀取資料庫時，可以直接用 row["欄位名稱"] 的方式來存取資料，
# 而不是 row[0]、row[1] 這樣的 index。
db.row_factory = sqlite3.Row


# ============================================================
# 2. 小工具：把 SQLite Row 轉成 dict
# ============================================================

def row_to_dict(row):
    return dict(row)


def clean_nan_for_json(records):
    """
    把 pandas/numpy 的 NaN 換成 Python 的 None。

    這裡刻意不是用 df.where(pd.notnull(df), None)，
    因為數值欄位（float64）的 dtype 不會因為 where() 就變成 object，
    所以 NaN 會被留下來 —— 而 NaN 不是合法的 JSON 值，
    瀏覽器的 JSON.parse() 遇到它會直接丟出錯誤。
    改成在「轉成一般 Python dict 之後」逐一檢查每個值最保險。
    """
    cleaned = []
    for record in records:
        cleaned.append({
            key: (None if pd.isna(value) else value)
            for key, value in record.items()
        })
    return cleaned


# ============================================================
# 2-1. 機器學習訓練狀態（全域變數）
# ============================================================
#
# 因為訓練是在「背景執行緒」跑的，Flask 主執行緒沒辦法直接回傳結果，
# 所以我們用一個全域的 dict 記錄目前的訓練狀態，
# 前端頁面用 setInterval 每隔一段時間打 /api/ml/status 來看狀態有沒有變化（polling）。
#
# status 可能的值：
#   "idle"    -> 還沒訓練過，或是上次伺服器重啟後還沒點過訓練
#   "running" -> 正在訓練中
#   "done"    -> 訓練完成
#   "error"   -> 訓練過程發生錯誤

training_state = {
    "status": "idle",
    "message": "",
    "best_model_name": None,
    "best_params": None,
    "cv_accuracy": None,
    "test_accuracy": None,
    "all_candidates_results": None,
    "n_samples": None,
    "trained_at": None,
    "error": None,
}

# 因為 training_state 會被「背景訓練執行緒」和「處理 HTTP 請求的執行緒」同時存取，
# 用 Lock 避免兩邊同時讀寫造成資料錯亂。
training_lock = threading.Lock()


def _load_previous_model_info():
    """伺服器啟動時，如果 models/model_info.json 已經存在（表示之前訓練過），
    就把上次的訓練結果讀進 training_state，這樣重新整理網頁還是看得到結果。"""
    info = ml_model.load_model_info()
    if info is not None:
        with training_lock:
            training_state["status"] = "done"
            training_state["message"] = "（讀取上次訓練結果）"
            training_state.update({k: info[k] for k in [
                "best_model_name", "best_params", "cv_accuracy",
                "test_accuracy", "all_candidates_results",
                "n_samples", "trained_at",
            ]})


_load_previous_model_info()


def _run_training_in_background():
    """這個函式會在背景執行緒被呼叫，實際執行「讀資料 -> 訓練 -> 存檔」的流程。"""
    try:
        # 從資料庫讀取目前的 titanic 資料表(學生可能已經新增/修改/刪除過資料，
        # 所以這裡是讀「目前資料庫的最新狀態」，而不是原始的 titanic.csv)
        df = pd.read_sql_query("SELECT * FROM titanic", db)

        def progress_callback(msg):
            with training_lock:
                training_state["message"] = msg

        result = ml_model.train_and_select_best(df, progress_callback=progress_callback)
        info = ml_model.save_model(result)

        with training_lock:
            training_state["status"] = "done"
            training_state["message"] = "訓練完成"
            training_state.update({k: info[k] for k in [
                "best_model_name", "best_params", "cv_accuracy",
                "test_accuracy", "all_candidates_results",
                "n_samples", "trained_at",
            ]})
            training_state["error"] = None

    except Exception as e:
        with training_lock:
            training_state["status"] = "error"
            training_state["message"] = "訓練失敗"
            training_state["error"] = str(e)


# ============================================================
# 3. 前端頁面 Routes
# ============================================================

# 首頁
@app.route("/")
def index_page():
    return render_template("index.html")

# 新增乘客頁面
@app.route("/passengers/new")
def new_passenger_page():
    return render_template("new.html")

# 編輯乘客頁面
@app.route("/passengers/<int:passenger_id>/edit")
def edit_passenger_page(passenger_id):
    return render_template("edit.html", passenger_id=passenger_id)

# 訓練模型頁面
@app.route("/ml/train")
def ml_train_page():
    return render_template("train.html")

# 預測頁面
@app.route("/ml/predict")
def ml_predict_page():
    return render_template("predict.html")

# 資料分析視覺化頁面
@app.route("/analysis")
def analysis_page():
    return render_template("analysis.html")


# ============================================================
# 4. API：取得全部乘客資料，包含簡單分頁
# GET /api/passengers?page=1&per_page=20
# ============================================================

@app.route("/api/passengers", methods=["GET"])
def get_passengers():
    # 讀取 query string 的 page 和 per_page 參數，並設定預設值
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)

    # 搜尋姓名
    search = request.args.get("search", "")

    # 計算 SQL 查詢的 offset，用於分頁
    offset = (page - 1) * per_page

    # 根據是否有搜尋關鍵字，執行不同的 SQL 查詢
    if search != "":
        # 有輸入搜尋關鍵字：只查詢姓名符合的資料
        total_row = db.execute(
            """
            SELECT COUNT(*) AS total
            FROM titanic
            WHERE Name LIKE ?
            """,
            (f"%{search}%",)
        ).fetchone()

        rows = db.execute(
            """
            SELECT *
            FROM titanic
            WHERE Name LIKE ?
            ORDER BY PassengerId
            LIMIT ?
            OFFSET ?
            """,
            (f"%{search}%", per_page, offset)
        ).fetchall()

    else:
        # 沒有輸入搜尋關鍵字：查詢全部資料
        total_row = db.execute(
            """
            SELECT COUNT(*) AS total
            FROM titanic
            """
        ).fetchone()

        # 根據 page 和 per_page 的值，從資料庫查詢對應的資料列，
        # 並按照 PassengerId 排序。
        rows = db.execute(
            """
            SELECT *
            FROM titanic
            ORDER BY PassengerId
            LIMIT ?
            OFFSET ?
            """,
            (per_page, offset)
        ).fetchall()

    # 總共有多少筆資料
    total = total_row["total"]

    # 最後回傳 JSON 格式的資料，包含 items（資料列表）、page、per_page 和 total。
    return jsonify({
        "message": "ok",
        "items": [row_to_dict(row) for row in rows],
        "page": page,
        "per_page": per_page,
        "total": total
    }), 200


# ============================================================
# 5. API：取得單一乘客
# GET /api/passengers/1
# ============================================================

@app.route("/api/passengers/<int:passenger_id>", methods=["GET"])
def get_passenger(passenger_id):
    # 根據 passenger_id 查詢資料庫，看看有沒有這個乘客的資料。
    row = db.execute(
        "SELECT * FROM titanic WHERE PassengerId = ?",
        (passenger_id,)
    ).fetchone()

    # 如果 row 是 None，代表資料庫裡沒有這個 passenger_id 的資料，我們就回傳 404 Not Found 的錯誤訊息。
    if row is None:
        return jsonify({"error": "找不到資料"}), 404

    # 如果有找到資料，我們就把這筆資料轉成 dict，然後回傳 JSON 格式的資料。
    return jsonify({
        "message": "ok", 
        "item": row_to_dict(row)}
    ), 200


# ============================================================
# 6. API：新增乘客
# POST /api/passengers
# ============================================================

@app.route("/api/passengers", methods=["POST"])
def create_passenger():
    # 從 request 的 JSON body 讀取資料
    data = request.get_json()

    # 執行 SQL INSERT 語句，把新的乘客資料新增到 titanic 資料表中。
    cursor = db.execute(
        """
        INSERT INTO titanic (
            Survived, Pclass, Name, Sex, Age,
            SibSp, Parch, Ticket, Fare, Cabin,
            Embarked
        )
        VALUES (
            ?, ?, ?, ?, ?, 
            ?, ?, ?, ?, ?, 
            ?
        )
        """,
        (
            data["Survived"],
            data["Pclass"],
            data["Name"],
            data["Sex"],
            data["Age"],
            data["SibSp"],
            data["Parch"],
            data["Ticket"],
            data["Fare"],
            data["Cabin"],
            data["Embarked"]
        )
    )

    # 執行 commit()，把剛剛的 INSERT 操作真正寫入資料庫。
    db.commit()

    # cursor.lastrowid 會回傳剛剛 INSERT 的那筆資料的自動增加的 ID，
    # 也就是 PassengerId。
    new_id = cursor.lastrowid

    # 根據 new_id 查詢剛剛新增的那筆資料，這樣我們就可以把完整的資料回傳給前端了。
    row = db.execute(
        "SELECT * FROM titanic WHERE PassengerId = ?",
        (new_id,)
    ).fetchone()

    # 最後回傳 JSON 格式的資料，包含 message 和 item（剛剛新增的那筆資料）。
    return jsonify({
        "message": "created",
        "item": row_to_dict(row)
    }), 201


# ============================================================
# 7. API：修改乘客
# PUT /api/passengers/1
# ============================================================

@app.route("/api/passengers/<int:passenger_id>", methods=["PUT"])
def update_passenger(passenger_id):
    # 從 request 的 JSON body 讀取資料
    data = request.get_json()

    # 執行 SQL UPDATE 語句，根據 passenger_id 把對應的資料更新成新的值。
    cursor = db.execute(
        """
        UPDATE titanic
        SET
            Survived = ?,
            Pclass = ?,
            Name = ?,
            Sex = ?,
            Age = ?,
            SibSp = ?,
            Parch = ?,
            Ticket = ?,
            Fare = ?,
            Cabin = ?,
            Embarked = ?
        WHERE PassengerId = ?
        """,
        (
            data["Survived"],
            data["Pclass"],
            data["Name"],
            data["Sex"],
            data["Age"],
            data["SibSp"],
            data["Parch"],
            data["Ticket"],
            data["Fare"],
            data["Cabin"],
            data["Embarked"],
            passenger_id
        )
    )

    # 執行 commit()，把剛剛的 UPDATE 操作真正寫入資料庫。
    db.commit()

    # 如果沒有更新任何資料，則回傳 404 Not Found 的錯誤訊息。
    if cursor.rowcount == 0:
        return jsonify({"error": "找不到資料"}), 404

    # 根據 passenger_id 查詢剛剛更新的那筆資料，這樣我們就可以把完整的資料回傳給前端了。
    row = db.execute(
        "SELECT * FROM titanic WHERE PassengerId = ?",
        (passenger_id,)
    ).fetchone()

    # 如果 row 是 None，代表資料庫裡沒有這個 passenger_id 的資料，我們就回傳 404 Not Found 的錯誤訊息。
    if row is None:
        return jsonify({"error": "找不到資料"}), 404

    # 最後回傳 JSON 格式的資料，包含 message 和 item（剛剛更新的那筆資料）。
    return jsonify({
        "message": "updated",
        "item": row_to_dict(row)
    }), 200


# ============================================================
# 8. API：刪除乘客
# DELETE /api/passengers/1
# ============================================================

@app.route("/api/passengers/<int:passenger_id>", methods=["DELETE"])
def delete_passenger(passenger_id):
    # 執行 SQL DELETE 語句，根據 passenger_id 把對應的資料從 titanic 資料表中刪除。
    cursor = db.execute(
        "DELETE FROM titanic WHERE PassengerId = ?",
        (passenger_id,)
    )

    # 執行 commit()，把剛剛的 DELETE 操作真正寫入資料庫。
    db.commit()

    # 如果沒有刪除任何資料，則回傳 404 Not Found 的錯誤訊息。
    if cursor.rowcount == 0:
        return jsonify({"error": "找不到資料"}), 404

    # 最後回傳 JSON 格式的資料，包含 message，告訴前端這筆資料已經被刪除了。
    return jsonify({
        "message": "deleted"
    }), 200 # 你也可以設定 204，但不會有 response body，前端無法判斷成功還是失敗


# ============================================================
# 9. API：機器學習 - 一鍵訓練模型
# POST /api/ml/train
# ============================================================

@app.route("/api/ml/train", methods=["POST"])
def start_training():
    with training_lock:
        # 如果已經在訓練中，不要再開一個新的訓練，避免搶資源、狀態互相覆蓋
        if training_state["status"] == "running":
            return jsonify({"error": "模型正在訓練中，請稍候"}), 409

        training_state["status"] = "running"
        training_state["message"] = "已加入訓練佇列..."
        training_state["error"] = None

    # 用背景執行緒跑訓練，這樣這個 API 可以馬上回應「已開始訓練」，
    # 不用讓使用者的瀏覽器整個卡住等訓練跑完（Titanic 資料集雖然訓練不算慢，
    # 但這個寫法在資料變大時依然適用，也讓「訓練中」狀態有機會被前端看到）。
    thread = threading.Thread(target=_run_training_in_background, daemon=True)
    thread.start()

    return jsonify({"message": "已開始訓練"}), 202


# ============================================================
# 10. API：機器學習 - 查詢訓練狀態
# GET /api/ml/status
# ============================================================

@app.route("/api/ml/status", methods=["GET"])
def get_training_status():
    with training_lock:
        # 回傳一份複本，避免外部拿到參照後意外修改到全域狀態
        state_copy = dict(training_state)
    return jsonify(state_copy), 200


# ============================================================
# 11. API：機器學習 - 預測
# POST /api/ml/predict
#   - 單筆預測：Content-Type application/json
#   - 批次預測：Content-Type multipart/form-data，帶一個叫 "file" 的 CSV 檔案
# ============================================================

@app.route("/api/ml/predict", methods=["POST"])
def predict():
    # 先確認模型有沒有訓練/存檔過
    pipeline = ml_model.load_model()
    if pipeline is None:
        return jsonify({"error": "尚未訓練模型，請先到「訓練模型」頁面點擊訓練"}), 400

    # ---- 批次預測：有上傳檔案 ----
    if "file" in request.files:
        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "沒有選擇檔案"}), 400

        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({"error": f"CSV 檔案讀取失敗: {e}"}), 400

        missing_cols = [col for col in ml_model.FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            return jsonify({"error": f"CSV 缺少必要欄位: {missing_cols}"}), 400

        result_df = ml_model.predict_dataframe(pipeline, df)
        items = clean_nan_for_json(result_df.to_dict(orient="records"))

        return jsonify({
            "message": "ok",
            "count": len(items),
            "items": items,
        }), 200

    # ---- 單筆預測：JSON body ----
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "請提供 JSON 格式的乘客資料，或上傳 CSV 檔案"}), 400

    missing_cols = [col for col in ml_model.FEATURE_COLUMNS if col not in data]
    if missing_cols:
        return jsonify({"error": f"缺少必要欄位: {missing_cols}"}), 400

    single_df = pd.DataFrame([data])
    result_df = ml_model.predict_dataframe(pipeline, single_df)
    item = clean_nan_for_json(result_df.to_dict(orient="records"))[0]

    return jsonify({
        "message": "ok",
        "item": item,
    }), 200


# ============================================================
# 12. API：資料分析 - 一次取得所有 EDA 統計結果
# GET /api/analysis/summary
# ============================================================

@app.route("/api/analysis/summary", methods=["GET"])
def get_analysis_summary():
    df = pd.read_sql_query("SELECT * FROM titanic", db)

    if len(df) == 0:
        return jsonify({"error": "資料庫目前沒有任何乘客資料，無法進行分析"}), 400

    # 如果已經訓練過模型，順便把「特徵重要性」也算進去，讓分析頁可以呼應模型頁的結果
    pipeline = ml_model.load_model()

    result = analysis.compute_full_analysis(df, pipeline=pipeline)
    return jsonify(result), 200


# ============================================================
# 13. 啟動 Flask
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )