# Workflow: Luxical-TW Development

## 1. 資料處理流程
1. 準備原始 NDJSON 資料。
2. 使用 `extract_ndjson_fields.py` 進行擷取與分行（支援滑動視窗）。
3. 產出 `corpus.parquet`（必須包含 `text` 欄位）。

## 2. 訓練流程
1. 配置 `data/examples/user_dict.txt`。
2. 執行 `train_really.py` 進行小規模測試（`--max_samples 5000`）。
3. 確認驗證指標後，執行全量訓練。

## 3. 驗證流程
1. 執行 `verify_model.py`。
2. 檢查 Top-5 檢索結果的語義關聯性。
3. 監控每秒推理量 (TPS)。
