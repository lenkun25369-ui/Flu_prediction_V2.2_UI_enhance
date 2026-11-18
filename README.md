# Flu Radar：流感預測介面

以 **Streamlit** 建立的流感預測研究原型。使用者可輸入 20 項生命徵象、症狀與病史，由隨附的 **XGBoost** 模型進行單筆推論，並以百分比、色彩及圓形圖呈現結果。介面另包含從 FHIR Observation 預填資料的整合程式。

> 本專案是研究與展示用途的推論介面，不能取代流感檢驗、醫師診斷或治療決策。倉庫未提供訓練資料、訓練程式、外部驗證結果或機率校正報告；不要將畫面上的分級直接視為經臨床驗證的判定標準。

## 功能

- 兩欄英文表單：Vitals & Timing、Symptoms & History。
- 修改輸入後，Streamlit 重新執行並更新預測，沒有獨立的「預測」按鈕。
- 使用 `feature_order.txt` 固定模型輸入順序。
- 顯示預測百分比及 LOW / MODERATE / HIGH RISK 標籤。
- 支援透過啟動網址的 `token`、`obs` 參數讀取 FHIR Observation，並預填表單。
- 目前沒有批次預測、重新訓練、結果下載或 FHIR 寫回功能。

## 專案結構

```text
.
├── app.py              # Streamlit 表單、結果視覺化與 FHIR 讀取
├── predict_core.py     # 載入模型、特徵排列與單筆推論
├── feature_order.txt   # 20 項模型特徵的固定順序
├── xgb_model.json      # 已匯出的 XGBoost 模型
├── requirements.txt    # 固定版本的 Python 套件清單
└── README.md
```

## 安裝與啟動

以下是依目前程式入口整理的啟動流程，**並非已完成跨平台實測的安裝保證**。倉庫未指定 Python 版本，也沒有環境鎖檔或自動化測試；建議先在獨立虛擬環境中驗證。

### 1. 下載專案

```bash
git clone https://github.com/lenkun25369-ui/Flu_prediction_V2.2_UI_enhance.git
cd Flu_prediction_V2.2_UI_enhance
python -m venv .venv
```

啟用虛擬環境：

```bash
# Linux / macOS / WSL
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### 2. 安裝相依套件

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

主要直接依賴為 Streamlit、NumPy、XGBoost 與 Requests。完整清單另包含 `nvidia-nccl-cu12` 等環境相關套件，因此不能假設在 Windows、macOS 或所有 CPU 環境均可直接安裝。程式沒有明確要求 CUDA，也沒有在推論時指定 GPU；遇到平台套件錯誤時，應先核對平台及依賴版本，不必因此直接安裝整套 CUDA。

### 3. 在專案根目錄執行

```bash
python -m streamlit run app.py
```

開啟終端機顯示的 Local URL。**務必從專案根目錄啟動**，因為目前程式使用相對路徑讀取 `xgb_model.json` 與 `feature_order.txt`。

若啟動時發生 `experimental_get_query_params` 相關錯誤，請注意 `app.py` 使用此舊介面，而套件清單固定為 Streamlit 1.52.1；需核對 API 相容性。本次文件整理沒有修改程式或依賴版本。

## 輸入資料

介面有預設值，因此開啟後即可能顯示預測。**這些預設值不代表真實個案資料，使用者應逐項確認。**

### 數值輸入

| 特徵 | 意義 | 介面允許範圍 |
|---|---|---|
| `temp` | 體溫（°C） | 30–42 |
| `height` | 身高（cm） | 1–400 |
| `weight` | 體重（kg） | 1–400 |
| `DOI` | 生病天數 | 1–14 |
| `WOS` | 一年中的週次 | 1–53 |
| `season` | 季節代碼 | 1–4 |
| `rr` | 呼吸頻率 | 10–30 |
| `sbp` | 收縮壓 | 50–250 |
| `o2s` | 血氧飽和度（%） | 1–100 |
| `pulse` | 脈搏 | 50–180 |

上述範圍是 UI 限制，不代表生理正常值或模型已驗證的適用範圍。季節代碼的實際對應及時間定義需與原始訓練資料一致，倉庫未附資料字典。

### 二元輸入

所有選單均以 `No = 0`、`Yes = 1` 輸入模型：

| 特徵 | 介面問題 |
|---|---|
| `fluvaccine` | 本年度是否接種流感疫苗 |
| `cough` | 是否出現新的或加重的咳嗽 |
| `coughsputum` | 是否有痰咳嗽 |
| `sorethroat` | 是否喉嚨痛 |
| `rhinorrhea` | 是否流鼻水或鼻塞 |
| `sinuspain` | 是否鼻竇痛 |
| `exposehuman` | 是否接觸確診流感者 |
| `travel` | 近期是否旅遊 |
| `medhistav` | 過去 30 天是否使用流感抗病毒藥物 |
| `pastmedchronlundis` | 是否有慢性肺部疾病 |

直接呼叫核心函式時，請使用精確的 `"No"` 或 `"Yes"` 字串。目前轉換函式會把所有非 `"No"` 值都轉為 1，未做額外輸入驗證。

## 預測結果如何計算

`predict_core.py` 依 `feature_order.txt` 排列輸入，轉成 `float32` 的單列陣列，再建立 `xgboost.DMatrix`。

目前程式將模型輸出當作「陰性類別機率」，以以下公式顯示陽性百分比：

```text
顯示百分比 = (1 - 模型輸出) × 100
```

這是**現行程式的解讀方式**；倉庫沒有原始標籤編碼及訓練流程，使用或替換模型前仍需確認類別方向。不要因檔名相同，就直接替換成輸出陽性機率的模型。

畫面以四捨五入至小數一位的百分比進行分級：

| 顯示百分比 | 標籤 |
|---|---|
| 小於 20% | LOW RISK |
| 20% 至小於 40% | MODERATE RISK |
| 40% 以上 | HIGH RISK |

這些是介面中的固定門檻，倉庫未提供其驗證依據。

## FHIR Observation 整合

目前讀取流程：

1. 從網址取得 `token` 與 `obs` 參數。
2. 使用 Bearer token 對 Observation URL 發出 GET 請求。
3. 遍歷 Observation 的 `component`，以 `component.code.text` 的**完整文字**對應欄位。
4. 將資料填入尚未設定的 Streamlit session state，供表單使用。

這不是完整的 SMART on FHIR OAuth 登入流程，也不是通用的 LOINC／FHIR 編碼映射。來源 Observation 需符合 `load_patient_data_from_fhir()` 的文字與值型別，例如：

- `Temperature (°C)`、`Height (CM)`、`Weight (KG)` 等使用 `valueQuantity.value`。
- `Season (1–4)`、`Week of Year`、`Days of illness` 使用 `valueInteger`。
- 二元欄位以 `valueInteger == 1` 對應 Yes，其他值會被當作 No。

### 整合限制與安全注意事項

- 目前 HTTP 請求設定 `verify=False`，沒有驗證 TLS 憑證；部署前應修正並配置可信憑證。
- token 放在網址可能進入瀏覽紀錄及伺服器紀錄。請勿把真實 token、帶 token 的網址或可識別病歷放入公開倉庫、截圖或示範文件。
- 底部 Token／Observation URL 文字欄位目前沒有接回上方讀取流程；手動修改這兩欄不等於重新載入資料。
- 網路或 JSON 解析失敗時函式可能回傳 `None`，後續 `.items()` 可能出錯；目前沒有完整的 HTTP 狀態、型別與欄位缺漏處理。
- session state 已有的值不會被新取得的資料覆寫；切換個案後應確認沒有沿用前一個案輸入。
- 缺少的值可能保留表單預設值；應避免把「未提供」誤當成已確認的正常值或 No。
- 頁面 CSS 會隱藏 Streamlit alert，除畫面外也應檢查終端機訊息。

## 目前範圍與後續維護

本倉庫提供的是**已訓練模型的推論展示**，不是可重現訓練的完整研究管線。目前未附：

- 訓練／驗證／測試資料與病人層級切分方式。
- 訓練程式、效能指標、校正結果與適用族群說明。
- 自動化測試、跨平台安裝驗證或部署設定。
- 授權條款檔案；公開可見不代表已授予任意再利用權利。

文件依現有原始碼整理。正式部署前，應先完成依賴相容性、模型載入、人工輸入、FHIR 成功／失敗及切換個案等測試，並完成適當的安全與臨床驗證。

