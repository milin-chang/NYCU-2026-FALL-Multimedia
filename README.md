# FFHQ 範例影像的純數學品質指標

輸入為 `example_photo/` 內的 6 張 1024×1024 PNG。程式逐張以 Pillow BICUBIC 縮至 256×256，再放大回 1024×1024。這是 **×4 雙三次插值基準**，不是 SwinIR 輸出，也不是整個測試集的模型成績。

有參考指標以放大圖為輸入、原圖為 HR 參考；無參考指標 PIQE 則分別計算原圖與放大圖。影像以 RGB 載入，PIQ 指標使用 `[0, 1]` 範圍及套件預設參數。VIF 是 `pyiqa` 的小波版本，VIFp 是 `piq` 的像素域版本，兩者不可視為同一指標。各指標的高低方向記在 CSV 的 `direction` 欄。

在有 Python 3.11 的 Windows PowerShell 中，從 repo 根目錄建立獨立虛擬環境並執行：

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass -Force
. .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python score_example_photos.py
deactivate
```

Linux/macOS 可用 `python3.11 -m venv .venv`、`source .venv/bin/activate`，再執行同一個 Python 程式。若本機已有 `.venv`，只需啟用並執行程式。目前工作機的環境由 uv 建立，沒有內建 `pip`，但已安裝所需套件。

結果位於 `outputs/example_photo/`：每張圖片各有 LR 圖、放大圖、`scores.csv` 與含套件版本及 SHA-256 的 `run.json`；`scores_all.csv` 合併所有圖片的分數。`not_computed` 代表尚未採用經確認的實作，不是 0 分。

若要評估 SwinIR 或改良模型，應以相同的 HR 與降採樣方式建立 LR，改用模型輸出代替 BICUBIC 放大圖，再於固定測試集逐張計算並彙總。發表比較時，需固定各指標的實作版本與參數。
