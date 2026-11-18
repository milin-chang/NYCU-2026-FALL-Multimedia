# FFHQ 範例影像品質指標

本專案的 Python 虛擬環境在 `metrics/.venv`。目前電腦已建立好環境及套件；在專案根目錄開啟 **PowerShell**，依序執行：

```powershell
cd C:\uwuowoawa\2_colledge\115-1\multimedia_final_project
Set-ExecutionPolicy -Scope Process Bypass -Force
. .\metrics\.venv\Scripts\Activate.ps1
python --version
python .\metrics\score_example_photos.py
deactivate
```

`Set-ExecutionPolicy -Scope Process Bypass` 只對目前這個 PowerShell 視窗生效，用來允許執行虛擬環境的啟用腳本。啟用後，`python` 會指向 `metrics/.venv/Scripts/python.exe`。每次開新視窗要重新啟用；做完可輸入 `deactivate`。

如果不想啟用，也可以直接執行：

```powershell
& .\metrics\.venv\Scripts\python.exe .\metrics\score_example_photos.py
```

程式會處理 `metrics/example_photo` 中的 6 張 PNG。每張影像的分數與生成圖在 `metrics/outputs/example_photo/<檔名>_bicubic_x4/`，合併表為 `metrics/outputs/example_photo/scores_all.csv`。計算方法與結果解讀見 [metrics/README.md](metrics/README.md)。

## 補裝套件

目前的 `.venv` 是用 uv 建立的，沒有內建 `pip`。如果要依 `requirements.txt` 補裝套件，請在專案根目錄執行：

```powershell
$env:UV_CACHE_DIR = (Resolve-Path .\metrics\.tools\cache).Path
& .\metrics\.tools\uv\uv.exe pip install --python .\metrics\.venv\Scripts\python.exe -r .\metrics\requirements.txt
```

`.venv` 與 `.tools` 都不會提交到 Git。換一台有 Python 3.11 的電腦時，可用 `py -3.11 -m venv .\metrics\.venv` 重建虛擬環境，再啟用並執行 `python -m pip install -r .\metrics\requirements.txt`。
