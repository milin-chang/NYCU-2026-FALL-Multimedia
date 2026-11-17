# FFHQ 人臉影像的數學式品質指標示範

本專案示範如何計算人臉超解析影像的品質指標。程式以六張 FFHQ 圖片製作「縮小後再放大」的比較組，保存每張圖的分數與處理後影像。這是指標計算範例，**沒有訓練或執行 SwinIR**。

## 實驗目的

超解析（Super Resolution, SR）要從低解析度影像產生較高解析度影像。評估時，有些指標必須將輸出圖與高解析度原圖比較；另一些指標只看輸出圖。本範例固定影像處理方式，示範兩類指標的輸入、輸出與分數讀法，供後續評估模型時沿用。

## 支援指標
```python
score("PSNR", "higher", lambda: piq.psnr(restored_tensor, reference_tensor), scores)
score("SSIM", "higher", lambda: piq.ssim(restored_tensor, reference_tensor), scores)
score("MS-SSIM", "higher", lambda: piq.multi_scale_ssim(restored_tensor, reference_tensor), scores)
score("FSIM", "higher", lambda: piq.fsim(restored_tensor, reference_tensor, chromatic=False), scores)
score("FSIMc", "higher", lambda: piq.fsim(restored_tensor, reference_tensor, chromatic=True), scores)
score("GMSD", "lower", lambda: piq.gmsd(restored_tensor, reference_tensor), scores)
score("MDSI", "lower", lambda: piq.mdsi(restored_tensor, reference_tensor), scores)
score("VIFp", "higher", lambda: piq.vif_p(restored_tensor, reference_tensor), scores)
score("SR-SIM", "higher", lambda: piq.srsim(restored_tensor, reference_tensor), scores)
score("VSI", "higher", lambda: piq.vsi(restored_tensor, reference_tensor), scores)
score("UQI", "higher", lambda: uqi(original, restored), scores)
score("PIQE (original)", "lower", lambda: piqe(original)[0], scores, "HR only")
score("PIQE (bicubic x4)", "lower", lambda: piqe(restored)[0], scores, "bicubic x4 only")
score("VIF", "higher", lambda: pyiqa.create_metric("vif", device="cpu")(restored_tensor, reference_tensor), scores)
```

## 實驗材料

`example_photo/` 包含六張 [Flickr-Faces-HQ（FFHQ）資料集](https://github.com/NVlabs/ffhq-dataset)的 1024 × 1024 PNG：`00034.png`、`00036.png`、`00048.png`、`00051.png`、`00061.png`、`00077.png`。每張圖都作為該次比較的高解析度原圖（HR）。六張圖只用於展示計算流程，不代表 FFHQ 測試集的平均表現。

## 實驗方法

程式對每張圖片依序執行：

1. 讀取 1024 × 1024 的 RGB 原圖，作為 HR 參考。
2. 使用 Pillow 的 `Image.Resampling.BICUBIC` 縮小至 256 × 256，得到低解析度圖（LR）。
3. 使用相同的雙三次插值方法放大回 1024 × 1024，得到 ×4 插值放大圖。
4. 將插值放大圖與 HR 原圖輸入有參考指標；PIQE 則分別評估 HR 原圖與插值放大圖。

這裡的「×4」指寬與高各放大四倍。雙三次插值放大圖是固定的比較基準，**不是模型生成的 SR 影像**。指標以 RGB 圖片計算；PIQ 套件接收範圍為 `[0, 1]` 的影像張量，其餘設定使用各套件的預設值。

| 類型 | 指標 | 分數方向 |
| --- | --- | --- |
| 有參考：比較插值放大圖與 HR 原圖 | PSNR、SSIM、MS-SSIM、FSIM、FSIMc、VIF、VIFp、CW-SSIM、SR-SIM、VSI、UQI | 越高越好 |
| 有參考：衡量兩圖差異 | GMSD、MDSI | 越低越好 |
| 無參考：單張影像品質 | PIQE（HR 原圖、插值放大圖各算一次） | 越低越好 |
| 尚未採用經確認的實作 | IFC、SPSIM、RFSIM、VSNR-FR、GSM | 未計算 |

VIF 與 VIFp 是不同版本：本範例分別使用 `pyiqa` 的小波域 VIF 與 `piq` 的像素域 VIFp，不能把兩者分數當作同一指標直接比較。`not_computed` 表示沒有計算結果，不是 0 分。這些指標是一般影像品質指標，並非專為人臉設計的辨識或品質模型。

## 執行方式

在分支根目錄、已安裝 Python 3.11 的 Windows PowerShell 執行：

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass -Force
. .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python score_example_photos.py
deactivate
```

第一行只需在建立虛擬環境時執行一次。`Set-ExecutionPolicy -Scope Process Bypass` 只影響目前的 PowerShell 視窗。再次執行時，啟用 `.venv` 後直接跑 `python score_example_photos.py` 即可。

Linux/macOS 可從分支根目錄執行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python score_example_photos.py
deactivate
```

## 結果檔案與讀法

| 路徑 | 內容 |
| --- | --- |
| [`outputs/example_photo/scores_all.csv`](outputs/example_photo/scores_all.csv) | 六張圖的合併表；每張圖有 15 筆實際分數及 5 筆未計算標記，共 120 列資料。 |
| `outputs/example_photo/<圖片編號>_bicubic_x4/scores.csv` | 該張圖片的各項指標、分數方向、比較對象及計算狀態。 |
| `outputs/example_photo/<圖片編號>_bicubic_x4/lr_256.png` | 256 × 256 的 LR 圖。 |
| `outputs/example_photo/<圖片編號>_bicubic_x4/bicubic_1024.png` | 1024 × 1024 的雙三次插值放大圖。 |
| `outputs/example_photo/<圖片編號>_bicubic_x4/run.json` | 原圖與放大圖的 SHA-256、處理設定、主要套件版本及逐項分數。 |

例如，[`00034.png` 的逐項結果](outputs/example_photo/00034_bicubic_x4/scores.csv)中，插值放大圖對 HR 原圖的 PSNR 為 **36.8769 dB**、SSIM 為 **0.9922**、VIF 為 **0.3203**；PIQE 對 HR 原圖單獨計算為 **14.4799**，對插值放大圖單獨計算為 **72.5047**。各指標的數值尺度不同，應比較同一指標在相同資料與設定下的結果。

## 使用限制與後續評估

目前的分數只描述這六張圖片的雙三次插值結果，不能用來宣稱 SwinIR 或改良模型的品質，也沒有測量人的主觀評分。若要比較模型，需以相同的 HR 圖與降採樣流程建立 LR，將程式中的雙三次放大步驟換成模型輸出，再在固定的測試集上逐張計算並彙總。同時保留相同的指標實作、版本與參數，才能公平比較。
