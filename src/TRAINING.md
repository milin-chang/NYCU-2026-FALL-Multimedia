# FFHQ 500 張：Lightweight SwinIR-S ×4 訓練

從專案根目錄執行。先用 `src/setup.ps1` 建立 CUDA 環境；既有的 500 張分割清單與 ×4 LR 影像分別在 `dataset/ffhq_split_seed42_500.json`、`dataset/ffhq_lightweight_x4/lr/`。

```powershell
& .\src\.venv\Scripts\python.exe .\src\train_lightweight.py --name swinir_s_ffhq500_x4_seed42_150k_20261118 --steps 150000
```

中斷後以**相同參數**續訓，在命令最後加 `--resume`。每 5,000 步覆寫 `last.pth`，所以非正常中斷最多需重做約 5,000 步。`Ctrl+C` 也依同一方式續訓。

## 資料與訓練

- 固定 seed 42，從 4,000 張 FFHQ 中挑 500 張：訓練 400、驗證 50、測試 50；三組互不重複。
- HR 為 1024×1024 RGB，LR 為 KAIR bicubic 降採樣 ×4 的 256×256 RGB。訓練時從同一對 LR/HR 擷取 64×64／256×256 對齊 patch，做相同的翻轉與旋轉。
- 使用官方 KAIR 來源的 Lightweight SwinIR-S ×4 結構，隨機初始化；batch 1、Adam、初始學習率 2×10⁻⁴、BF16、L1 像素損失、EMA 0.999。學習率在 75,000、120,000、135,000、142,500 步各乘 0.5。
- 每 15,000 步用 EMA 模型評估全部 50 張驗證圖；依驗證 PSNR 選最佳 checkpoint。訓練完成後用這個 checkpoint 在全部 50 張測試圖評分一次。完整規格以 `spec/<名稱>.json` 為準。

## 八個評分指標

評分使用重建後的 8 位元 RGB SR 圖與原始 HR 圖，**只在驗證與測試時使用，不參與 loss 或梯度更新**。

| 組別 | 指標 | 方向 |
| --- | --- | --- |
| 一般影像品質 | PSNR、SSIM | 越高越好 |
| 一般感知品質 | LPIPS、DISTS | 越低越好 |
| 應用於人臉影像的傳統指標 | MS-SSIM、FSIM、VIFp、UQI | 越高越好 |

後四項是在 FFHQ 人臉上計算的通用影像品質指標，並非人臉辨識或專屬人臉模型。LPIPS 與 DISTS 需要預訓練的特徵網路，但它們不會更新，也不參與 SR 模型訓練。

## 檔案位置

同一名稱串起三個資料夾：

- `spec/<名稱>.json`：固定的資料、模型、超參數與評分規格。
- `runs/<名稱>/`：`events.jsonl`、`status.json`、`checkpoints/last.pth`、`checkpoints/best.pth`、每次驗證的 `per_image.csv` 與 `summary.json`，以及最終測試的 CSV、摘要和 50 張 SR PNG。此分支包含這些檔案，clone 時需預留空間。
- `reports/<名稱>/`：完成後的 `README.md` 與 `results.json`，可提交到 Git。

查看執行進度：

```powershell
Get-Content .\runs\swinir_s_ffhq500_x4_seed42_150k_20261118\status.json
Get-Content .\runs\swinir_s_ffhq500_x4_seed42_150k_20261118\events.jsonl -Tail 5
```

測試流程可用 `--steps 2 --val-every 2 --save-every 1 --val-limit 1 --test-limit 1`，但請另取測試名稱；正式訓練不設 `--val-limit` 或 `--test-limit`。

訓練期間重新產生收斂圖：

```powershell
& .\src\.venv\Scripts\python.exe .\src\plot_convergence.py --name swinir_s_ffhq500_x4_seed42_150k_20261118
```

執行一次會更新 `reports/<名稱>/` 中的三張快照：`convergence.png`（L1 loss 與八個驗證指標平均值）、`learning_rate.png`（實際記錄的學習率）、`per_image_validation.png`（每次驗證的 50 張影像逐點分數與分布）。

## 第二組：bicubic ×4 後加入高頻雜訊

這組沿用同一份 FFHQ 500 張分割，改由 1024×1024 HR 先產生 bicubic 256×256 LR，再在 LR 加入高頻 Gaussian 雜訊（σ=8／255、高通截止頻率 0.08 cycles/pixel）。訓練時每個 patch 重新抽雜訊；驗證與測試使用固定 noisy LR 圖，HR 保持原圖。先產生固定 LR：

```powershell
& .\src\.venv\Scripts\python.exe .\src\prepare_highfreq_lr.py
```

從專案根目錄執行 100,000 步：

```powershell
& .\src\.venv\Scripts\python.exe .\src\train_lightweight.py --name swinir_s_ffhq500_x4_highfreq_s8_seed42_100k_20261118 --lr-images dataset/ffhq_lightweight_x4_highfreq_s8/lr --noise-sigma 8 --steps 100000 --val-every 10000 --resume
```

正式 run 已由 `--prepare-only` 建立規格，因此首次 GPU 執行也帶 `--resume`。此後中斷續訓使用同一命令。每 5,000 步保存 checkpoint；完整設定在 `spec/<名稱>.json`，訓練紀錄在 `runs/<名稱>/`，完成後的評分報告在 `reports/<名稱>/`。
