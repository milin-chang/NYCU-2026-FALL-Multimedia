# smoke_swinir_s_ffhq500_highfreq_s8_2step

## 實驗規格

- 資料：FFHQ 500 張；訓練／驗證／測試 = 400／50／50
- 模型：Lightweight SwinIR-S ×4，隨機初始化，訓練 2 步
- 訓練損失：L1 pixel only；八個品質指標僅用於評分
- 驗證：1 val images at each val interval；測試：1 test images once, using best validation checkpoint
- 最佳模型：第 2 步，按驗證 PSNR 選出

## 測試集平均分數

| 指標 | 分數 | 方向 |
| --- | ---: | --- |
| PSNR | 8.112578 | higher |
| SSIM | 0.354545 | higher |
| LPIPS | 0.956805 | lower |
| DISTS | 0.720569 | lower |
| MS_SSIM | 0.099340 | higher |
| FSIM | 0.662755 | higher |
| VIFP | 0.010856 | higher |
| UQI | 0.516974 | higher |

## 驗證過程

| 步數 | PSNR | SSIM | LPIPS | DISTS | MS-SSIM | FSIM | VIFp | UQI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 8.1276 | 0.4054 | 0.9959 | 0.7203 | 0.1778 | 0.6854 | 0.0080 | 0.5291 |

逐張分數與 SR 圖：`runs/smoke_swinir_s_ffhq500_highfreq_s8_2step/test/`
