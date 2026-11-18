# smoke_swinir_s_ffhq500_masked_s16_2step

## 實驗規格

- 資料：FFHQ 500 張；訓練／驗證／測試 = 400／50／50
- 模型：Lightweight SwinIR-S ×4，隨機初始化，訓練 2 步
- 訓練損失：L1 pixel only；八個品質指標僅用於評分
- 驗證：1 val images at each val interval；測試：1 test images once, using best validation checkpoint
- 最佳模型：第 2 步，按驗證 PSNR 選出

## 測試集平均分數

| 指標 | 分數 | 方向 |
| --- | ---: | --- |
| PSNR | 8.106805 | higher |
| SSIM | 0.343452 | higher |
| LPIPS | 0.973261 | lower |
| DISTS | 0.717594 | lower |
| MS_SSIM | 0.098157 | higher |
| FSIM | 0.660549 | higher |
| VIFP | 0.010985 | higher |
| UQI | 0.516800 | higher |

## 驗證過程

| 步數 | PSNR | SSIM | LPIPS | DISTS | MS-SSIM | FSIM | VIFp | UQI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 8.1273 | 0.4124 | 1.0250 | 0.7177 | 0.1793 | 0.6885 | 0.0078 | 0.5292 |

逐張分數與 SR 圖：`runs/smoke_swinir_s_ffhq500_masked_s16_2step/test/`
