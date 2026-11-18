# smoke_swinir_s_ffhq500_x4_2step

## 實驗規格

- 資料：FFHQ 500 張；訓練／驗證／測試 = 400／50／50
- 模型：Lightweight SwinIR-S ×4，隨機初始化，訓練 2 步
- 訓練損失：L1 pixel only；八個品質指標僅用於評分
- 最佳模型：第 2 步，按驗證 PSNR 選出

## 測試集平均分數

| 指標 | 分數 | 方向 |
| --- | ---: | --- |
| PSNR | 8.117893 | higher |
| SSIM | 0.370931 | higher |
| LPIPS | 0.944736 | lower |
| DISTS | 0.721326 | lower |
| MS_SSIM | 0.100921 | higher |
| FSIM | 0.655762 | higher |
| VIFP | 0.011423 | higher |
| UQI | 0.517241 | higher |

## 驗證過程

| 步數 | PSNR | SSIM | LPIPS | DISTS | MS-SSIM | FSIM | VIFp | UQI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 8.1312 | 0.4305 | 1.0176 | 0.7224 | 0.1816 | 0.6826 | 0.0082 | 0.5293 |

逐張分數與 SR 圖：`runs/smoke_swinir_s_ffhq500_x4_2step/test/`
