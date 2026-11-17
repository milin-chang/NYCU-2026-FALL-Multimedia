# FFHQ 人臉影像的數學式品質指標示範

本專案示範如何計算人臉超解析影像的品質指標。程式以六張 FFHQ 圖片製作「縮小後再放大」的比較組，保存每張圖的分數與處理後影像。這是指標計算範例，**沒有訓練或執行 SwinIR**。

## 實驗目的

超解析（Super Resolution, SR）要從低解析度影像產生較高解析度影像。評估時，有些指標必須將輸出圖與高解析度原圖比較；另一些指標只看輸出圖。本範例固定影像處理方式，示範兩類指標的輸入、輸出與分數讀法，供後續評估模型時沿用。

## 操作教學
Watch how_to_use.md Thanks!


## 現在推薦使用指標

```python
## 通用傳統指標
score("PSNR", "higher", lambda: piq.psnr(restored_tensor, reference_tensor), scores)
score("SSIM", "higher", lambda: piq.ssim(restored_tensor, reference_tensor), scores)

## 人臉感知指標
score("MS-SSIM", "higher", lambda: piq.multi_scale_ssim(restored_tensor, reference_tensor), scores)
score("FSIM", "higher", lambda: piq.fsim(restored_tensor, reference_tensor, chromatic=False), scores)
score("VIFp", "higher", lambda: piq.vif_p(restored_tensor, reference_tensor), scores)
score("UQI", "higher", lambda: uqi(original, restored), scores)

```

## 使用這些指標的原因

https://github.com/milin-chang/NYCU-2026-FALL-Multimedia/discussions/1#discussioncomment-19684228




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
