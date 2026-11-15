# SwinIR-M ×4 FFHQ Baseline

從官方 `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth` 的 `params_ema` 初始化 SwinIR-M，在 FFHQ 上做 ×4 real-world SR 微調。實驗設定為 **1,000 張訓練、125 張驗證、125 張測試、seed 42、100,000 optimizer steps**。生成器維持官方架構與 KAIR real-SR GAN 的損失組合。

課程 proposal 原先把 FFHQ 列為額外測試集。本分支依確認後的任務設定進行 FFHQ 微調，並保留獨立的 FFHQ test split。要比較後續改良分支，請共用 `manifest.json`、初始權重、輸入退化、batch size、訓練步數與評估設定。[目前的 Spec](https://github.com/milin-chang/NYCU-2026-FALL-Multimedia/discussions/1) · [開會紀錄](https://github.com/milin-chang/NYCU-2026-FALL-Multimedia/discussions/2) · [期末專題提案](https://github.com/user-attachments/files/34649674/Group_06_Final_Project_Proposal.pdf)

以下指令可在 Linux Bash 或 Windows Anaconda Prompt 執行。從 repo 根目錄執行。訓練設定使用一張支援 CUDA 12.1 的 NVIDIA GPU，CPU 可用於推論與評估。

## 1. 建置 Conda 環境

```sh
git clone --branch 314554035_Baseline --single-branch https://github.com/milin-chang/NYCU-2026-FALL-Multimedia.git
cd NYCU-2026-FALL-Multimedia
conda create --name swinir-baseline --override-channels -c conda-forge python=3.11.11 pip=25.0.1 -y
conda activate swinir-baseline
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

CPU 環境可將 PyTorch 指令的索引改成 `https://download.pytorch.org/whl/cpu`，並在推論及評估指令加 `--device cpu`。[PyTorch 安裝指令](https://pytorch.org/get-started/previous-versions/#v251)

## 2. 下載官方預訓練權重

```sh
python prepare.py assets
```

權重存到 `assets/weights/`，腳本核對 SHA-256。這一步也會下載固定版本的 SwinIR 與 KAIR 檔案至 `assets/upstream/`，並將 KAIR 的舊版 `scipy.finfo` 別名換成原本的 `numpy.finfo`。訓練首次執行時，TorchVision 會快取 VGG19 與 AlexNet 權重，第一次計算 DISTS 時會另外快取 VGG16。[SwinIR v0.0 權重](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0) · [KAIR SwinIR real-world GAN 設定](https://github.com/cszn/KAIR/blob/fc1732f4a4514e42ce15e5b3a1e18c828af47a1e/options/swinir/train_swinir_sr_realworld_x4_gan.json)

## 3. 下載 FFHQ 並建立固定切分

```sh
python prepare.py dataset --counts 1000 125 125
```

這會透過 KaggleHub 下載 Spec 指定的 `arnaud58/flickrfaceshq-dataset-ffhq/versions/1` 至 `assets/kaggle/`，再抽取 1,250 張。首次下載會拉取整份 Kaggle archive，需要預留 archive、解壓目錄與處理後影像的空間。Kaggle 若要求登入，可依 KaggleHub 文件設定憑證後重跑。[FFHQ Kaggle 來源](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq/versions/1) · [KaggleHub 驗證方式](https://github.com/Kaggle/kagglehub#authenticate)

有同版本且已解壓的 Kaggle 資料時，可直接指定含完整 PNG 樹的資料夾。從相同的解壓根目錄執行，以保留一致的相對路徑。

```sh
python prepare.py dataset --source /path/to/extracted-ffhq --source-id arnaud58/flickrfaceshq-dataset-ffhq/versions/1 --counts 1000 125 125
```

腳本依排序後的 PNG 相對路徑用 seed 42 洗牌，依序取 train、val、test，輸出到 `data/ffhq/`。HR 統一維持 512×512 RGB。訓練時抽取 256×256 HR patch、隨機水平翻轉，再用 KAIR 的 BSRGAN 生成 64×64 LR patch。驗證／測試以整張 512×512 HR 生成固定 128×128 LR PNG。`manifest.json` 記錄每張原始路徑、輸出檔 SHA-256 和驗證／測試退化種子。跨機器共用 manifest 和 PNG，或重建後核對 manifest 的 SHA-256。[NVIDIA FFHQ 來源與授權](https://github.com/NVlabs/ffhq-dataset#license)

## 4. 訓練與續訓

```sh
python baseline.py train --data data/ffhq --run runs/baseline --steps 100000 --batch 1 --workers 0 --g-lr 1e-5 --d-lr 1e-4 --val-every 5000
```

生成器全網更新。損失為 `L1 pixel + VGG19 feature L1 + 0.1 × GAN`，光譜正規化 U-Net 判別器以官方 KAIR 設定初始化，G／D 使用 Adam 和固定學習率。驗證使用 decay 0.999 的生成器 EMA。支援 BF16 的 CUDA GPU 預設啟用 BF16 AMP，其餘使用 FP32。可用 `--fp32` 固定精度，顯存不足可加 `--activation-checkpoint`，CPU 備料跟不上可提高 `--workers`。更改 batch size 會改變每步樣本數與資料順序，正式比較時請固定。

中斷後使用同樣的設定加 `--resume`，從 `last.pth` 的步數繼續到 `--steps`。同一目錄的 `events.jsonl` 會追加續訓事件。

```sh
python baseline.py train --data data/ffhq --run runs/baseline --steps 100000 --batch 1 --workers 0 --g-lr 1e-5 --d-lr 1e-4 --val-every 5000 --resume
```

| 檔案 | 內容 |
| --- | --- |
| `events.jsonl` | 權重與資料 manifest SHA-256、seed、版本、GPU、設定、G／D 各項 loss、耗時、峰值訓練顯存、step 0 和後續驗證指標 |
| `last.pth` | 最近一次驗證的 G、EMA、D、optimizer、精度設定和 RNG 狀態 |
| `best.pth` | 微調後 validation LPIPS 最低的 EMA 權重與對應步數 |

## 5. 推論與測試

對單張 LR 檔案或一個 LR 資料夾推論。`--tile` 以 LR 像素為單位，須為 8 的倍數，使用 `--tile 0` 時會整張推論。

```sh
python baseline.py infer --weights runs/baseline/best.pth --input data/ffhq/test/lr --out outputs/inference --tile 128
```

用同一批 test LR 比較原始 pretrained 與微調後的模型。輸出 SR PNG、逐張 CSV、平均分數及 checkpoint／manifest SHA-256。四項指標都使用完整的 8-bit RGB SR 和 HR，PSNR、SSIM 不切邊，LPIPS 使用 AlexNet，DISTS 使用官方 PyTorch 實作。

```sh
python baseline.py evaluate --data data/ffhq --split test --out outputs/pretrained --tile 128
python baseline.py evaluate --data data/ffhq --split test --weights runs/baseline/best.pth --out outputs/finetuned --tile 128
```

目前訓練期間記錄 PSNR、SSIM、LPIPS，測試時另外計算 DISTS。Spec 的 IFQA、DSL-FIQA 可對上述相同的 `sr/` 影像另外執行。團隊比較前應固定這兩項的 checkpoint、臉部前處理與逐張分數彙總方式。[IFQA 官方實作](https://github.com/AIM-SKKU/IFQA) · [DSL-FIQA 官方實作](https://github.com/DSL-FIQA/DSL-FIQA)
