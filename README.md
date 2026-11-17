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

### 在新機器測試 batch size

先執行 `python prepare.py assets`，再在目標 GPU 空閒時測試。探測會使用合成 LR/HR patch 跑同一個訓練步驟，包括 SwinIR 生成器、U-Net 判別器、VGG19 feature loss、反向傳播、Adam 更新及 EMA。每個候選 batch 也會試跑一張驗證尺寸的影像與 LPIPS，並以新的 Python 程序隔離 CUDA 記憶體。首次執行可能需要下載 VGG19 和 AlexNet 權重。探測過程不寫入訓練 checkpoint。

```sh
python probe_batch.py --device cuda:0 --max-batch 32 --steps 5
python probe_batch.py --device cuda:0 --batch 1 --steps 10
```

第一個指令會加倍測試直到 CUDA OOM 或到達上限，再縮小搜尋範圍。`largest_passing_batch` 是通過的最大測試值。若 `reached_search_limit` 為 `true`，可提高 `--max-batch` 繼續測。第二個指令可在不同機器以相同 batch 比較。輸出包含 `seconds_per_step`、`images_per_sec`、PyTorch 已配置及保留顯存的各自峰值，以及測試開始時 CUDA 回報的可用顯存。若正式訓練使用 `--fp32` 或 `--activation-checkpoint`，探測時也加上相同旗標。

探測時間只包括張量已在 GPU 上的訓練步驟。要估算實際時數，可另用同一份 FFHQ manifest 跑約 200 步，比較 `events.jsonl` 的 step 100 與 step 200 時差，再加上定期驗證及 checkpoint 時間。最大可用 batch 和最高 `images_per_sec` 的 batch 可能不同。固定 100,000 steps 時，batch 4 會處理 batch 1 四倍的訓練 patch。跨分支做受控比較時固定 batch 和 steps。

## 5. 推論與測試

對單張 LR 檔案或一個 LR 資料夾推論。`--tile` 以 LR 像素為單位，須為 8 的倍數，使用 `--tile 0` 時會整張推論。

```sh
python baseline.py infer --weights runs/baseline/best.pth --input data/ffhq/test/lr --out outputs/inference --tile 128
```

用同一批 test LR 比較原始 pretrained 與微調後的模型。輸出 SR PNG、逐張 CSV、平均分數及 checkpoint／manifest SHA-256。舊環境先重新執行 `python -m pip install -r requirements.txt` 和 `python prepare.py assets`，以取得新增的評估套件與 BasicSR NIQE 參數。

```sh
python baseline.py evaluate --data data/ffhq --split test --out outputs/pretrained --tile 128
python baseline.py evaluate --data data/ffhq --split test --weights runs/baseline/best.pth --out outputs/finetuned --tile 128
```

若目錄中已有相同 checkpoint、manifest、split 與 tile 設定產生的報告，可加 `--reuse-sr` 補算指標並覆寫 CSV 和 JSON。此流程會沿用既有的 SR PNG 和前四項逐張分數。

```sh
python baseline.py evaluate --data data/ffhq --split test --out outputs/pretrained --reuse-sr
python baseline.py evaluate --data data/ffhq --split test --weights runs/baseline/best.pth --out outputs/finetuned --reuse-sr
```

| 指標 | 設定 | 輸出 |
| --- | --- | --- |
| PSNR、SSIM、LPIPS、DISTS | 完整 8-bit RGB SR/HR 配對。PSNR 和 SSIM 不切邊，LPIPS 使用 AlexNet | 逐張與平均 |
| MS-SSIM ↑ | 五尺度，8-bit RGB SR/HR 配對 | 逐張與平均 |
| NIQE ↓ | BasicSR pristine 參數，SR 轉為 YCbCr 的 Y 通道 | 逐張與平均 |
| FID、KID ↓ | Torch-Fidelity 相容 Inception-v3 的 2048 維特徵。比較整組 SR 與 HR，KID 使用 100 次子抽樣，每次 100 張或取可用的較小張數，seed 42 | 整組結果，KID 另存子抽樣標準差 |

首次算 FID 或 KID 會下載 Inception-v3 權重。集合少於 2 張時，FID 和 KID 回傳 `null`。125 張的 FID 受樣本量影響，應以同一份 test manifest 比較。[Torch-Fidelity](https://torch-fidelity.readthedocs.io/en/v0.3.0/api.html) · [MS-SSIM](https://github.com/VainF/pytorch-msssim) · [BasicSR NIQE](https://github.com/XPixelGroup/BasicSR/blob/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/basicsr/metrics/niqe.py) · [FID 樣本量偏差](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_Them_CVPR_2020_paper.html)

訓練期間記錄 PSNR、SSIM 和 LPIPS。Spec 的 IFQA、DSL-FIQA 可對上述相同的 `sr/` 影像另外執行。團隊比較前應固定這兩項的 checkpoint、臉部前處理與逐張分數彙總方式。[IFQA 官方實作](https://github.com/AIM-SKKU/IFQA) · [DSL-FIQA 官方實作](https://github.com/DSL-FIQA/DSL-FIQA)

## FFT 實驗

`--fft` 在 SwinIR 的 `patch_unembed` 輸出與 `conv_after_body` 之間加上一條 FFT 殘差分支。先嚴格載入官方 checkpoint，再加入分支。分支以 `1×1 Conv` 將 180 個通道降到 32，在 FP32 執行 `rfft2`、實／虛部的通道混合與 `irfft2`，最後投影回 180 個通道。末層的權重與偏置初始化為 0。分支初始化使用獨立的 RNG 範圍，保留 baseline 的判別器初始化與訓練 RNG 順序。

同一份 FFHQ manifest、原始權重與 loss 設定下跑 100,000 steps。驗證每 5,000 步執行，仍以 validation LPIPS 選 `best.pth`。續訓時在原訓練指令加 `--resume`。推論及評估會從 checkpoint 自動辨識 FFT 分支。

```sh
python baseline.py train --fft --data data/ffhq --run runs/fft --steps 100000 --batch 1 --workers 2 --val-every 5000
python baseline.py evaluate --data data/ffhq --split test --weights runs/fft/best.pth --out outputs/fft
```

和 baseline 比較時使用同一個 `--tile`。FFT 在各次 forward 的整張特徵圖上運作，分塊推論時則以各塊作為變換範圍。可用 `python probe_batch.py --fft --max-batch 32` 測這個變體的 batch 上限。

## DWT 實驗

`--dwt` 在同一位置加入單層 Haar 殘差分支。先將 180 個通道投影成 32 個通道，再以 FP32 分解成 `LL`、`LH`、`HL`、`HH` 四個子頻帶。分組 `1×1 Conv` 分別混合各頻帶的通道，接 LeakyReLU 與 IDWT，投影回 180 個通道後加回原特徵。分支末層以零初始化。DWT 新增 15,956 個參數，FFT 分支為 15,892 個。DWT 與 IDWT 用原生 PyTorch 運算，推論及評估從 checkpoint 自動載入對應架構。[二維 DWT 與 IDWT](https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html)

正式比較可沿用原先的 FFHQ manifest、官方權重、loss、100,000 steps 和 validation LPIPS 選模規則。只在續訓時加 `--resume`。

```sh
python probe_batch.py --dwt --max-batch 32
python baseline.py train --dwt --data data/ffhq --run runs/dwt --steps 100000 --batch 1 --workers 2 --val-every 5000
python baseline.py evaluate --data data/ffhq --split test --weights runs/dwt/best.pth --out outputs/dwt
```

## SwinFR + DWT 深度實驗

`--num-groups 1` 先嚴格載入官方 SwinIR-M，保留第 1、2 個 RSTB 的完整權重與原本的 DropPath 設定，再接末端 DWTB。對照組使用 `--dwt`。實驗組使用 `--swinfr`，在每個保留 block 的原 Conv 旁加 Fourier 分支，並在每對 block 之後加 `z + ConvBranch(z - x)`，其中 `x`、`z` 是該對的輸入與輸出。新增分支末層皆為零初始化，兩組在 step 0 輸出一致。

`--num-groups 2` 和 `--num-groups 3` 分別保留前 4 和前 6 個 RSTB。推論及評估從 checkpoint 自動辨識深度與分支。續訓時沿用訓練旗標並加 `--resume`。

```sh
python baseline.py train --dwt --num-groups 1 --data data/ffhq --run runs/dwt_n1 --steps 100000 --batch 1 --workers 2 --val-every 5000
python baseline.py train --swinfr --num-groups 1 --data data/ffhq --run runs/swinfr_dwt_n1 --steps 100000 --batch 1 --workers 2 --val-every 5000
python baseline.py evaluate --data data/ffhq --split test --weights runs/dwt_n1/best.pth --out outputs/dwt_n1
python baseline.py evaluate --data data/ffhq --split test --weights runs/swinfr_dwt_n1/best.pth --out outputs/swinfr_dwt_n1
```

兩個 `n=1` 版本的生成器分別有 4,263,675 和 4,316,439 個參數。要比較 GPU 訓練步速，可分別用 `python probe_batch.py --dwt --num-groups 1 --batch 1` 和 `python probe_batch.py --swinfr --num-groups 1 --batch 1`。
