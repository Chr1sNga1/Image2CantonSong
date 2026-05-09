# Image2CantonSong  
# 圖像到粵語歌曲生成系統

**Image2CantonSong** is an end-to-end multimodal song generation project that converts an input image into Cantonese lyrics and then generates a song through a lyrics-to-music pipeline.

**Image2CantonSong** 是一個端到端的多模態歌曲生成系統，目標是將使用者輸入的圖片轉換為粵語歌詞，並進一步通過音樂生成模型產生歌曲。

The project combines image understanding, Cantonese lyric generation, retrieval-augmented generation, prompt engineering, music generation, and multiple evaluation modules.

本項目結合了圖像理解、粵語歌詞生成、檢索增強生成、Prompt Engineering、音樂生成，以及多個評估模組。

---

## Table of Contents  
## 目錄

- [Project Overview / 項目概述](#project-overview--項目概述)
- [Main Workflow / 主要流程](#main-workflow--主要流程)
- [Key Features / 主要功能](#key-features--主要功能)
- [Repository Structure / 倉庫結構](#repository-structure--倉庫結構)
- [Environment Setup / 環境配置](#environment-setup--環境配置)
- [Running the Demo / 啟動 Demo](#running-the-demo--啟動-demo)
- [Usage Guide / 使用方法](#usage-guide--使用方法)
- [Lyrics Format / 歌詞格式要求](#lyrics-format--歌詞格式要求)
- [Evaluation Modules / 評估模組](#evaluation-modules--評估模組)
- [Technologies Used / 使用技術](#technologies-used--使用技術)
- [Troubleshooting / 常見問題](#troubleshooting--常見問題)
- [Known Limitations / 已知限制](#known-limitations--已知限制)
- [Future Improvements / 未來改進方向](#future-improvements--未來改進方向)
- [Acknowledgement / 致謝](#acknowledgement--致謝)
- [License and Usage / 授權與使用說明](#license-and-usage--授權與使用說明)
- [Project Status / 項目狀態](#project-status--項目狀態)

---

## Project Overview / 項目概述

Image2CantonSong is designed as a research-oriented prototype for image-to-song generation. Given a user-uploaded image, the system extracts visual objects, scene information, mood, and emotional cues. These visual signals are then used to generate Cantonese lyrics and a corresponding genre prompt. After manual review and editing, the lyrics and prompt can be passed to the YuE-based music generation pipeline to produce a song.

Image2CantonSong 是一個面向研究與課程展示的圖像到歌曲生成原型系統。使用者上傳圖片後，系統會提取圖片中的物體、場景、氛圍與情緒線索，並基於這些視覺資訊生成粵語歌詞與音樂風格 Prompt。使用者可以人工檢查與修改生成結果，最後將歌詞與 Prompt 傳入基於 YuE 的音樂生成流程，產生完整歌曲。

The project focuses especially on Cantonese lyric generation. Compared with general Chinese lyric generation, Cantonese lyrics require more attention to Hong Kong-style expressions, natural Cantonese wording, lyric structure, and singability.

本項目特別關注粵語歌詞生成。相比一般中文歌詞生成，粵語歌詞更需要考慮香港語境、粵語用詞自然度、歌詞結構，以及是否適合演唱。

---

## Main Workflow / 主要流程

```text
Image Upload
    ↓
Multimodal Image Understanding
    ↓
Visual Anchor, Mood and Scene Extraction
    ↓
Cantonese Lyrics Generation
    ↓
Genre Prompt Generation / Style Prompt Selection
    ↓
Manual Review and Editing
    ↓
Evaluation
    ↓
YuE-based Music Generation
    ↓
Generated Cantonese Song
```

中文流程：

```text
圖片上傳
    ↓
多模態圖像理解
    ↓
提取視覺主題、場景與情緒
    ↓
生成粵語歌詞
    ↓
生成或選擇音樂風格 Prompt
    ↓
人工確認與修改
    ↓
評估
    ↓
基於 YuE 的歌曲生成
    ↓
輸出粵語歌曲
```

---

## Key Features / 主要功能

### 1. Image-to-Lyrics Generation / 圖像到歌詞生成

The system accepts an image as input and generates Cantonese lyrics based on the visual content.

系統支援以上傳圖片作為輸入，並根據圖片中的視覺內容生成粵語歌詞。

The generated lyrics are expected to reflect:

生成的歌詞應能反映：

- Main objects in the image  
  圖片中的主要物件
- Scene and background  
  場景與背景
- Emotional atmosphere  
  情緒氛圍
- Possible story or theme  
  潛在故事與主題
- Cantonese lyric style  
  粵語歌詞風格

---

### 2. Multimodal LLM Support / 多模態大語言模型支援

The project supports multimodal language models for image understanding and lyric generation.

本項目支援使用多模態大語言模型進行圖像理解與歌詞生成。

These models are used to:

這些模型主要用於：

- Understand uploaded images  
  理解上傳圖片內容
- Extract visual and emotional cues  
  提取視覺與情緒線索
- Generate structured Cantonese lyrics  
  生成結構化粵語歌詞
- Generate or refine genre prompts  
  生成或優化音樂風格 Prompt

---

### 3. Cantonese Lyric Generation / 粵語歌詞生成

The project emphasizes Cantonese-style lyric generation rather than generic Mandarin or written Chinese output.

本項目重點是生成粵語風格歌詞，而不是一般普通話或書面中文歌詞。

The generated lyrics are expected to use:

生成結果應盡量使用：

- Traditional Chinese characters  
  繁體中文
- Cantonese expressions  
  粵語表達
- Hong Kong-style lyric wording  
  香港流行歌詞風格
- Structured lyric sections  
  清晰的歌詞段落結構

---

### 4. Retrieval-Augmented Generation / 檢索增強生成

The project includes a Cantopop lyric corpus for retrieval-augmented generation.

本項目包含粵語流行歌語料，可用於檢索增強生成。

The retrieval module can retrieve similar lyric examples and provide them as references to the generation model.

檢索模組可以根據當前圖片或生成需求，找出相似的歌詞示例，並將其作為 few-shot 參考提供給模型。

This helps improve:

這有助於提升：

- Lyric style consistency  
  歌詞風格一致性
- Cantonese naturalness  
  粵語自然度
- Section structure  
  段落結構
- Theme relevance  
  主題相關性

---

### 5. Genre Prompt Control / 音樂風格 Prompt 控制

The system supports different ways to control the final music style.

系統支援多種方式控制最終歌曲風格。

Supported prompt modes include:

支援的 Prompt 模式包括：

- Preset genre prompt  
  預設音樂風格 Prompt
- Tag-based style selection  
  基於標籤選擇風格
- Model-generated genre prompt  
  由模型自動生成風格 Prompt
- Manually edited prompt  
  使用者人工修改 Prompt

---

### 6. Manual Review and Editing / 人工檢查與修改

Before music generation, users can review and edit the generated content.

在進入音樂生成之前，使用者可以檢查並修改生成內容。

Editable fields may include:

可修改內容包括：

- Song title  
  歌曲標題
- Lyrics  
  歌詞
- Genre prompt  
  音樂風格 Prompt
- Style hints  
  風格提示詞

This design allows the system to combine automatic generation with human control.

這種設計可以結合自動生成能力與人工控制，避免直接將不穩定輸出送入音樂生成模型。

---

### 7. YuE Music Generation Bridge / YuE 音樂生成橋接

The project uses a YuE-based pipeline for lyrics-to-music generation.

本項目使用基於 YuE 的流程進行歌詞到音樂生成。

The Streamlit demo is designed as a bridge. It prepares lyrics and prompt files, then calls the YuE inference environment externally.

Streamlit Demo 作為橋接層使用，負責準備歌詞與 Prompt 文件，然後外部調用 YuE 推理環境。

The demo does not directly modify the original YuE source code.

Demo 不直接修改 YuE 原始碼，便於維持 YuE 官方環境的獨立性。

---

### 8. Evaluation Modules / 評估模組

The repository contains multiple evaluation modules for checking the quality of generated lyrics and songs.

倉庫中包含多個評估模組，用於檢查生成歌詞與歌曲的質量。

Evaluation aspects include:

評估方向包括：

- Image-lyrics semantic alignment  
  圖像與歌詞語義一致性
- Image-lyrics emotion similarity  
  圖像與歌詞情緒一致性
- Lyrics format correctness  
  歌詞格式正確性
- Cantonese lyric quality  
  粵語歌詞質量
- Genre prompt alignment  
  音樂風格 Prompt 一致性

---

## Repository Structure / 倉庫結構

```text
Image2CantonSong/
├── canto_project_official_yue_bridge_demo_v2/
│   ├── app.py
│   ├── generator.py
│   ├── schemas.py
│   ├── state_utils.py
│   ├── launch.sh
│   ├── README.md
│   ├── examples/
│   ├── modules/
│   └── outputs/
│
├── Evaluation/
│   ├── genre_alignment/
│   ├── image_lyrics_alignment/
│   ├── image_lyrics_emotion/
│   ├── lyrics_format/
│   └── lyrics_quality/
│
├── Emotion/
├── Dataset/
├── Images/
├── Image-Prompt-Pairs/
├── Song_evaluation/
├── YuE/
├── envs/
│   ├── yue_project_clean/
│   └── clip-e/
│
├── cantopop_corpus_final_583_yue.csv
├── finetune_internvl2_4b_cantopop.ipynb
├── rag_internvl2_4b_cantopop.ipynb
├── paths.py
└── README.md
```

### Main directories / 主要目錄說明

| Path | English Description | 中文說明 |
|---|---|---|
| `canto_project_official_yue_bridge_demo_v2/` | Main Streamlit demo | 主要 Streamlit Demo |
| `Evaluation/` | Evaluation modules | 評估模組 |
| `Emotion/` | Emotion-related models and scripts | 情緒分析相關模組 |
| `Dataset/` | Dataset resources | 數據集資源 |
| `Images/` | Image examples or assets | 圖片示例或資源 |
| `Image-Prompt-Pairs/` | Image-prompt data | 圖片與 Prompt 配對數據 |
| `Song_evaluation/` | Song evaluation resources | 歌曲評估相關資源 |
| `YuE/` | YuE-related files or integration resources | YuE 相關文件或整合資源 |
| `envs/` | Conda environment specifications | Conda 環境配置文件 |
| `cantopop_corpus_final_583_yue.csv` | Cantopop lyric corpus for RAG | 用於 RAG 的粵語歌詞語料 |
| `paths.py` | Project path configuration | 項目路徑配置 |

---

## Environment Setup / 環境配置

This project uses multiple Conda environments because different modules have different dependency requirements.

本項目使用多個 Conda 環境，因為 Streamlit Demo、YuE 音樂生成、CLIP 評估等部分對依賴版本有不同要求。

---

### 1. Clone the repository / 下載倉庫

```bash
git clone https://github.com/Chr1sNga1/Image2CantonSong.git
cd Image2CantonSong
```

---

### 2. Create the main Streamlit environment / 建立主要 Streamlit 環境

The main environment is used for the Streamlit demo and image-to-lyrics generation.

主要環境用於運行 Streamlit Demo 以及圖像到歌詞生成模組。

```bash
cd envs

conda env create -f ./yue_project_clean/yue_project_clean.yml
conda activate yue_project_clean
```

If additional packages are required:

如果仍有缺失依賴，可安裝 requirements 文件：

```bash
pip install -r ./yue_project_clean/yue_project_clean-requirements.txt
```

If CUDA-specific PyTorch wheels are required, install them according to your own CUDA version.

如果需要 CUDA 版本的 PyTorch，請根據本機 CUDA 版本安裝對應版本。

Example:

```bash
pip install -r ./yue_project_clean/yue_project_clean-requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

---

### 3. Set up the YuE environment / 配置 YuE 環境

The final music generation step depends on the official YuE environment.

最終音樂生成部分依賴 YuE 官方環境。

Please follow the official YuE installation instructions and make sure the YuE inference script can run independently before connecting it with this demo.

請先按照 YuE 官方說明完成安裝，並確認 YuE 推理腳本可以獨立運行，再將其與本 Demo 連接。

The demo calls YuE through an external Python environment by subprocess.

Demo 通過 subprocess 調用外部 YuE Python 環境，而不是直接在 Streamlit 環境中運行 YuE。

You may need to adjust the following paths in the project according to your own machine:

你可能需要根據自己的機器修改以下路徑：

```text
Path to YuE Python executable
Path to YuE inference script
Path to output directory
Path to lyrics and genre prompt files
```

---

### 4. Optional CLIP evaluation environment / 可選：CLIP 評估環境

Some evaluation modules may require a separate environment, such as `clip-e`.

部分評估模組可能需要獨立環境，例如 `clip-e`。

Example:

```bash
cd envs

conda create -n clip-e python=3.11
conda activate clip-e

pip install -r ./clip-e/clip-e-requirements.txt
```

If CUDA and cuDNN are required:

如果需要 CUDA 與 cuDNN：

```bash
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit
conda install -c conda-forge cudnn=8.8 cuda-version=11.8
```

---

### 5. Hugging Face token / Hugging Face Token

Some models may require access to Hugging Face.

部分模型可能需要 Hugging Face 權限。

If required, set your token:

如有需要，可設置：

```bash
export HF_TOKEN=your_huggingface_token
```

or:

```bash
huggingface-cli login
```

---

## Running the Demo / 啟動 Demo

Activate the main environment:

啟動主要環境：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yue_project_clean
```

Go to the demo directory:

進入 Demo 目錄：

```bash
cd canto_project_official_yue_bridge_demo_v2
```

Install local requirements if needed:

如有需要，安裝 Demo 依賴：

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

啟動 Streamlit 應用：

```bash
streamlit run app.py --server.port 8501 --server.address 127.0.0.1
```

For debug mode:

Debug 模式：

```bash
streamlit run app.py --server.port 8501 --server.address 127.0.0.1 -- --debug
```

For presentation mode:

展示模式：

```bash
streamlit run app.py --server.port 8501 --server.address 127.0.0.1 -- --presentation
```

If port `8501` is already in use, use another port:

如果 `8501` 端口已被佔用，可改用其他端口：

```bash
streamlit run app.py --server.port 8502 --server.address 127.0.0.1
```

---

## Usage Guide / 使用方法

### English

1. Open the Streamlit web interface.
2. Upload an image.
3. Select a multimodal model.
4. Choose a lyric generation mode.
5. Choose a genre prompt mode:
   - Preset prompt
   - Tag-based prompt
   - Model-generated prompt
6. Select the desired lyric length.
7. Click the generation button to generate lyrics and genre prompt.
8. Review and edit the generated title, lyrics, and prompt.
9. Run evaluation modules if needed.
10. Send the final lyrics and prompt to the YuE generation pipeline.
11. Check the generated audio output.

### 中文

1. 打開 Streamlit 網頁界面。
2. 上傳圖片。
3. 選擇多模態模型。
4. 選擇歌詞生成模式。
5. 選擇音樂風格 Prompt 模式：
   - 預設 Prompt
   - 標籤選擇 Prompt
   - 模型自動生成 Prompt
6. 選擇歌詞長度。
7. 點擊生成按鈕，生成歌詞與音樂風格 Prompt。
8. 檢查並修改歌曲標題、歌詞與 Prompt。
9. 如有需要，運行評估模組。
10. 將最終歌詞與 Prompt 傳入 YuE 生成流程。
11. 查看生成的音頻結果。

---

## Lyrics Format / 歌詞格式要求

The generated lyrics should follow a clear structure.

生成歌詞應遵循清晰結構。

Example:

```text
[verse]
望住霓虹慢慢散
街角風聲又再返

[chorus]
如果今晚仲有夢
可唔可以留低一分鐘

[end]
```

Recommended rules:

建議規則：

| Rule | English | 中文 |
|---|---|---|
| Language | Use Traditional Chinese | 使用繁體中文 |
| Style | Use Cantonese expressions | 使用粵語表達 |
| Section tags | Use lowercase tags such as `[verse]` and `[chorus]` | 使用小寫段落標籤，例如 `[verse]`、`[chorus]` |
| Ending | `[end]` should be the final non-empty line | `[end]` 應為最後一個非空行 |
| Formatting | Avoid bullet points, numbering, or explanations inside lyrics | 歌詞中不要加入項目符號、編號或解釋 |
| Blank lines | Use blank lines between sections | 段落之間使用空行 |
| Content | Lyrics should reflect the image content and emotion | 歌詞應反映圖片內容與情緒 |

Common section tags:

常見段落標籤：

```text
[verse]
[chorus]
[bridge]
[outro]
[end]
```

---

## Evaluation Modules / 評估模組

The repository contains several evaluation modules under the `Evaluation/` directory.

倉庫中的 `Evaluation/` 目錄包含多個評估模組。

---

### 1. Image-Lyrics Alignment / 圖像與歌詞語義一致性

This module evaluates whether the generated lyrics are semantically related to the input image.

該模組用於評估生成歌詞是否與輸入圖片在語義上相關。

Possible methods include CLIP-based image-text similarity.

可使用基於 CLIP 的圖文相似度方法進行評估。

---

### 2. Image-Lyrics Emotion Similarity / 圖像與歌詞情緒一致性

This module evaluates whether the emotion expressed in the lyrics matches the emotional atmosphere of the image.

該模組用於評估歌詞中的情緒是否與圖片氛圍一致。

For example, a sunset image may be expected to generate lyrics with nostalgic, calm, or emotional tones.

例如，日落圖片可能更適合生成帶有懷舊、平靜或抒情氛圍的歌詞。

---

### 3. Lyrics Format Evaluation / 歌詞格式評估

This module checks whether the generated lyrics follow the required structure.

該模組用於檢查生成歌詞是否符合指定格式。

It may check:

可能檢查：

- Whether required tags exist  
  是否包含必要標籤
- Whether `[end]` appears correctly  
  `[end]` 是否正確出現
- Whether blank lines are valid  
  空行是否符合要求
- Whether the output contains unwanted explanations  
  是否包含不應出現的解釋文字

---

### 4. Cantonese Lyrics Quality Evaluation / 粵語歌詞質量評估

This module evaluates the quality of Cantonese lyrics.

該模組用於評估粵語歌詞質量。

Possible criteria include:

可能標準包括：

- Cantonese naturalness  
  粵語自然度
- Lyric fluency  
  歌詞流暢度
- Thematic consistency  
  主題一致性
- Singability  
  可唱性
- Emotional expression  
  情緒表達

---

### 5. Genre Prompt Alignment / 音樂風格 Prompt 一致性

This module evaluates whether the genre prompt matches the lyrics and intended song style.

該模組用於評估音樂風格 Prompt 是否與歌詞內容和目標歌曲風格一致。

---

## Technologies Used / 使用技術

| Category | English | 中文 |
|---|---|---|
| Programming | Python | Python 編程 |
| Web Demo | Streamlit | Streamlit 網頁 Demo |
| Deep Learning | PyTorch | PyTorch 深度學習框架 |
| Model Library | Hugging Face Transformers | Hugging Face Transformers 模型庫 |
| Multimodal AI | Vision-Language Models | 視覺語言模型 |
| Generation | Large Language Models | 大語言模型 |
| Retrieval | Retrieval-Augmented Generation | 檢索增強生成 |
| Evaluation | CLIP-based similarity | 基於 CLIP 的相似度評估 |
| Music Generation | YuE | YuE 音樂生成 |
| Environment | Conda | Conda 環境管理 |

---

## Troubleshooting / 常見問題

### 1. Port already in use / 端口被佔用

If Streamlit reports that the port is already in use:

如果 Streamlit 顯示端口已被佔用：

```bash
streamlit run app.py --server.port 8502 --server.address 127.0.0.1
```

---

### 2. CUDA is not available / 無法使用 CUDA

Check whether PyTorch can detect GPU:

檢查 PyTorch 是否可以檢測 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If the output is `False`, check:

如果輸出為 `False`，請檢查：

- NVIDIA driver  
  NVIDIA 驅動
- CUDA version  
  CUDA 版本
- PyTorch installation  
  PyTorch 安裝版本
- Conda environment  
  Conda 環境

---

### 3. Hugging Face token error / Hugging Face Token 錯誤

If a model requires Hugging Face authentication:

如果模型需要 Hugging Face 驗證：

```bash
export HF_TOKEN=your_huggingface_token
```

or:

```bash
huggingface-cli login
```

---

### 4. YuE does not generate vocals / YuE 沒有生成有效人聲

Possible reasons:

可能原因包括：

- Incorrect lyrics format  
  歌詞格式不正確
- Genre prompt not suitable  
  音樂風格 Prompt 不合適
- YuE environment configuration issue  
  YuE 環境配置問題
- Insufficient generation length  
  生成長度不足
- GPU memory limitation  
  GPU 記憶體不足

Try simplifying the lyrics and genre prompt first.

可以先嘗試簡化歌詞與音樂風格 Prompt。

---

## Known Limitations / 已知限制

### English

- Cantonese singing quality depends heavily on the downstream music generation model.
- Generated lyrics may still require manual editing.
- Some multimodal models may not fully understand complex image details.
- Long lyric generation may be unstable under limited GPU memory.
- YuE generation can be computationally expensive.
- Some models require large GPU memory and additional access permissions.
- Cantonese pronunciation in generated audio may not always be accurate.
- Evaluation results should be treated as reference indicators rather than absolute judgments.

### 中文

- 粵語演唱質量高度依賴下游音樂生成模型。
- 生成歌詞仍可能需要人工修改。
- 部分多模態模型對複雜圖片細節理解不足。
- 在 GPU 記憶體有限的情況下，長歌詞生成可能不穩定。
- YuE 生成計算成本較高。
- 部分模型需要較大 GPU 記憶體或額外權限。
- 生成音頻中的粵語發音未必完全準確。
- 評估結果應作為參考指標，而不是絕對判斷。

---

## Future Improvements / 未來改進方向

### English

Future work may include:

- Improve Cantonese pronunciation in generated songs.
- Improve lyric-to-melody alignment.
- Add more controllable music styles.
- Improve RAG retrieval quality with a larger Cantopop corpus.
- Add automatic lyric format repair.
- Add automatic prompt optimization.
- Improve the stability of long-song generation.
- Provide a cleaner deployment configuration.
- Add more quantitative and human evaluation metrics.
- Support more image styles, such as social media photos, landscapes, portraits, and illustrations.

### 中文

未來可以改進的方向包括：

- 提升生成歌曲中的粵語發音質量。
- 改進歌詞與旋律的匹配度。
- 增加更多可控音樂風格。
- 使用更大的粵語流行歌語料提升 RAG 檢索質量。
- 增加自動歌詞格式修復功能。
- 增加自動 Prompt 優化功能。
- 提升長歌曲生成穩定性。
- 提供更清晰的部署配置。
- 增加更多量化評估與人工評估指標。
- 支援更多圖片類型，例如社交媒體圖片、風景、人像與插畫。

---

## Acknowledgement / 致謝

This project builds upon open-source tools and models from the multimodal generation, language modeling, information retrieval, and music generation communities.

本項目基於多模態生成、大語言模型、資訊檢索與音樂生成社群中的開源工具和模型構建。

Special thanks to the developers and researchers of:

特別感謝以下相關工具與研究方向的開發者和研究者：

- Streamlit
- PyTorch
- Hugging Face Transformers
- CLIP and image-text similarity models
- Multimodal large language models
- YuE music generation framework
- Retrieval-Augmented Generation methods

---

## License and Usage / 授權與使用說明

This repository is mainly intended for academic research, coursework, prototyping, and demonstration purposes.

本倉庫主要用於學術研究、課程項目、原型開發與展示用途。

Please check the licenses of all third-party models, datasets, and dependencies before using this project for commercial or public deployment.

如需將本項目用於商業用途或公開部署，請先檢查所有第三方模型、數據集與依賴庫的授權條款。

If no explicit license file is provided in this repository, please contact the repository owner before redistribution or reuse.

如果倉庫中未提供明確 License 文件，請在重新分發或復用前聯繫倉庫作者。

---

## Project Status / 項目狀態

This project is currently a research prototype.

本項目目前屬於研究原型階段。

The core pipeline has been implemented, including image understanding, Cantonese lyric generation, prompt preparation, YuE bridge integration, and evaluation modules.

目前已實現主要流程，包括圖像理解、粵語歌詞生成、Prompt 準備、YuE 橋接整合與評估模組。

Further improvements are still required for stable deployment and high-quality Cantonese singing generation.

若要實現穩定部署與高質量粵語演唱生成，仍需要進一步優化。
