# 🧠 AD 语音识别演示

基于多模态融合的阿尔茨海默病识别 Web 演示应用（Flask + 本项目模型）。

原版是 Whisper 转录应用，现已将后端替换为本项目的 **NCMMSC 多模态融合模型**
（语音 Wav2Vec2 + 声学 OpenSMILE + SVM soft-voting），将语音分类为 HC / MCI / AD。

## 功能
- 📤 拖拽上传语音（WAV / MP3 / M4A / OGG / FLAC），**仅接受 10–60 秒**的语音
- 🧠 三分类识别：HC（健康对照）/ MCI（轻度认知障碍）/ AD（阿尔茨海默病）
- 📊 概率条形图 + 置信度展示
- 🔬 可选各模态（语音 / 声学）概率明细
- 📥 结果复制 / 下载 .txt


## 项目结构
```
whisper-transcriber/
├── app.py            # Flask web server + API 路由
├── predictor.py      # AD 识别推理逻辑（复用 NCMMSC 模型）
├── index.html        # 前端 UI（HTML/CSS/JS）
├── requirements.txt  # Python 依赖
├── deploy/           # frp 公网穿透（frpc 二进制 + 配置 + 启动脚本）
└── README.md
```

## 依赖与模型就绪检查

推理复用 `../NCMMSC/` 下的科研代码与工件，需要以下就位：

| 资源 | 路径 | 说明 |
|------|------|------|
| 模型权重 | `../NCMMSC/save_models/paper_wav_seed49.pth` 或 `../paper_wav_seed49.pth` | 可用 `AD_MODEL_PATH` 覆盖 |
| 推理工件 | `../NCMMSC/artifacts/` | scaler + SVM + config |
| HF 缓存 | `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` | 本地离线加载 |

环境需安装：`torch transformers librosa opensmile scikit-learn flask numpy`。

## 运行

```bash
# 使用真实模型（默认）
python app.py

# 无权重时用 Mock 后端跑通 UI
AD_BACKEND=mock python app.py

# 指定权重路径 / GPU
AD_MODEL_PATH=/path/to/paper_wav_seed49.pth AD_GPU_ID=2 python app.py

# 自定义时长限制（默认 10–60 秒）
MIN_AUDIO_SECONDS=10 MAX_AUDIO_SECONDS=60 python app.py
```

打开 http://127.0.0.1:5000

## 音频时长限制

服务端强制要求上传语音时长为 **10–60 秒**（可通过 `MIN_AUDIO_SECONDS` / `MAX_AUDIO_SECONDS` 环境变量调整）。时长越界或无法解析的音频返回 `400` 并附中文错误说明；前端也会在选文件时预检一次。

## API

- `GET  /models` — 模型信息
- `POST /predict` — multipart 上传 `audio` 文件，可选 `detail=true` 返回各模态概率


## 推理流程
```
音频文件 → Flask 接收 (app.py) → 转 16kHz 单声道 WAV (ffmpeg)
    → Wav2Vec2 (192d 语音特征) ┐
    → OpenSMILE (6373d 声学特征) -> SVM soft-voting → HC / MCI / AD
    → JSON 响应 → UI 展示概率条形图
```
