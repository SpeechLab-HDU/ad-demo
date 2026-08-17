"""
predictor.py
============
AD 识别推理后端（替换原 Whisper 后端）。

复用本项目 NCMMSC 科研代码（api/extractors.py + artifacts + 微调权重）做
三模态融合推理:
    音频 → Wav2Vec2 (192维语音特征) ┐
           OpenSMILE (6373维声学特征) ┼→ SVM soft-voting → HC / MCI / AD
           BERT 文本模态在推理时不可用（仅语音/声学）  ┘

路径解析优先级:
    1. 环境变量 AD_MODEL_PATH（显式指定权重文件）
    2. <repo>/NCMMSC/save_models/paper_wav_seed49.pth
    3. <repo>/paper_wav_seed49.pth（根目录）

后端切换（演示用）:
    AD_BACKEND=ncmsc  真实模型（默认，需权重与工件就位）
    AD_BACKEND=mock   占位后端，返回演示结果，方便在没有权重时跑通全链路

任何加载失败都会抛 ModelNotLoadedError，由 app.py 捕获并以友好 JSON 返回。
"""

import os
import sys
import time

# ── 路径解析 ─────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NC_ROOT = os.path.join(REPO_ROOT, "NCMMSC")

DEFAULT_MODEL_CANDIDATES = [
    os.environ.get("AD_MODEL_PATH", ""),
    os.path.join(NC_ROOT, "save_models", "paper_wav_seed49.pth"),
    os.path.join(REPO_ROOT, "paper_wav_seed49.pth"),
]

ARTIFACT_DIR = os.environ.get("AD_ARTIFACT_DIR", "") or os.path.join(NC_ROOT, "artifacts")


class ModelNotLoadedError(Exception):
    """模型未加载完成（权重缺失/损坏/加载异常）。"""


# ── 模型信息（供 UI 展示） ────────────────────────────────────────────────────

def get_model_info() -> dict:
    """返回模型元信息，供前端展示与 /models 接口使用。"""
    backend = os.environ.get("AD_BACKEND", "ncmsc").strip().lower()
    info = {
        "backend": backend,
        "name": "NCMMSC 多模态融合 (Wav2Vec2 + OpenSMILE + SVM)",
        "description": "语音 Wav2Vec2 (192d) + 声学 OpenSMILE (6373d) → SVM soft-voting",
        "labels": ["HC", "MCI", "AD"],
        "task": "阿尔茨海默病识别（NCMMSC 2021，中文）",
    }
    if backend == "mock":
        info["name"] = "Mock 占位后端（演示 UI，无真实模型）"
        info["description"] = "返回固定演示结果，用于无权重时跑通前端全链路"
    return info


# ── 模型加载 ─────────────────────────────────────────────────────────────────

def _resolve_model_path() -> str:
    for path in DEFAULT_MODEL_CANDIDATES:
        if path and os.path.isfile(path):
            return path
    raise ModelNotLoadedError(
        f"未找到模型权重。请在下列位置放置 paper_wav_seed49.pth："
        f"{', '.join(p for p in DEFAULT_MODEL_CANDIDATES if p)}，"
        f"或设置环境变量 AD_MODEL_PATH 指向权重文件。"
    )


def load_model(model_size: str = None):
    """加载 AD 识别模型（真实 NCMMSC 或 Mock 后端）。"""
    backend = os.environ.get("AD_BACKEND", "ncmsc").strip().lower()
    if backend == "mock":
        print("[Predictor] Mock 后端：不加载真实模型。")
        return "mock"

    if not os.path.isdir(ARTIFACT_DIR):
        raise ModelNotLoadedError(f"推理工件目录不存在: {ARTIFACT_DIR}（缺少 artifacts/）")

    model_path = _resolve_model_path()
    print(f"[Predictor] 加载权重: {model_path}")
    print(f"[Predictor] 使用工件目录: {ARTIFACT_DIR}")

    if NC_ROOT not in sys.path:
        sys.path.insert(0, NC_ROOT)

    # 用本服务配置覆盖科研代码的默认路径/超参
    import api.config as nc_config
    nc_config.MODEL_PATH = model_path
    nc_config.ARTIFACT_DIR = ARTIFACT_DIR
    nc_config.GPU_ID = os.environ.get("AD_GPU_ID", nc_config.GPU_ID)
    nc_config.SAMPLE_RATE = int(os.environ.get("AD_SAMPLE_RATE", str(nc_config.SAMPLE_RATE)))
    nc_config.MAX_LEN = int(os.environ.get("AD_MAX_LEN", str(nc_config.MAX_LEN)))
    nc_config.K_SEGMENTS = int(os.environ.get("AD_K_SEGMENTS", str(nc_config.K_SEGMENTS)))

    from api.predictor import Predictor, torch_device

    device = torch_device(nc_config.GPU_ID)
    try:
        predictor = Predictor.get_instance(device)
    except Exception as e:
        # 权重损坏/不完整是最常见的失败原因（zip 中央目录缺失）
        raise ModelNotLoadedError(
            f"模型权重加载失败: {e}\n"
            f"检查文件是否完整: {model_path}\n"
            f"（若文件被截断，请重新完整拷贝后再启动）"
        )
    print(f"[Predictor] 模型加载完成 (device={device})")
    return predictor


# ── 推理 ─────────────────────────────────────────────────────────────────────

def classify_audio(audio_path: str, model, detail: bool = False) -> dict:
    """
    对一段音频做 AD 三分类。

    Args:
        audio_path: 音频文件路径（.wav 最佳，其他格式由 ffmpeg 转 16k 单声道）
        model:      load_model() 返回的预测器（或 "mock" 字符串）
        detail:     是否返回各模态（wav / opensmile）概率明细

    Returns:
        dict: {label, probs, confidence, duration_seconds, processing_time, [detail]}
    """
    start_time = time.time()

    if model == "mock":
        import numpy as np
        duration_seconds = _audio_duration(audio_path)
        probs = {"HC": 0.82, "MCI": 0.12, "AD": 0.06}
        result = {
            "label": "HC",
            "probs": probs,
            "confidence": probs["HC"],
            "duration_seconds": round(duration_seconds, 2),
            "processing_time": round(time.time() - start_time, 2),
        }
        if detail:
            result["detail"] = {
                "wav_probs": {"HC": 0.85, "MCI": 0.10, "AD": 0.05},
                "comp_probs": {"HC": 0.79, "MCI": 0.14, "AD": 0.07},
            }
        return result

    if detail:
        raw = model.predict_detail(audio_path)
        result = {
            "label": raw["label"],
            "probs": raw["probs"],
            "confidence": raw["confidence"],
            "detail": {
                "wav_probs": raw["wav_probs"],
                "comp_probs": raw["comp_probs"],
            },
        }
    else:
        raw = model.predict(audio_path)
        result = {
            "label": raw["label"],
            "probs": raw["probs"],
            "confidence": raw["confidence"],
        }

    result["duration_seconds"] = round(_audio_duration(audio_path), 2)
    result["processing_time"] = round(time.time() - start_time, 2)
    return result


def _audio_duration(audio_path: str) -> float:
    """用 librosa 快速读取音频时长（秒）。"""
    try:
        import librosa
        d, sr = librosa.load(audio_path, sr=None, mono=True)
        return d.shape[0] / sr
    except Exception:
        return 0.0
