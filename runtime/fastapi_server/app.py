#!/usr/bin/env python3
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import safetensors.torch
import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from huggingface_hub import hf_hub_download
from lhotse.utils import fix_random_seed
from vocos import Vocos

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import (
    add_punctuation,
    batchify_tokens,
    chunk_tokens_punctuation,
    cross_fade_concat,
    remove_silence,
    rms_norm,
)
from zipvoice.utils.tensorrt import load_trt

HUGGINGFACE_REPO = "k2-fsa/ZipVoice"
MODEL_DIR = {
    "zipvoice": "zipvoice",
    "zipvoice_distill": "zipvoice_distill",
}

LOGGER = logging.getLogger("zipvoice.fastapi")


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class ZipVoiceEngine:
    def __init__(self) -> None:
        self.model_name = os.getenv("ZIPVOICE_MODEL_NAME", "zipvoice_distill")
        if self.model_name not in MODEL_DIR:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        self.model_dir = os.getenv("ZIPVOICE_MODEL_DIR", "")
        self.checkpoint_name = os.getenv("ZIPVOICE_CHECKPOINT_NAME", "model.pt")
        self.vocoder_path = os.getenv("ZIPVOICE_VOCODER_PATH", "")
        self.tokenizer_type = os.getenv("ZIPVOICE_TOKENIZER", "emilia")
        self.lang = os.getenv("ZIPVOICE_LANG", "en-us")
        self.trt_engine_path = os.getenv("ZIPVOICE_TRT_ENGINE_PATH", "")
        self.allow_hf_download = _read_bool_env("ZIPVOICE_ALLOW_HF_DOWNLOAD", True)

        self.target_rms = float(os.getenv("ZIPVOICE_TARGET_RMS", "0.1"))
        self.feat_scale = float(os.getenv("ZIPVOICE_FEAT_SCALE", "0.1"))
        self.default_speed = float(os.getenv("ZIPVOICE_DEFAULT_SPEED", "1.0"))
        self.default_t_shift = float(os.getenv("ZIPVOICE_DEFAULT_T_SHIFT", "0.5"))
        self.default_max_duration = float(os.getenv("ZIPVOICE_DEFAULT_MAX_DURATION", "100"))
        self.default_remove_long_sil = _read_bool_env("ZIPVOICE_DEFAULT_REMOVE_LONG_SIL", False)

        model_defaults = {
            "zipvoice": {"num_step": 16, "guidance_scale": 1.0},
            "zipvoice_distill": {"num_step": 8, "guidance_scale": 3.0},
        }
        self.default_num_step = int(
            os.getenv(
                "ZIPVOICE_DEFAULT_NUM_STEP",
                str(model_defaults[self.model_name]["num_step"]),
            )
        )
        self.default_guidance_scale = float(
            os.getenv(
                "ZIPVOICE_DEFAULT_GUIDANCE_SCALE",
                str(model_defaults[self.model_name]["guidance_scale"]),
            )
        )

        self.num_threads = int(os.getenv("ZIPVOICE_NUM_THREADS", "1"))
        self.max_concurrency = int(os.getenv("ZIPVOICE_MAX_CONCURRENCY", "1"))
        if self.max_concurrency < 1:
            raise ValueError("ZIPVOICE_MAX_CONCURRENCY must be >= 1")
        self.seed = int(os.getenv("ZIPVOICE_SEED", "666"))

        torch.set_num_threads(self.num_threads)
        torch.set_num_interop_threads(self.num_threads)
        fix_random_seed(self.seed)

        self.device = self._resolve_device(os.getenv("ZIPVOICE_DEVICE", "auto"))
        LOGGER.info("Using device: %s", self.device)

        model_ckpt, model_config_path, token_file = self._resolve_model_assets()

        self.tokenizer = self._build_tokenizer(token_file)
        tokenizer_config = {
            "vocab_size": self.tokenizer.vocab_size,
            "pad_id": self.tokenizer.pad_id,
        }

        with open(model_config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)

        if self.model_name == "zipvoice":
            model = ZipVoice(**model_config["model"], **tokenizer_config)
        else:
            model = ZipVoiceDistill(**model_config["model"], **tokenizer_config)

        if str(model_ckpt).endswith(".safetensors"):
            safetensors.torch.load_model(model, model_ckpt)
        elif str(model_ckpt).endswith(".pt"):
            load_checkpoint(filename=model_ckpt, model=model, strict=True)
        else:
            raise NotImplementedError(
                f"Unsupported model checkpoint format: {model_ckpt}"
            )

        self.model = model.to(self.device).eval()

        if self.trt_engine_path:
            load_trt(self.model, self.trt_engine_path)
            LOGGER.info("Loaded TensorRT engine: %s", self.trt_engine_path)

        self.vocoder = self._build_vocoder().to(self.device).eval()
        self.feature_extractor = VocosFbank()

        feature_type = model_config["feature"]["type"]
        if feature_type != "vocos":
            raise NotImplementedError(f"Unsupported feature type: {feature_type}")
        self.sampling_rate = int(model_config["feature"]["sampling_rate"])

        self._infer_semaphore = threading.BoundedSemaphore(self.max_concurrency)

    def _resolve_device(self, configured: str) -> torch.device:
        configured = configured.lower().strip()
        if configured != "auto":
            return torch.device(configured)
        if torch.cuda.is_available():
            return torch.device("cuda", 0)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_model_assets(self) -> Tuple[str, str, str]:
        if self.model_dir:
            model_dir = Path(self.model_dir)
            if not model_dir.is_dir():
                raise FileNotFoundError(f"Model dir does not exist: {model_dir}")

            model_ckpt = model_dir / self.checkpoint_name
            model_config = model_dir / "model.json"
            token_file = model_dir / "tokens.txt"

            for path in [model_ckpt, model_config, token_file]:
                if not path.is_file():
                    raise FileNotFoundError(f"Required file missing: {path}")
            return str(model_ckpt), str(model_config), str(token_file)

        if not self.allow_hf_download:
            raise ValueError(
                "ZIPVOICE_MODEL_DIR is empty and ZIPVOICE_ALLOW_HF_DOWNLOAD is false"
            )

        model_ckpt = hf_hub_download(
            HUGGINGFACE_REPO,
            filename=f"{MODEL_DIR[self.model_name]}/{self.checkpoint_name}",
        )
        model_config = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[self.model_name]}/model.json"
        )
        token_file = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[self.model_name]}/tokens.txt"
        )
        return model_ckpt, model_config, token_file

    def _build_tokenizer(self, token_file: str):
        if self.tokenizer_type == "emilia":
            return EmiliaTokenizer(token_file=token_file)
        if self.tokenizer_type == "libritts":
            return LibriTTSTokenizer(token_file=token_file)
        if self.tokenizer_type == "espeak":
            return EspeakTokenizer(token_file=token_file, lang=self.lang)
        if self.tokenizer_type == "simple":
            return SimpleTokenizer(token_file=token_file)
        raise ValueError(f"Unsupported tokenizer: {self.tokenizer_type}")

    def _build_vocoder(self):
        if self.vocoder_path:
            vocoder = Vocos.from_hparams(f"{self.vocoder_path}/config.yaml")
            state_dict = torch.load(
                f"{self.vocoder_path}/pytorch_model.bin",
                weights_only=True,
                map_location="cpu",
            )
            vocoder.load_state_dict(state_dict)
            return vocoder
        return Vocos.from_pretrained("charactr/vocos-mel-24khz")

    def _load_prompt_audio(self, wav_bytes: bytes) -> torch.Tensor:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            f.write(wav_bytes)

        try:
            wav, sr = torchaudio.load(temp_path)
        finally:
            os.remove(temp_path)

        if wav.numel() == 0:
            raise ValueError("Empty prompt wav")

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sampling_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sampling_rate)

        return wav.squeeze(0)

    def _serialize_wav_bytes(self, wav: torch.Tensor) -> bytes:
        wav = wav.unsqueeze(0).to(dtype=torch.float32).cpu()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name

        try:
            torchaudio.save(out_path, wav, sample_rate=self.sampling_rate)
            with open(out_path, "rb") as fr:
                return fr.read()
        finally:
            os.remove(out_path)

    @torch.inference_mode()
    def synthesize(
        self,
        prompt_wav_bytes: bytes,
        prompt_text: str,
        text: str,
        num_step: Optional[int],
        guidance_scale: Optional[float],
        speed: Optional[float],
        t_shift: Optional[float],
        max_duration: Optional[float],
        remove_long_sil: Optional[bool],
    ) -> Tuple[bytes, dict]:
        t0 = time.time()

        num_step = self.default_num_step if num_step is None else int(num_step)
        guidance_scale = (
            self.default_guidance_scale
            if guidance_scale is None
            else float(guidance_scale)
        )
        speed = self.default_speed if speed is None else float(speed)
        t_shift = self.default_t_shift if t_shift is None else float(t_shift)
        max_duration = (
            self.default_max_duration if max_duration is None else float(max_duration)
        )
        remove_long_sil = (
            self.default_remove_long_sil
            if remove_long_sil is None
            else bool(remove_long_sil)
        )

        prompt_wav = self._load_prompt_audio(prompt_wav_bytes)

        prompt_wav = remove_silence(
            prompt_wav,
            self.sampling_rate,
            only_edge=False,
            trail_sil=200,
        )
        prompt_wav, prompt_rms = rms_norm(prompt_wav, self.target_rms)

        prompt_duration = prompt_wav.shape[-1] / self.sampling_rate
        if prompt_duration > 20:
            LOGGER.warning(
                "Given prompt wav is too long (%.2fs), recommended 1-3s",
                prompt_duration,
            )
        elif prompt_duration > 10:
            LOGGER.warning(
                "Given prompt wav is long (%.2fs), this may slow down inference",
                prompt_duration,
            )

        prompt_features = self.feature_extractor.extract(
            prompt_wav, sampling_rate=self.sampling_rate
        ).to(self.device)
        prompt_features = prompt_features.unsqueeze(0) * self.feat_scale

        text = add_punctuation(text)
        prompt_text = add_punctuation(prompt_text)
        tokens_str = self.tokenizer.texts_to_tokens([text])[0]
        prompt_tokens_str = self.tokenizer.texts_to_tokens([prompt_text])[0]

        token_duration = (prompt_wav.shape[-1] / self.sampling_rate) / (
            len(prompt_tokens_str) * speed
        )
        max_tokens = max(1, int((25 - prompt_duration) / token_duration))
        chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)

        chunked_tokens = self.tokenizer.tokens_to_token_ids(chunked_tokens_str)
        prompt_tokens = self.tokenizer.tokens_to_token_ids([prompt_tokens_str])

        tokens_batches, chunked_index = batchify_tokens(
            chunked_tokens,
            max_duration=max_duration,
            prompt_duration=prompt_duration,
            token_duration=token_duration,
        )

        if not self._infer_semaphore.acquire(timeout=1200):
            raise RuntimeError("Server is busy")

        try:
            chunked_features: List[Tuple[torch.Tensor, torch.Tensor]] = []

            for batch_tokens in tokens_batches:
                batch_prompt_tokens = prompt_tokens * len(batch_tokens)
                batch_prompt_features = prompt_features.repeat(len(batch_tokens), 1, 1)
                batch_prompt_features_lens = torch.full(
                    (len(batch_tokens),),
                    prompt_features.size(1),
                    device=self.device,
                )

                (
                    pred_features,
                    pred_features_lens,
                    _,
                    _,
                ) = self.model.sample(
                    tokens=batch_tokens,
                    prompt_tokens=batch_prompt_tokens,
                    prompt_features=batch_prompt_features,
                    prompt_features_lens=batch_prompt_features_lens,
                    speed=speed,
                    t_shift=t_shift,
                    duration="predict",
                    num_step=num_step,
                    guidance_scale=guidance_scale,
                )

                pred_features = pred_features.permute(0, 2, 1) / self.feat_scale
                chunked_features.append((pred_features, pred_features_lens))

            chunked_wavs = []
            for pred_features, pred_features_lens in chunked_features:
                for i in range(pred_features.size(0)):
                    wav = (
                        self.vocoder.decode(
                            pred_features[i][None, :, : pred_features_lens[i]]
                        )
                        .squeeze(1)
                        .clamp(-1, 1)
                    )
                    if prompt_rms < self.target_rms:
                        wav = wav * prompt_rms / self.target_rms
                    chunked_wavs.append(wav)
        finally:
            self._infer_semaphore.release()

        indexed_chunked_wavs = [
            (index, wav) for index, wav in zip(chunked_index, chunked_wavs)
        ]
        sequential_indexed_chunked_wavs = sorted(indexed_chunked_wavs, key=lambda x: x[0])
        sequential_chunked_wavs = [
            sequential_indexed_chunked_wavs[i][1]
            for i in range(len(sequential_indexed_chunked_wavs))
        ]

        final_wav = cross_fade_concat(
            sequential_chunked_wavs,
            fade_duration=0.1,
            sample_rate=self.sampling_rate,
        )
        final_wav = remove_silence(
            final_wav,
            self.sampling_rate,
            only_edge=(not remove_long_sil),
            trail_sil=0,
        )

        duration_s = final_wav.shape[-1] / self.sampling_rate
        elapsed_s = time.time() - t0
        metrics = {
            "elapsed_s": elapsed_s,
            "wav_seconds": duration_s,
            "rtf": elapsed_s / max(duration_s, 1e-6),
            "num_step": num_step,
            "guidance_scale": guidance_scale,
            "speed": speed,
            "t_shift": t_shift,
            "max_duration": max_duration,
            "remove_long_sil": remove_long_sil,
        }

        wav_bytes = self._serialize_wav_bytes(final_wav)
        return wav_bytes, metrics


LOG_LEVEL = os.getenv("ZIPVOICE_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

APP = FastAPI(title="ZipVoice HTTP Server", version="1.0.0")
ENGINE = ZipVoiceEngine()


@APP.get("/healthz")
def healthz():
    return {
        "ok": True,
        "model_name": ENGINE.model_name,
        "device": str(ENGINE.device),
        "sampling_rate": ENGINE.sampling_rate,
        "max_concurrency": ENGINE.max_concurrency,
    }


@APP.post("/v1/clone")
def clone_voice(
    prompt_wav: UploadFile = File(..., description="Prompt wav file"),
    prompt_text: str = Form(..., description="Prompt transcription"),
    text: str = Form(..., description="Text to synthesize"),
    num_step: Optional[int] = Form(default=None),
    guidance_scale: Optional[float] = Form(default=None),
    speed: Optional[float] = Form(default=None),
    t_shift: Optional[float] = Form(default=None),
    max_duration: Optional[float] = Form(default=None),
    remove_long_sil: Optional[bool] = Form(default=None),
):
    if not prompt_text.strip():
        raise HTTPException(status_code=400, detail="prompt_text is empty")
    if not text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    try:
        wav_bytes = prompt_wav.file.read()
        if not wav_bytes:
            raise HTTPException(status_code=400, detail="prompt_wav is empty")

        audio_bytes, metrics = ENGINE.synthesize(
            prompt_wav_bytes=wav_bytes,
            prompt_text=prompt_text,
            text=text,
            num_step=num_step,
            guidance_scale=guidance_scale,
            speed=speed,
            t_shift=t_shift,
            max_duration=max_duration,
            remove_long_sil=remove_long_sil,
        )
    except HTTPException:
        raise
    except Exception as ex:
        LOGGER.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=str(ex)) from ex

    headers = {
        "X-ZipVoice-Elapsed-S": f"{metrics['elapsed_s']:.4f}",
        "X-ZipVoice-RTF": f"{metrics['rtf']:.4f}",
        "X-ZipVoice-Wav-Seconds": f"{metrics['wav_seconds']:.4f}",
    }
    return Response(content=audio_bytes, media_type="audio/wav", headers=headers)


@APP.get("/v1/config")
def show_config():
    return JSONResponse(
        {
            "model_name": ENGINE.model_name,
            "model_dir": ENGINE.model_dir,
            "tokenizer": ENGINE.tokenizer_type,
            "lang": ENGINE.lang,
            "device": str(ENGINE.device),
            "sampling_rate": ENGINE.sampling_rate,
            "default_num_step": ENGINE.default_num_step,
            "default_guidance_scale": ENGINE.default_guidance_scale,
            "default_speed": ENGINE.default_speed,
            "default_t_shift": ENGINE.default_t_shift,
            "default_max_duration": ENGINE.default_max_duration,
            "default_remove_long_sil": ENGINE.default_remove_long_sil,
            "max_concurrency": ENGINE.max_concurrency,
            "num_threads": ENGINE.num_threads,
            "trt_engine_path": ENGINE.trt_engine_path,
        }
    )