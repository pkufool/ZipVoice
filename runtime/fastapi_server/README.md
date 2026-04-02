# ZipVoice FastAPI HTTP 服务

这个目录提供了一个基于 FastAPI 的零样本语音克隆服务，复用了 `zipvoice/bin/infer_zipvoice.py` 的核心推理流程。

## 1. HTTP 接口

- 健康检查: `GET /healthz`
- 查看运行配置: `GET /v1/config`
- 语音克隆: `POST /v1/clone`

`/v1/clone` 使用 `multipart/form-data`：

- `prompt_wav`: 参考音频文件（wav）
- `prompt_text`: 参考音频对应文本
- `text`: 要合成的目标文本
- 可选参数（均为 form 字段）:
  - `num_step`
  - `guidance_scale`
  - `speed`
  - `t_shift`
  - `max_duration`
  - `remove_long_sil`

返回值为 `audio/wav` 二进制音频，响应头包含：

- `X-ZipVoice-Elapsed-S`
- `X-ZipVoice-RTF`
- `X-ZipVoice-Wav-Seconds`

## 2. 并发与资源优化

服务内置如下策略：

- 模型与 vocoder 进程内单次加载，避免重复初始化。
- `torch.inference_mode()` 推理，降低显存与开销。
- `ZIPVOICE_MAX_CONCURRENCY` 控制并发推理数，避免内存占用过高。
- `ZIPVOICE_NUM_THREADS` 控制 CPU 线程使用。
- 长文本自动按标点切分并分批处理，减少峰值显存。

## 3. Docker 打包（模型打进镜像）

### 3.1 准备模型目录

先将模型放在仓库内一个明确目录，例如：

- `models/zipvoice/model.pt`
- `models/zipvoice/model.json`
- `models/zipvoice/tokens.txt`

或：

- `models/zipvoice_distill/model.pt`
- `models/zipvoice_distill/model.json`
- `models/zipvoice_distill/tokens.txt`

### 3.2 构建镜像

在仓库根目录执行：

```bash
docker build \
  -f runtime/fastapi_server/Dockerfile \
  --build-arg MODEL_DIR=models/zipvoice_distill \
  --build-arg MODEL_NAME=zipvoice_distill \
  -t zipvoice-fastapi:latest \
  .
```

如果你使用基础模型，把参数改为：

```bash
--build-arg MODEL_DIR=models/zipvoice
--build-arg MODEL_NAME=zipvoice
```

### 3.3 启动容器（CPU）

```bash
docker run --rm -it \
  -p 8000:8000 \
  -e ZIPVOICE_DEVICE=cpu \
  -e ZIPVOICE_MAX_CONCURRENCY=2 \
  -e ZIPVOICE_NUM_THREADS=4 \
  zipvoice-fastapi:latest
```

## 4. 调用示例

```bash
curl -X POST "http://127.0.0.1:8000/v1/clone" \
  -F "prompt_wav=@prompt.wav" \
  -F "prompt_text=I am a prompt." \
  -F "text=I am the target sentence." \
  -F "num_step=8" \
  -F "guidance_scale=3.0" \
  -F "speed=1.0" \
  -F "t_shift=0.5" \
  -F "max_duration=100" \
  -F "remove_long_sil=false" \
  --output result.wav
```

## 5. 关键环境变量

- `ZIPVOICE_MODEL_NAME`: `zipvoice` 或 `zipvoice_distill`
- `ZIPVOICE_MODEL_DIR`: 模型目录（Docker 中默认 `/opt/zipvoice_model`）
- `ZIPVOICE_DEVICE`: `cpu`（CPU 镜像建议固定为 `cpu`）
- `ZIPVOICE_MAX_CONCURRENCY`: 并发推理上限
- `ZIPVOICE_NUM_THREADS`: CPU 线程数
- `ZIPVOICE_TRT_ENGINE_PATH`: 可选 TensorRT engine 路径
- `ZIPVOICE_VOCODER_PATH`: 可选本地 vocoder 路径
- `ZIPVOICE_DEFAULT_NUM_STEP`
- `ZIPVOICE_DEFAULT_GUIDANCE_SCALE`
- `ZIPVOICE_DEFAULT_SPEED`
- `ZIPVOICE_DEFAULT_T_SHIFT`
- `ZIPVOICE_DEFAULT_MAX_DURATION`
- `ZIPVOICE_DEFAULT_REMOVE_LONG_SIL`