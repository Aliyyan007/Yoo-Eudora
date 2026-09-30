FROM python:3.12.7-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_WAIT_POLICY=PASSIVE

WORKDIR /app

# libopus0  — Opus codec required by opuslib-next (voice receive/send)
# libgomp1  — OpenMP runtime needed by onnxruntime / sherpa-onnx
# git       — pip git+ installs (discord.py-self)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-render.txt .
# discord-native-voice installed with --no-deps: it declares
# discord.py-self>=2.2.0 but the pinned git build reports 2.2.0a0
RUN pip install -r requirements-render.txt \
    && pip install --no-deps "discord-native-voice>=0.1.1"

COPY . .

CMD ["python", "-m", "src.main"]
