# Python 3.10 + CUDA 12 (same as official Colab notebook)
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Disable XLA autotuner for older GPU compatibility (Titan RTX SM 7.5)
ENV XLA_FLAGS="--xla_gpu_autotune_level=0"
# JAX: use dynamic GPU memory allocation instead of preallocating
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
# TensorFlow: also use dynamic allocation
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

RUN apt-get update && apt-get install -y \
    wget git curl \
    libfluidsynth3 build-essential libasound2-dev libjack-dev \
    ffmpeg ca-certificates apt-transport-https gnupg && \
    mkdir -p /usr/share/keyrings && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    wget -qO- https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda create -n mt3 python=3.11 -y
SHELL ["conda", "run", "-n", "mt3", "/bin/bash", "-c"]

WORKDIR /app

# Install MT3 (same as official notebook)
RUN git clone --branch=main https://github.com/magenta/mt3 && \
    mv mt3 mt3_tmp && mv mt3_tmp/* . && rm -r mt3_tmp && \
    pip install jax[cuda12] nest-asyncio pyfluidsynth==1.3.0 -e . \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Download checkpoints
RUN gsutil -q -m cp -r gs://mt3/checkpoints .

# Install Flask and Gunicorn
RUN pip install flask gunicorn

RUN useradd -ms /bin/bash mt3user
RUN chown -R mt3user:mt3user /app /opt/conda
USER mt3user
WORKDIR /home/mt3user

COPY --chown=mt3user:mt3user ismir2021.gin mt3.gin model.gin app.py ./

# Copy checkpoints to mt3user directory
RUN cp -r /app/checkpoints ./

EXPOSE 5000

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mt3", "gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "600", "app:app"]
