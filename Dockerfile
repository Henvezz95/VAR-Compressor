# Base image for JetPack 6 (L4T r36) with PyTorch and CUDA enabled
FROM dustynv/pytorch:2.6-r36.4.0-cu128

# Define generic build arguments with safe fallbacks (1000 is the standard Linux default)
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=varuser

# Create a non-root user matching the host's UID/GID to prevent permission issues
RUN groupadd -g ${GROUP_ID} ${USERNAME} || true && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -ms /bin/bash ${USERNAME} || true && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install essential development tools
RUN apt-get update && apt-get install -y \
    git \
    nano \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configure Hugging Face cache directories to reside in the mounted workspace
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV HF_HOME=/workspace/.cache/huggingface

# Set the working directory
WORKDIR /workspace

# Switch to the synchronized non-root user
USER ${USERNAME}