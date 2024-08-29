FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
ENV PATH="/root/miniconda3/bin:$PATH"
ARG PATH="/root/miniconda3/bin:$PATH"
RUN apt-get update && apt-get install -y libopenblas-dev
RUN apt-get update && apt-get install -y curl wget gcc build-essential
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
    
# install conda
# Install Miniconda on x86 or ARM platforms
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh
WORKDIR /app
COPY ./ /app
# create env with python 3.8
RUN conda init bash \
    && . ~/.bashrc \
    && conda create --name openmask3d python=3.8.5 \
    && conda activate openmask3d \
    && bash install_requirements.sh \
    && pip install -e .

#ENV PATH=/opt/conda/envs/openmask3d/bin:$PATH

CMD ["/bin/bash"]
