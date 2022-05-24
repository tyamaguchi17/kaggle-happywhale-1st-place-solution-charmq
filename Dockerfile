FROM nvcr.io/nvidia/pytorch:22.02-py3

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    tmux &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt
