# syntax=docker/dockerfile:1.3-labs
# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.191.1/containers/python-3/.devcontainer/base.Dockerfile

FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

RUN apt update

RUN apt install -y wget curl git sudo nano

RUN apt install -y \
        build-essential \
        ca-certificates \
        libbz2-dev \
        zlib1g-dev \
        libffi-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        liblzma-dev \
        libssl-dev

RUN useradd -ms /bin/bash vscode
RUN echo 'vscode ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER vscode

RUN curl https://pyenv.run | bash
RUN cat <<EOF >> /home/vscode/.bashrc
# pyenv
command -v pyenv >/dev/null || export PATH="\$PATH:/home/vscode/.pyenv/bin"
eval "\$(pyenv init -)"
EOF

ENV PATH="$PATH:/home/vscode/.pyenv/bin"
RUN pyenv install 3.9.13
RUN pyenv global 3.9.13

WORKDIR /workspace
