FROM python:3.10.9-slim-buster

WORKDIR /work_dir

# # nvidia-container-runtime
# ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
# ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# install apt
RUN apt update && \
    apt install -y --no-install-recommends mesa-utils x11-apps && \
    apt install -y --no-install-recommends build-essential cmake libblkid-dev e2fslibs-dev libboost-all-dev libaudit-dev clang && \
    apt install -y --no-install-recommends git libopencv-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /fractal
COPY fractal-pretraining/ /fractal/
RUN cd /fractal/ && pip install -e .

# install pip
ADD requirements.txt /work_dir
RUN pip install -U pip && \
    pip install -r requirements.txt && \
    rm requirements.txt

# ユーザー指定
ARG UID
ARG GID
ARG USERNAME
ARG GROUPNAME

RUN groupadd -g $GID $GROUPNAME
RUN useradd -m -u $UID -g $GID $USERNAME

RUN chown $USERNAME:$GROUPNAME -R /work_dir
USER $USERNAME

# ターミナルの色指定
RUN echo "PS1='\[\e[41;37m\]\u@\h\[\e[0m\]:\[\e[34m\]\w\[\e[0m\]$ '" >> /home/$USERNAME/.bashrc

ENTRYPOINT ["/bin/bash"]