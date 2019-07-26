FROM tensorflow/tensorflow:latest-gpu-py3

########################################  BASE SYSTEM
# set noninteractive installation# Apt add py3.6
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install -y apt-utils
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y\
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    python3.6-dev \
    pkg-config \
    python3.6 \
    tzdata \
    cmake \
    gnupg \
    curl

######################################## PYTHON3
RUN apt-get install -y \
    python3 \
    python3-pip

# set local timezone
RUN ln -fs /usr/share/zoneinfo/Europe/Copenhagen /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# transfer-learning-conv-ai
ENV PYTHONPATH /usr/local/lib/python3.6 
COPY . ./
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# model zoo
RUN mkdir models && \
    curl https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz > models/finetuned_chatbot_gpt.tar.gz && \
    cd models/ && \
    tar -xvzf finetuned_chatbot_gpt.tar.gz && \
    rm finetuned_chatbot_gpt.tar.gz

# initalize on build
RUN python ./initscript.py
    
CMD gunicorn --bind 0.0.0.0:80 server:app
