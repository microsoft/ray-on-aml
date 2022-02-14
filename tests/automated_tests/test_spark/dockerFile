FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20210615.v1 
ARG HTTP_PROXY
ARG HTTPS_PROXY
# set http_proxy & https_proxy
ENV http_proxy=${{HTTPS_PROXY}}
ENV https_proxy=${{HTTPS_PROXY}}
RUN http_proxy=${{HTTPS_PROXY}} https_proxy=${{HTTPS_PROXY}} apt-get update -y \
    && mkdir -p /usr/share/man/man1 \
    && http_proxy=${{HTTPS_PROXY}} https_proxy=${{HTTPS_PROXY}} apt-get install -y openjdk-11-jdk \
    && mkdir /raydp \
    && pip --no-cache-dir install raydp torchvision==0.8.1 ray[tune] ray[serve] ray[rllib] pyarrow>= 5.0.0 dask[complete]==2021.12.0 adlfs==2021.10.0 fsspec==2021.10.1 xgboost_ray==0.1.6 fastparquet==0.7.2
WORKDIR /raydp
# unset http_proxy & https_proxy
ENV http_proxy=
ENV https_proxy=