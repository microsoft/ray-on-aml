FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20210615.v1
ARG HTTP_PROXY
ARG HTTPS_PROXY
# set http_proxy & https_proxy
ENV http_proxy=${HTTPS_PROXY}
ENV https_proxy=${HTTPS_PROXY}
RUN http_proxy=${HTTPS_PROXY} https_proxy=${HTTPS_PROXY} apt-get update -y \
    && mkdir -p /usr/share/man/man1 \
    && http_proxy=${HTTPS_PROXY} https_proxy=${HTTPS_PROXY} apt-get install -y openjdk-11-jdk \
    && mkdir /raydp \
    && pip --no-cache-dir install raydp git+https://github.com/microsoft/ray-on-aml.git@james-dev azureml-defaults==1.41.0
WORKDIR /raydp
# unset http_proxy & https_proxy
ENV http_proxy=
ENV https_proxy=