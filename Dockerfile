# docker build -t snapp_na . && docker run --rm -it -v %CD%:/work snapp_na
FROM mambaorg/micromamba:1.5.8

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git emacs htop && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

WORKDIR /work

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba create -y -n geo -f /tmp/environment.yml && micromamba clean -a -y

ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV MAMBA_DEFAULT_ENV=geo
ENV PATH=/opt/conda/envs/geo/bin:$PATH
ENV PROJ_LIB=/opt/conda/envs/geo/share/proj
ENV GDAL_DATA=/opt/conda/envs/geo/share/gdal
SHELL ["/usr/local/bin/_entrypoint.sh", "/bin/bash", "-lc"]

RUN micromamba run -n geo git clone --depth 1 https://github.com/springinnovate/ecoshard /tmp/ecoshard && \
    micromamba run -n geo python -m pip install -U pip setuptools wheel && \
    micromamba run -n geo pip install --no-build-isolation /tmp/ecoshard

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["/bin/bash"]