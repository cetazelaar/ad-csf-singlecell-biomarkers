FROM rocker/rstudio:4.3.1

RUN apt-get update && apt-get install -y \
    libpython3-dev \
    python3-pip \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages(c('Seurat', 'tidyverse', 'patchwork'))"
RUN R -e "if (!requireNamespace('BiocManager', quietly = TRUE)) install.packages('BiocManager')"
RUN R -e "BiocManager::install(c('MAST', 'fgsea'))"

RUN pip3 install pandas numpy scikit-learn matplotlib seaborn scanpy gseapy

WORKDIR /home/rstudio/project
