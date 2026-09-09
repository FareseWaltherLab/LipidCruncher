# Single image for the whole application: the lipidomics pipeline, the AI
# Studio, and the stable-isotope-tracing module (which needs an R runtime with
# IsoCorrectoR).
#
# One Dockerfile, many deployments. Which app you get is decided by the branch
# you build from, not by picking a different file here; separate deployments
# are separate ECR tags + ECS services pointing at them.
#
# The R layer roughly triples the image size and dominates a cold build. It is
# placed before the application COPY so it stays cached across app changes —
# only the first build (and R dependency bumps) pays the cost.
#
# Reproducibility note: dev was validated on R 4.6 / IsoCorrectoR 1.30.0.
# Debian's r-base may install a slightly older IsoCorrectoR; outputs reproduce
# within tolerance (1.28 <-> 1.30 verified). If an exact version is required,
# add the CRAN apt repo to install a newer R before installing IsoCorrectoR.
#
# Build for Fargate (x86_64) explicitly — an arm64 image built on Apple Silicon
# fails to start with "exec format error":
#   docker build --platform linux/amd64 -t <ecr>/<repo>:<tag> .

FROM python:3.11-slim
WORKDIR /app

# System deps:
#   poppler-utils                      -> pdf2image (PDF report generation)
#   r-base r-base-dev gfortran ...     -> R + toolchain to compile R packages
#   lib*-dev                           -> headers some Bioconductor deps need
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        r-base r-base-dev \
        gfortran build-essential \
        libcurl4-openssl-dev libssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# R packages: IsoCorrectoR (Bioconductor) + jsonlite (our wrapper reads a JSON
# config). Verified in the same layer so a broken install fails the build
# rather than surfacing at runtime as a missing-package error.
RUN Rscript -e 'install.packages(c("BiocManager", "jsonlite"), repos="https://cloud.r-project.org"); \
                BiocManager::install("IsoCorrectoR", ask=FALSE, update=FALSE)' \
    && Rscript -e 'stopifnot(requireNamespace("IsoCorrectoR", quietly=TRUE), requireNamespace("jsonlite", quietly=TRUE)); \
                   cat("IsoCorrectoR", as.character(packageVersion("IsoCorrectoR")), "OK\n")'

# Python deps.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application.
COPY . .

# Keep the upload cap on the image rather than relying on the working
# directory finding .streamlit/config.toml, or on a task-definition
# override that lives outside this repo.
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024

EXPOSE 8501

# The AI Studio reads ANTHROPIC_API_KEY from the environment at call time. It is
# deliberately NOT baked in — inject it at run time (ECS `secrets` from Secrets
# Manager / SSM, or `docker run -e`). Without it the rest of the app runs and
# only the AI Studio reports the key as missing.
ENTRYPOINT ["streamlit", "run"]
CMD ["src/main_app.py"]
