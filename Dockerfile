FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # Install python 3.8, and other tools if needed.
    # Note: Since we are in a Ubunut 20.04 image, this installs python-3.8
    python3 python3-pip python3-venv wget git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Turn off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Configure Poetry
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install Poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==1.3.0 
# Add `poetry` to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /workspace

# Install Python dependencies using Poetry
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

COPY music-ner/ music-ner/

CMD poetry run python3 music-ner/__init__.py