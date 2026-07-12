FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Dependencies for the sibling MCP servers (short-term-trader / stock-recommender).
# The container launches those servers with its own Python, so their deps must be
# present here. Pre-download the NLTK VADER lexicon so sentiment works offline.
COPY requirements-siblings.txt ./
RUN pip install -r requirements-siblings.txt \
    && python -m nltk.downloader -d /usr/share/nltk_data vader_lexicon
ENV NLTK_DATA=/usr/share/nltk_data \
    MCP_PYTHON=python3

COPY . .

RUN mkdir -p data

EXPOSE 8501

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "run.sh"]
