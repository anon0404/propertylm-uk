FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./

# Install dependencies before copying source code so this layer is cached.
RUN mkdir -p propertylm \
    && touch propertylm/__init__.py \
    && python -m pip install --upgrade pip \
    && python -m pip install .

COPY propertylm ./propertylm

EXPOSE 8000

CMD ["uvicorn", "propertylm.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
