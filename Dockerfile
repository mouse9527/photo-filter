FROM python:3.13-slim AS builder
WORKDIR /build
RUN pip install --no-cache-dir build
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libwebp7 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY config.example.yaml /app/config.example.yaml

ENTRYPOINT ["photo-filter"]
CMD ["scan"]
