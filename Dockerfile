FROM python:3.13-slim AS builder
WORKDIR /build
RUN pip install --no-cache-dir build
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install . \
    && cp -r src/photo_filter/static /install/lib/python3.13/site-packages/photo_filter/static

FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libwebp7 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY config.example.yaml /app/config.example.yaml

EXPOSE 8000

ENTRYPOINT ["photo-filter"]
CMD ["scan"]
