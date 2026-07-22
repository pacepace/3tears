# Debian (not Ubuntu) base: Debian's `chromium` apt package is a real binary;
# Ubuntu's is a snap-wrapper stub that fails with no snapd in a container.
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium \
        xvfb \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir \
        "nodriver>=0.50,<0.51" \
        "fastapi>=0.115,<0.116" \
        "uvicorn[standard]>=0.32,<0.33"

COPY main.py /app/main.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV DISPLAY_NUM=99 \
    SCREEN_GEOM=1920x1080x24 \
    CHROMIUM_PATH=/usr/bin/chromium \
    PORT=8088

EXPOSE 8088

ENTRYPOINT ["/app/entrypoint.sh"]
