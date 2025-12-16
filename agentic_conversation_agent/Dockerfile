ARG BUILD_FROM
FROM ${BUILD_FROM}

# Base images already include s6-overlay and bashio.

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apk add --no-cache \
      python3 \
      py3-pip \
      ca-certificates \
      curl

WORKDIR /app

COPY rootfs/ /

RUN pip3 install --no-cache-dir -r /app/requirements.txt

# s6 overlay will start /etc/services.d/*/run
