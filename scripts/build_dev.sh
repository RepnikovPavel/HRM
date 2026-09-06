#!/bin/bash

check_nvidia_container_toolkit() {
    if command -v nvidia-ctk &> /dev/null; then
        echo "✅ nvidia-container-toolkit установлен."
    else
        echo "❌ Установите nvidia-container-toolkit"
        exit 1
    fi
}

check_nvidia_container_toolkit


DOCKER_BUILDKIT=1 docker buildx build \
    --pull=false \
    -t hrm130dev:latest \
    -f docker/DockerFileDevCuda13 \
    --progress=plain \
    . \
&& echo "✅ Сборка завершена!" \
&& docker image ls hrm130dev:latest
