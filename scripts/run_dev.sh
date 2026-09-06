#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Ошибка: необходимо указать путь для монтирования как аргумент скрипта"
    echo "Пример использования: $0 /путь/к/директории"
    exit 1
fi

MOUNT_PATH=$1
 
IMG_NAME=hrm130dev:latest
CONTAINER_NAME=hrm130devcontainer

xhost +local:docker

docker run --rm -it \
  --mount type=bind,src="$MOUNT_PATH",target=/work \
  --mount type=bind,src=/tmp/.X11-unix,target=/tmp/.X11-unix,consistency=cached \
  --mount type=bind,src=/mnt,target=/mnt \
  --net host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,display \
  -e DISPLAY="$(echo $DISPLAY)" \
  --runtime=nvidia \
  --gpus all \
  --name $CONTAINER_NAME \
  --user "$(id -u):$(id -g)" \
  -p 64023:64023 \
  --detach \
  $IMG_NAME