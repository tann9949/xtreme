GPU_DEVICES=6

docker run -it \
    --gpus="device=${GPU_DEVICES}" \
    --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/workspace \
    --net=host \
    --privileged \
    --name xtreme \
    chompk/xtreme:latest \
    bash
