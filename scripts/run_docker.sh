GPU_DEVICES=0

docker run -it \
    --gpus="device=${GPU_DEVICES}" \
    --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd)/download:/workspace/download \
    -v $(pwd)/output:/workspace/output \
    --net=host \
    --privileged \
    --name xtreme \
    chompk/xtreme:latest \
    bash
