#!/bin/bash
nvidia-docker build -t opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime:latest -f Dockerfile-runtime --rm=false --build-arg cuda=${dockerimage} --build-arg wheel=${encodedFullVersionTag}${extratag}/h2o4gpu-${encodedFullVersionTag}-py36-none-any.whl --build-arg buckettype=${buckettype} .
nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime
nvidia-docker exec ${CONTAINER_NAME} rm -rf data
nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
nvidia-docker exec ${CONTAINER_NAME} bash -c '. /h2o4gpu_env/bin/activate ; pip freeze'
echo "Building on linux - Getting Data"
nvidia-docker exec ${CONTAINER_NAME} bash -c '. /h2o4gpu_env/bin/activate ; mkdir -p scripts ; rm -rf scripts/fcov_get.py ; echo "from sklearn.datasets import fetch_covtype" > ./scripts/fcov_get.py ; echo "cov = fetch_covtype()" >> ./scripts/fcov_get.py'
nvidia-docker exec ${CONTAINER_NAME} bash -c '. /h2o4gpu_env/bin/activate ; python ./scripts/fcov_get.py'
nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /open_data/creditcard.csv .'
nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; wget https://s3.amazonaws.com/h2o-public-test-data/h2o4gpu/open_data/kmeans_data/h2o-logo.jpg'
nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /data/ipums_1k.csv .'
nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /data/ipums.feather .'
nvidia-docker commit ${CONTAINER_NAME} opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime:latest
echo "Building on linux - stopped docker"
nvidia-docker stop ${CONTAINER_NAME}
nvidia-docker save opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime | gzip > h2o4gpu-${fullVersionTag}${extratag}-runtime.tar.gz
echo "Building on linux - saving docker"
