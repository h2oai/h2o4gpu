#!/bin/bash
nvidia-docker build  -t opsh2oai/h2o4gpu-${extratag}-build -f Dockerfile-build --rm=false --build-arg cuda=${dockerimage} .
nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-${extratag}-build
nvidia-docker exec ${CONTAINER_NAME} rm -rf data
nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
nvidia-docker exec ${CONTAINER_NAME} rm -rf py3nvml
nvidia-docker exec ${CONTAINER_NAME} bash -c 'export HOME=`pwd`; eval \"\$(/root/.pyenv/bin/pyenv init -)\"  ; /root/.pyenv/bin/pyenv global 3.6.1; pip install `find src/interface_py/${dist} -name "*h2o4gpu-${versionTag}*.whl"`; make ${target}'
nvidia-docker exec ${CONTAINER_NAME} touch src/interface_py/h2o4gpu/__init__.py
nvidia-docker exec ${CONTAINER_NAME} bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\"  ;  /root/.pyenv/bin/pyenv global 3.6.1; make pylint'
nvidia-docker stop ${CONTAINER_NAME}
