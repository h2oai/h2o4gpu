#!/bin/bash
#
# Note:  This run script is meant to be run inside the docker container.
#

set -e

if [ "x$1" != "x" ]; then
    d=$1
    cd $d
    shift
    exec "$@"
fi

logdir=/log/`date "+%Y%m%d-%H%M%S"`
mkdir -p "$logdir"

export HOME=/jupyter && \
cd /jupyter && \
jupyter --paths >> "$logdir"/jupyter.log && \
exec jupyter notebook --ip='*' --no-browser --allow-root >> "$logdir"/jupyter.log 2>&1
