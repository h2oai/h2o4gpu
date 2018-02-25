#!/bin/bash

ARGS=()
for var in "$@"; do
	[ "$var" != '-fno-plt' ] && [ "$var" != '-mtune=haswell' ] && ARGS+=("$var")
done
/usr/bin/gcc "${ARGS[@]}"

