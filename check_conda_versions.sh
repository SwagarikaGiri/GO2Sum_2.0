#!/bin/bash

envs=$(conda env list | awk '{print $1}' | grep -v "#" | grep -v "^$")

for env in $envs; do
    echo "Environment: $env"
    conda run -n $env conda --version
    echo "-----------------------"
done
