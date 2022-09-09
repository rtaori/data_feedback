#!/bin/bash

./download_cifar5m.sh
./download_cinic10.sh
python preprocess_cinic10.py
./download_realtoxicityprompts.sh
./download_imsitu.sh