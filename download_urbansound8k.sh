#!/bin/bash

echo "Downloading Urban Sound 8K (6GB)
mkdir -p "urbansound8k"
cd urbansound8k

kaggle datasets download -d chrisfilo/urbansound8k

echo "Unzipping.."
unzip urbansound8k.zip

echo "Done!"


