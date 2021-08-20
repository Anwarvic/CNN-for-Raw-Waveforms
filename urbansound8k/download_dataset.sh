#!/bin/bash

echo "Downloading Urban Sound 8K (6GB)"
kaggle datasets download -d chrisfilo/urbansound8k

echo "Unzipping.."
unzip urbansound8k.zip

echo "Done!"
