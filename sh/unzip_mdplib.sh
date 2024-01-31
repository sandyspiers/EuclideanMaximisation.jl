#!/bin/bash

# Inflates MDPLIB20.zip into data/instances/

echo "Looking for mdplib2.0.zip in bin/..."

mdplib2="bin/mdplib_2.0.zip"
instance_folder="data/instance/"
keyword="GKD"

# Check if the source file exists
if [ ! -e "$mdplib2" ]; then
    echo "$mdplib2 does not exists!"
    echo "Please download from https://www.uv.es/rmarti/paper/mdp.html and put into bin/ folder"
    exit -1
fi
echo "Found $mdplib2"

echo "Unzipping to $instance_folder, keeping any file that contains"
echo "the keyword 'GKD' or is a documentation of instance format"
mkdir -p "$instance_folder"
unzip -o "$mdplib2" "*/$keyword*" "instance*" -d "$instance_folder"

echo "Complete!"
