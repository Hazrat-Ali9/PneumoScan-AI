#!/bin/bash

# Create main folders
mkdir -p project
mkdir -p project/cloud
mkdir -p project/components
mkdir -p project/constants
mkdir -p project/configeration
mkdir -p project/pipeline
mkdir -p project/entity
mkdir -p project/logger
mkdir -p project/exception
mkdir -p project/utils

mkdir -p config_yaml
touch config_yaml/config.yaml
touch config_yaml/param.yaml
touch config_yaml/secrets.yaml


# # Create __init__.py in each folder
touch project/__init__.py
touch project/cloud/__init__.py
touch project/components/__init__.py
touch project/constants/__init__.py
touch project/configeration/__init__.py
touch project/pipeline/__init__.py
touch project/entity/__init__.py
touch project/logger/__init__.py
touch project/exception/__init__.py
touch project/utils/__init__.py


# Create main files
touch app.py
touch requirements.txt
touch notex.txt 
touch setup.py 
touch templates.sh




## bash templates.sh


