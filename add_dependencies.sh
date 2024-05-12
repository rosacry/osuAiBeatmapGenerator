#!/bin/bash

# Script to install packages and automatically update requirements.txt

# Activate your virtual environment
activate env/

# Install package(s) passed as arguments to the script
pip install "$@"

# Update requirements.txt file
pip freeze > requirements.txt
