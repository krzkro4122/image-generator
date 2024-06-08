#!/bin/bash

if [ -z $1 ]; then
    echo "Need an argument damn it! (App name)"
    exit 1
fi

app_name=$1

.venv/bin/python web/manage.py makemigrations $app_name
