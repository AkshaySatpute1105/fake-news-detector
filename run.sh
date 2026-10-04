#!/bin/sh
[ -f model.pkl ] || python3 model.py
python3 app.py
