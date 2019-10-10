#!/bin/bash
screen -d -m -S jupyter bash -c "cd ~ && /home/ubuntu/anaconda3/envs/infutor/bin/python -m jupyter notebook --no-browser --allow-root --NotebookApp.token=''"
