#!/bin/bash
screen -d -m -S jupyter bash -c "source activate infutor && cd /home/ubuntu && /home/ubuntu/anaconda3/bin/jupyter notebook --no-browser --NotebookApp.token=''"

