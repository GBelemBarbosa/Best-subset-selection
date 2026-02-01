#!/bin/bash

# final_launch.sh

echo "Syncing code..."
scp cv_comp.jl cdss_vs_l0learn.jl new_test_terminal.jl queue_tseng_*.sh tseng:~/
scp cv_comp.jl cdss_vs_l0learn.jl new_test_terminal.jl queue_borwein_*.sh borwein:~/

echo "Cleaning and launching tseng..."
ssh tseng "mkdir -p logs; pkill -9 julia; tmux kill-server; sleep 5; 
tmux new-session -d -s w1 'bash queue_tseng_1.sh'; sleep 5;
tmux new-session -d -s w2 'bash queue_tseng_2.sh'; sleep 5;
tmux new-session -d -s w4 'bash queue_tseng_4.sh'; sleep 5"

echo "Cleaning and launching borwein..."
ssh borwein "mkdir -p logs; pkill -9 julia; tmux kill-server; sleep 5; 
tmux new-session -d -s w1 'bash queue_borwein_1.sh'; sleep 5;
tmux new-session -d -s w2 'bash queue_borwein_2.sh'; sleep 5;
tmux new-session -d -s w4 'bash queue_borwein_4.sh'; sleep 5"

echo "Waiting for stabilization..."
sleep 30

echo "TSENG STATUS:"
ssh tseng "hostname; tmux ls; pgrep julia | xargs -r ps -fp"
echo "---"
echo "BORWEIN STATUS:"
ssh borwein "hostname; tmux ls; pgrep julia | xargs -r ps -fp"
