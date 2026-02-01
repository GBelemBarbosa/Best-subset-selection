import os

# Setups for Phase 2 (new_test_terminal.jl)
# We want to run 3 total. We will use the standard setups but with tridiagonal=true.
# Setup list:
# S1: exp 0.9 1000 5.0 20
# S2: const 0.5 2000 10.0 100
# S3: exp 0.5 2000 10.0 100

# Allocation:
# Tseng has const 0.5 (cv) and exp 0.5 (cv) -> 2 processes. Can take 1 more.
# Borwein has const 0.9 (cv) and exp 0.9 (cv) -> 2 processes. Can take 1 more.
# Total 3 is requested. We can run 1 on Tseng, 1 on Borwein, and 1 more on whichever machine has better capacity?
# Actually S1 is ρ=0.9, S2/S3 are ρ=0.5.
# Let's put S1 on Borwein (worker 3).
# Let's put S2 on Tseng (worker 3).
# Let's put S3 on Borwein (worker 4/alternate session). Wait, let's keep it tidy.
# Total 3: Tseng (1), Borwein (2). Borwein will have 4 processes total but only 3 are julia? 
# Wait, worker 1,2 are running cv_comp. 
# worker 3 will run new_test_terminal.
# I'll put S1 and S3 on Borwein (sequentially in worker 3).
# I'll put S2 on Tseng (worker 3).

tseng_queue3 = [
    "julia new_test_terminal.jl const 0.5 2000 10.0 100 200:200:2000 10 true >> logs/NewTerminal_const_0.5_2000_10.0_100_tridiagonal.log 2>&1"
]

borwein_queue3 = [
    "julia new_test_terminal.jl exp 0.9 1000 5.0 20 100:100:1000 10 true >> logs/NewTerminal_exp_0.9_1000_5.0_20_tridiagonal.log 2>&1",
    "julia new_test_terminal.jl exp 0.5 2000 10.0 100 200:200:2000 10 true >> logs/NewTerminal_exp_0.5_2000_10.0_100_tridiagonal.log 2>&1"
]

def write_queue(filename, commands):
    with open(filename, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("mkdir -p logs\n")
        for cmd in commands:
            f.write(f"echo \"Starting Phase 2: {cmd}\"\n")
            f.write(f"{cmd}\n")
            f.write(f"echo \"Finished (exit code $?): {cmd}\"\n")
        f.write("echo \"Phase 2 Queue completed.\"\n")
        f.write("sleep 86400\n")

write_queue("queue_tseng_phase2.sh", tseng_queue3)
write_queue("queue_borwein_phase2.sh", borwein_queue3)
