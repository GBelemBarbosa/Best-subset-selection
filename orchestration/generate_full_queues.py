import os

# Setups for standard comparison (Phase 1 style)
setups = [
    ("const", 0.5, 2000, 10.0, 100),
    ("const", 0.9, 1000, 5.0, 20),
    ("exp", 0.5, 2000, 10.0, 100),
    ("exp", 0.9, 1000, 5.0, 20)
]

tseng_queues = [[], [], [], []]
borwein_queues = [[], [], [], []]

# Distrib cv_comp.jl (Phase 1)
# Tseng 1: const 0.5 2000
# Borwein 1: exp 0.5 2000
# Tseng 2: const 0.9 1000
# Borwein 2: exp 0.9 1000

# Setup indices: 0:const 0.5, 1:const 0.9, 2:exp 0.5, 3:exp 0.9
tseng_queues[0].append(f"julia cv_comp.jl const 0.5 2000 10.0 100 200:200:2000 L0Hybrid 100 false >> logs/L0Hybrid_const_0.5_2000_10.0_100.log 2>&1")
borwein_queues[0].append(f"julia cv_comp.jl exp 0.5 2000 10.0 100 200:200:2000 L0Hybrid 100 false >> logs/L0Hybrid_exp_0.5_2000_10.0_100.log 2>&1")
tseng_queues[1].append(f"julia cv_comp.jl const 0.9 1000 5.0 20 100:100:1000 L0Hybrid 100 false >> logs/L0Hybrid_const_0.9_1000_5.0_20.log 2>&1")
borwein_queues[1].append(f"julia cv_comp.jl exp 0.9 1000 5.0 20 100:100:1000 L0Hybrid 100 false >> logs/L0Hybrid_exp_0.9_1000_5.0_20.log 2>&1")

# Distrib cdss_vs_l0learn.jl 
# Tseng 4: const 0.5 2000, const 0.9 1000
# Borwein 4: exp 0.5 2000, exp 0.9 1000
tseng_queues[3].append(f"julia cdss_vs_l0learn.jl const 0.5 2000 10.0 100 200:200:2000 10 true >> logs/cdss_vs_l0learn_const_0.5_2000_10.0_100.log 2>&1")
tseng_queues[3].append(f"julia cdss_vs_l0learn.jl const 0.9 1000 5.0 20 100:100:1000 10 true >> logs/cdss_vs_l0learn_const_0.9_1000_5.0_20.log 2>&1")
borwein_queues[3].append(f"julia cdss_vs_l0learn.jl exp 0.5 2000 10.0 100 200:200:2000 10 true >> logs/cdss_vs_l0learn_exp_0.5_2000_10.0_100.log 2>&1")
borwein_queues[3].append(f"julia cdss_vs_l0learn.jl exp 0.9 1000 5.0 20 100:100:1000 10 true >> logs/cdss_vs_l0learn_exp_0.9_1000_5.0_20.log 2>&1")

# Phase 2: New Test Terminal
# Tseng 3: NewTerminal const 0.5 2000 (S2)
# Borwein 3: NewTerminal exp 0.5 2000 (S3) + NewTerminal exp 0.9 1000 (S1)
tseng_queues[2].append(f"julia new_test_terminal.jl const 0.5 2000 10.0 100 200:200:2000 10 true >> logs/NewTerminal_const_0.5_2000_10.0_100_tridiagonal.log 2>&1")
borwein_queues[2].append(f"julia new_test_terminal.jl exp 0.9 1000 5.0 20 100:100:1000 10 true >> logs/NewTerminal_exp_0.9_1000_5.0_20_tridiagonal.log 2>&1")
borwein_queues[2].append(f"julia new_test_terminal.jl exp 0.5 2000 10.0 100 200:200:2000 10 true >> logs/NewTerminal_exp_0.5_2000_10.0_100_tridiagonal.log 2>&1")

def write_queue(filename, commands):
    with open(filename, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("mkdir -p logs\n")
        for cmd in commands:
            f.write(f"echo \"Starting: {cmd}\"\n")
            f.write(f"{cmd}\n")
            f.write(f"echo \"Finished (exit code $?): {cmd}\"\n")
        f.write("echo \"Queue completed.\"\n")
        f.write("sleep 86400\n")

for i in range(4):
    write_queue(f"queue_tseng_{i+1}.sh", tseng_queues[i])
    write_queue(f"queue_borwein_{i+1}.sh", borwein_queues[i])
