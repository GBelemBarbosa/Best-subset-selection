import os

# We will move the pending cv_comp tasks to borwein
# The running ones on tseng are:
# - const 0.5 2000 (worker 1)
# - exp 0.5 2000 (worker 2)

# The pending ones are:
# - const 0.9 1000 (worker 1)
# - exp 0.9 1000 (worker 2)

borwein_queues = [
    [f"julia cv_comp.jl const 0.9 1000 5.0 20 100:100:1000 L0Hybrid 100 false >> logs/L0Hybrid_const_0.9_1000_5.0_20.log 2>&1"],
    [f"julia cv_comp.jl exp 0.9 1000 5.0 20 100:100:1000 L0Hybrid 100 false >> logs/L0Hybrid_exp_0.9_1000_5.0_20.log 2>&1"]
]

def write_queue(filename, commands):
    with open(filename, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("mkdir -p logs\n")
        for cmd in commands:
            f.write(f"echo \"Starting: {cmd}\"\n")
            f.write(f"{cmd}\n")
            f.write(f"echo \"Finished (exit code $?): {cmd}\"\n")
        f.write("echo \"All tasks completed.\"\n")
        f.write("sleep 86400\n")

# We only need worker1 and worker2 on borwein for these
write_queue("queue_borwein_worker1.sh", borwein_queues[0])
write_queue("queue_borwein_worker2.sh", borwein_queues[1])
