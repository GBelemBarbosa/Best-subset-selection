import glob
import re
import os

def parse_logs():
    log_files = glob.glob(os.path.join('/mnt/c/Users/gggt4/Documents/Pesquisa/Best selection/graphs/L0Hybrid/logs', 'exp*.log'))
    
    results = {} # {algo_name: {'sim': [], 'time': []}}

    # Regex patterns
    # Algo name: "Name" = "Name"
    re_algo = re.compile(r'^"(.+)" = "(.+)"')
    # Sim variants
    re_sim_std = re.compile(r'^Sim = ([\d\.]+)')
    re_sim_l0 = re.compile(r'^sim_l0 = ([\d\.]+)')
    re_sim_l0psi = re.compile(r'^sim_l0psi = ([\d\.]+)')
    # Time
    re_time = re.compile(r'^round\(elapsed, digits = 2\) = ([\d\.]+)')

    current_algo = None
    
    for log_file in log_files:
        print(f"Parsing {log_file}...")
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Check for Algo Header
                m_algo = re_algo.match(line)
                if m_algo:
                    current_algo = m_algo.group(1)
                    if current_algo not in results:
                        results[current_algo] = {'sim': [], 'time': []}
                    continue
                
                if current_algo:
                    # Check for Sim
                    sim_val = None
                    if current_algo.startswith("L0Learn+PSI1"):
                        m = re_sim_l0psi.match(line)
                        if m: 
                            sim_val = float(m.group(1))
                        else:
                            # Fallback to standard Sim if l0psi specific not found
                            m_std = re_sim_std.match(line)
                            if m_std: sim_val = float(m_std.group(1))

                    elif current_algo == "L0Learn":
                        m = re_sim_l0.match(line)
                        if m: 
                            sim_val = float(m.group(1))
                        else:
                             # Fallback to standard Sim just in case
                            m_std = re_sim_std.match(line)
                            if m_std: sim_val = float(m_std.group(1))

                    else:
                        m = re_sim_std.match(line)
                        if m: sim_val = float(m.group(1))
                    
                    if sim_val is not None:
                        results[current_algo]['sim'].append(sim_val)
                    
                    # Check for Time
                    m_time = re_time.match(line)
                    if m_time:
                        time_val = float(m_time.group(1))
                        results[current_algo]['time'].append(time_val)
                        # We don't reset current_algo immediately because sometimes other lines follow, 
                        # but we should be ready for the next algo header which will reset it.

    print("\nResults Summary:")
    print(f"{'Algorithm':<30} | {'Count':<5} | {'Avg Sim':<10} | {'Avg Time (s)':<10}")
    print("-" * 65)
    
    best_sim = -1
    best_sim_algo = None
    
    for algo, metrics in results.items():
        count = len(metrics['sim'])
        if count == 0:
            print(f"{algo:<30} | {0:<5} | {'N/A':<10} | {'N/A':<10}")
            continue
            
        avg_sim = sum(metrics['sim']) / count
        avg_time = sum(metrics['time']) / count
        
        print(f"{algo:<30} | {count:<5} | {avg_sim:.4f}     | {avg_time:.4f}")
        
        if avg_sim > best_sim:
            best_sim = avg_sim
            best_sim_algo = algo
            
    print("-" * 65)
    print(f"Best performing algorithm by Sim: {best_sim_algo} (Sim: {best_sim:.4f})")

if __name__ == "__main__":
    parse_logs()
