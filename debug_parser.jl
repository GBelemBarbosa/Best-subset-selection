# Debug log parsing logic
const LOG = "plots/NewTestTerminal_mixed/logs/robust_mixed_v2_const_0.5_p2000.log"

function debug_parse(log_path)
    lines = readlines(log_path)
    trial_buffer = Dict{String, Any}()
    
    println("Scanning log: $log_path")
    
    for (i, line) in enumerate(lines)
        # Check for flush trigger
        m_n = match(r"--- n=(\d+) Results", line)
        if m_n !== nothing
            n_val = m_n.captures[1]
            # Check buffer stats
            count = length(trial_buffer)
            println("Line $i: Found Table for n=$n_val. Buffer has $count algos.")
            for (k, v) in trial_buffer
                n_sim = length(v[:sim])
                println("  Algo $k: $n_sim sim values")
            end
            
            # Simulate clear
            empty!(trial_buffer)
            continue
        end
        
        # Check Algo Match
        m_algo = match(r"\"(.+)\" = \"\1\"", line)
        if m_algo !== nothing
            algo = m_algo.captures[1]
            if !haskey(trial_buffer, algo)
                trial_buffer[algo] = Dict(:sim => Float64[])
            end
            # Assume we are in algo block, check next few lines
            # (In real loop we use state, here just check match)
            # println("Line $i: Algo Header $algo")
        end
        
        # Check Metric Match (Sim)
        if contains(line, "Sim =")
            # println("Line $i: Found Sim ($line)")
            # In real code we need current_algo state
        end
    end
end

debug_parse(LOG)
