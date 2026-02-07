using Plots, StatsPlots, Plots.PlotMeasures
using LaTeXStrings
using Printf

# Common Parameters
T = 10
k_star = 100
is_tridiagonal = false # Default, overwritten per exp
SNR = 10.0

# Define experiments to process
# List of (type, corr, rho, p, n_range)
# Types: "Regular", "Tridiagonal"
# Based on logs and dirs.
experiments = [
    ("Regular", "const", 0.5, 2000, 200:200:2000),
    ("Regular", "const", 0.9, 1000, 100:100:1000),
    ("Regular", "exp", 0.5, 2000, 200:200:2000),
    ("Regular", "exp", 0.9, 1000, 100:100:1000),
    # Tridiagonal
    ("Tridiagonal", "const", 0.5, 2000, 200:200:2000),
    ("Tridiagonal", "const", 0.9, 1000, 100:100:1000),
    ("Tridiagonal", "exp", 0.5, 2000, 200:200:2000),
    ("Tridiagonal", "exp", 0.9, 1000, 100:100:1000)
]

# Updated Algorithm Names
# User request: "l0learn+cpsi1 -> l0learn+cpsi1 val path"
algo_names = [
    L"NSPG+CPSI1", 
    L"NSPG+PGCCD+CPSI1", 
    L"L0Learn+CPSI1\ val",  
    L"L0Learn\ val\ (CD\ only)", 
    L"L0Learn+CPSI1\ val\ path"
]
n_algos = length(algo_names)

base_log_dir = "/home/gggt4/TeX/CDSSvsSPG/src/images/Mixed_v2/logs"
base_img_dir = "/home/gggt4/TeX/CDSSvsSPG/src/images/Mixed_v2" 
# Structure: base_img_dir / Type / Config

function process_experiment(type, corr, ρ, p, ns_range)
    ns = collect(ns_range)
    println("\nProcessing: Type=$type, corr=$corr, rho=$ρ, p=$p, ns=$(ns)")
    
    is_tri = (type == "Tridiagonal")
    
    # Construct filenames
    # Log file pattern: 
    # Regular: mixed_v2_$(corr)_$(rho)_p$(p).log
    # Tridiagonal: mixed_v2_tri_$(corr)_$(rho)_p$(p).log 
    
    log_prefix = is_tri ? "mixed_v2_tri" : "mixed_v2"
    log_filename = "$(log_prefix)_$(corr)_$(ρ)_p$(p).log"
    log_path = joinpath(base_log_dir, log_filename)
    
    # Output Directory pattern: Type/$(corr)-$(rho)-$(p)
    img_subdir = joinpath(type, "$(corr)-$(ρ)-$(p)")
    out_dir = joinpath(base_img_dir, img_subdir)
    
    # Plot Filename pattern
    # Tridiagonal uses "NewTerminal_mixed_v2_tri_..."
    plot_prefix = is_tri ? "NewTerminal_mixed_v2_tri" : "NewTerminal_mixed_v2"
    out_name = "$(plot_prefix)_$(corr)_$(ρ)_p$(p)"

    if !isfile(log_path)
        println("  Error: Log file not found at $log_path")
        return
    end
    
    if !isdir(out_dir)
        println("  Error: Output directory not found at $out_dir")
        # Ensure it exists?
        try
            mkpath(out_dir)
            println("  Created missing directory: $out_dir")
        catch e
            println("  Failed to create directory: $e")
            return
        end
    end

    # Data Containers
    Predhist = zeros(length(ns), n_algos)
    Simhist = zeros(length(ns), n_algos)
    Timehist = zeros(length(ns), n_algos)
    Infhist = zeros(length(ns), n_algos)
    SIhist = zeros(length(ns), n_algos) 
    RefStats = zeros(length(ns), 2, 4) 

    # Parse Summary Tables
    lines = readlines(log_path)
    current_n_idx = 0
    
    for (i, line) in enumerate(lines)
        if startswith(line, "--- n=")
            m = match(r"--- n=(\d+)", line)
            if m !== nothing
                n_val = parse(Int, m.captures[1])
                idx = findfirst(==(n_val), ns)
                if idx !== nothing
                    current_n_idx = idx
                    # Row 1: SPG+PSI1
                    vals1 = split(lines[i+3], "|")
                    Predhist[current_n_idx, 1] = parse(Float64, strip(vals1[3]))
                    Simhist[current_n_idx, 1] = parse(Float64, strip(vals1[4]))
                    Timehist[current_n_idx, 1] = parse(Float64, strip(vals1[5]))
                    ref_part = match(r"\((\d+)/(\d+)/(\d+)\)", lines[i+3])
                    if ref_part !== nothing
                        w, t, l = parse.(Int, ref_part.captures)
                        RefStats[current_n_idx, 1, 2] = w; RefStats[current_n_idx, 1, 3] = l; RefStats[current_n_idx, 1, 1] = w+t+l
                    end
                    SIhist[current_n_idx, 1] = parse(Float64, strip(vals1[7]))

                    # Row 2: SPGpCDSS+PSI1
                    vals2 = split(lines[i+4], "|")
                    Predhist[current_n_idx, 2] = parse(Float64, strip(vals2[3]))
                    Simhist[current_n_idx, 2] = parse(Float64, strip(vals2[4]))
                    Timehist[current_n_idx, 2] = parse(Float64, strip(vals2[5]))
                    ref_part = match(r"\((\d+)/(\d+)/(\d+)\)", lines[i+4])
                    if ref_part !== nothing
                         w, t, l = parse.(Int, ref_part.captures)
                        RefStats[current_n_idx, 2, 2] = w; RefStats[current_n_idx, 2, 3] = l; RefStats[current_n_idx, 2, 1] = w+t+l
                    end
                    SIhist[current_n_idx, 2] = parse(Float64, strip(vals2[7]))

                    # Row 3: L0LearnPSI1 val
                    vals3 = split(lines[i+5], "|")
                    Predhist[current_n_idx, 3] = parse(Float64, strip(vals3[3]))
                    Simhist[current_n_idx, 3] = parse(Float64, strip(vals3[4]))
                    Timehist[current_n_idx, 3] = parse(Float64, strip(vals3[5]))

                    # Row 4: L0Learn val CD
                    vals4 = split(lines[i+6], "|")
                    Predhist[current_n_idx, 4] = parse(Float64, strip(vals4[3]))
                    Simhist[current_n_idx, 4] = parse(Float64, strip(vals4[4]))
                    Timehist[current_n_idx, 4] = parse(Float64, strip(vals4[5]))

                    # Row 5: L0Learn+P1 l-cv
                    vals5 = split(lines[i+7], "|")
                    Predhist[current_n_idx, 5] = parse(Float64, strip(vals5[3]))
                    Simhist[current_n_idx, 5] = parse(Float64, strip(vals5[4]))
                    Timehist[current_n_idx, 5] = parse(Float64, strip(vals5[5]))
                end
            end
        end
    end

    # Parse Infv using counter-based approach (robust against missing Info logs)
    # Reset counters
    Infhist .= 0.0
    current_n_idx = 1
    trials_seen_for_n = 0
    
    # We track Algorithm 1 ("SPG+PSI1") to count trials.
    # Assumes sequential execution: 10 trials of n_1, then 10 trials of n_2, etc.
    
    # Pre-calculate algo strings to match
    # Note: Log lines are like: "SPG+PSI1" = "SPG+PSI1"
    algo_keys = [
        "\"SPG+PSI1\" = \"SPG+PSI1\"",
        "\"SPGpCDSS+PSI1\" = \"SPGpCDSS+PSI1\"",
        "\"L0LearnPSI1 val\" = \"L0LearnPSI1 val\"",
        "\"L0Learn val CD\" = \"L0Learn val CD\"",
        "\"L0Learn+P1 val l-cv\" = \"L0Learn+P1 val l-cv\""
    ]
    
    current_algo_idx = 0
    
    for line in lines
        # Check if we hit a new run start (reset)
        if occursin("Loading packages", line)
             # If we see a restart, reset parsers to ensure we get the latest clean run? 
             # Or maybe just ignore if we already have data. 
             # For now, let's just parse the first clean block we find.
             # If we are deep in the file, we might be in appended logs.
             # Let's rely on the first sequence found.
        end

        # Identify Algorithm
        matched_algo = 0
        for (idx, key) in enumerate(algo_keys)
            if occursin(key, line)
                matched_algo = idx
                break
            end
        end
        
        if matched_algo == 1
            # Start of a new trial for the FIRST algorithm
            trials_seen_for_n += 1
            if trials_seen_for_n > T
                # Moved to next n
                current_n_idx += 1
                trials_seen_for_n = 1
            end
        end
        
        if matched_algo > 0
            current_algo_idx = matched_algo
        end
        
        # Parse Infv value
        if startswith(strip(line), "Infv =") && current_algo_idx > 0 && current_n_idx <= length(ns)
            try
                val = parse(Float64, strip(split(line, "=")[2]))
                Infhist[current_n_idx, current_algo_idx] += val
            catch
            end
        end
    end
    Infhist ./= T
    
    RefStats[:, 1, 4] = SIhist[:, 1] .* T
    RefStats[:, 2, 4] = SIhist[:, 2] .* T

    println("  Parsing complete. Generating plots to $out_dir...")

    # Plotting
    names_mat = reshape(algo_names, 1, :)
    
    # Use \left\Vert and \right\Vert for robust "norm" look
    pPred = plot(ns, Predhist, labels=names_mat, xlabel=L"n", ylabel=L"\frac{\left\Vert Ax-b\right\Vert^2}{\left\Vert b\right\Vert^2}", left_margin=15mm, dpi=600)
    savefig(pPred, joinpath(out_dir, "pnPred-$(out_name).png"))

    # Conditional Legend for Similarity
    # Const: Regular->TopLeft, Tridiagonal->BottomRight (per user feedback)
    # Exp: BottomRight
    sim_legend = :bottomright
    if corr == "const" && !is_tri
        sim_legend = :topleft
    end
    
    pSim = plot(ns, Simhist, labels=names_mat, xlabel=L"n", ylabel=L"\frac{|S\cap S^\dagger|}{\max\{|S|,k^\dagger\}}", left_margin=15mm, dpi=600, legend=sim_legend)
    savefig(pSim, joinpath(out_dir, "pnSim-$(out_name).png"))

    # Time: User requested fixes imply TopLeft is better (avoiding rising curves)
    pTime = plot(ns, Timehist, labels=names_mat, xlabel=L"n", ylabel="Time (s)", left_margin=15mm, dpi=600, legend=:topleft)
    savefig(pTime, joinpath(out_dir, "pnTime-$(out_name).png"))

    pInf = plot(ns, Infhist, labels=names_mat, xlabel=L"n", ylabel=L"\left\Vert x-x^\dagger\right\Vert_\infty", left_margin=15mm, dpi=600)
    savefig(pInf, joinpath(out_dir, "pnInf-$(out_name).png"))

    pSI = plot(ns, SIhist[:, 1:2], labels=names_mat[:, 1:2], xlabel=L"n", ylabel="Avg Sim Improvement", title="Similarity Improvement Comparison", left_margin=15mm, dpi=600)
    savefig(pSI, joinpath(out_dir, "pnSimImprov-$(out_name).png"))

    choices_spg = RefStats[:,1,1]
    choices_cdss = RefStats[:,2,1]
    pRef_SPG = plot(ns, [choices_spg RefStats[:,1,2] RefStats[:,1,3] RefStats[:,1,4]], 
        label=["Choices" "Wins" "Losses" "Total Sim Improv"], 
        title="NSPG+CPSI1 Refinement", 
        xlabel="n", ylabel="Metric", left_margin=15mm, dpi=600)

    pRef_CDSS = plot(ns, [choices_cdss RefStats[:,2,2] RefStats[:,2,3] RefStats[:,2,4]], 
        label=["Choices" "Wins" "Losses" "Total Sim Improv"], 
        title="NSPG+PGCCD+CPSI1 Refinement", 
        xlabel="n", ylabel="Metric", left_margin=15mm, dpi=600)

    pRef = plot(pRef_SPG, pRef_CDSS, layout=(2,1), size=(800, 800))
    savefig(pRef, joinpath(out_dir, "pnRefinement-$(out_name).png"))
    
    # Sync to Best selection repo
    # Destination structure: /mnt/c/Users/gggt4/Documents/Pesquisa/Best selection/plots/Mixed_v2 / (Tridiagonal/Regular) / config
    # Note: User provided path "plots/Mixed_v2" in Best selection.
    
    bs_base = "/mnt/c/Users/gggt4/Documents/Pesquisa/Best selection/plots/Mixed_v2"
    bs_subdir = joinpath(bs_base, type, "$(corr)-$(ρ)-$(p)") # e.g. Regular/const-0.5-2000
    
    if !isdir(bs_subdir)
         try mkpath(bs_subdir) catch end
    end
    
    if isdir(bs_subdir)
        cp(joinpath(out_dir, "pnPred-$(out_name).png"), joinpath(bs_subdir, "pnPred-$(out_name).png"), force=true)
        cp(joinpath(out_dir, "pnSim-$(out_name).png"), joinpath(bs_subdir, "pnSim-$(out_name).png"), force=true)
        cp(joinpath(out_dir, "pnTime-$(out_name).png"), joinpath(bs_subdir, "pnTime-$(out_name).png"), force=true)
        # cp(joinpath(out_dir, "pnTime_Clean-$(out_name).png"), joinpath(bs_subdir, "pnTime_Clean-$(out_name).png"), force=true) # Removed
        cp(joinpath(out_dir, "pnInf-$(out_name).png"), joinpath(bs_subdir, "pnInf-$(out_name).png"), force=true)
        cp(joinpath(out_dir, "pnSimImprov-$(out_name).png"), joinpath(bs_subdir, "pnSimImprov-$(out_name).png"), force=true)
        cp(joinpath(out_dir, "pnRefinement-$(out_name).png"), joinpath(bs_subdir, "pnRefinement-$(out_name).png"), force=true)
        println("  Synced to Best selection: $bs_subdir")
    else
        println("  Warning: Could not sync to $bs_subdir (dir unavailable)")
    end
end

# Global plot theme settings
theme(:seaborn_bright)
default(lw=3)

# Run batch
for exp in experiments
    process_experiment(exp...)
end

println("\nAll experiments processed and synced.")
