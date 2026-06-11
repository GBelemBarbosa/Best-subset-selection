# Command-line running script with memory-optimized snr_experiment.jl functions
# Focuses on n-sweep comparison as in new_test_terminal.jl

# Backup ARGS to prevent snr_experiment.jl from parsing them incorrectly on include
original_args = copy(ARGS)
empty!(ARGS)
include("best_selection_functions.jl")
append!(ARGS, original_args)

# Defaults
corr = length(ARGS) >= 1 ? ARGS[1] : "exp"
ρ = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.9
p = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1000
SNR = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 5.0
k⃰ = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 20

# Parse ns range (format: start:step:end)
if length(ARGS) >= 6
    ns_parts = split(ARGS[6], ":")
    ns_start = parse(Int, ns_parts[1])
    ns_step = parse(Int, ns_parts[2])
    ns_end = parse(Int, ns_parts[3])
    ns_range = ns_start:ns_step:ns_end
else
    ns_range = 100:100:1000  # Default
end

# Parse T (number of trials)
T = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 10

# Parse tridiagonal flag
is_tridiagonal = length(ARGS) >= 8 ? parse(Bool, ARGS[8]) : false

@info "Parameters" corr ρ p SNR k⃰ ns_range T is_tridiagonal L0LEARN_AVAILABLE
flush(stdout)

kₘₐₓ = 1000
ϵ = 10^-7


# Main experiment execution
function main()
    x⁰ = zeros(Float64, p)
    consecutive_perfect_recoveries = 0
    last_completed_idx = 0

    algo_names = [L"NSPG+CPSI1", L"NSPG+PGCCD+CPSI1", L"L0Learn", L"L0Learn\ (CD\ only)", L"PGCCD+CPSI1"]
    n_algos = length(algo_names)
    @info "Experiment configuration" samples = ns_range trials = T algorithms = algo_names

    Predhist = zeros(length(ns_range), n_algos)
    SUPhist = zeros(length(ns_range), n_algos)
    Infhist = zeros(length(ns_range), n_algos)
    Simhist = zeros(length(ns_range), n_algos)
    SIhist = zeros(length(ns_range), n_algos)
    ZWhist = zeros(Float64, length(ns_range), n_algos)
    ZThist = zeros(Float64, length(ns_range), n_algos)
    ZLhist = zeros(Float64, length(ns_range), n_algos)
    Timehist = zeros(length(ns_range), n_algos)

    for (t, n) in enumerate(ns_range)
        @info "Sample size n=$n" progress = "$(t)/$(length(ns_range))"
        total_start = time()

        for i = 1:T
            vars = variables(corr=corr, ρ=ρ, n=n, p=p, SNR=SNR, k⃰=k⃰, is_tridiagonal=is_tridiagonal)
            X, y, Xval, yval, XTX, _, β⃰, _ = vars
            
            suppsim(b) = count(j -> !iszero(β⃰[j]) && !iszero(b[j]), 1:p) / max(k⃰, norm(b, 0))
            predval(b) = norm(X * b .- yval)^2 / norm(yval)^2

            # 1. NSPG+PSI1 (Inverse CV)
            trial_start = time()
            β, best_λ, SUP, Pred, Sim, Infv, zw, ss, si = inverse_cross_validation((x, f; kw...) -> SolverPSI1(NSPG, x, f; kw...), vars; NSPG=true, use_refinement=true)
            dt = time() - trial_start
            SUPhist[t, 1] += SUP; Predhist[t, 1] += Pred; Simhist[t, 1] += Sim; Infhist[t, 1] += Infv; Timehist[t, 1] += dt
            SIhist[t, 1] += si
            if zw
                if ss == 1 ZWhist[t, 1] += 1 end
                if ss == 0 ZThist[t, 1] += 1 end
                if ss == -1 ZLhist[t, 1] += 1 end
            end
            @show "NSPG+CPSI1" SUP Pred Sim Infv round(dt, digits=1)

            if L0LEARN_AVAILABLE
                # Sync data to R once per trial for any algorithms using L0LearnStep under the hood
                RCall.globalEnv[:X_train] = collect(vars[1])
                RCall.globalEnv[:y_train] = collect(vars[2])
                RCall.globalEnv[:X_val] = collect(vars[3])
                RCall.globalEnv[:y_val] = collect(vars[4])
            end

            # 2. NSPG_PGCCD+PSI1 (Smart Adaptive CV, WITH REFINEMENT) [Mixed Strategy]
            trial_start = time()
            β, best_λ, SUP, Pred, Sim, Infv, zw, ss, si = smart_adaptive_cross_validation((x, f; kw...) -> SolverPSI1(NSPG_PGCCD, x, f; kw...), vars; NSPG=true, use_refinement=true)
            dt = time() - trial_start
            SUPhist[t, 2] += SUP; Predhist[t, 2] += Pred; Simhist[t, 2] += Sim; Infhist[t, 2] += Infv; Timehist[t, 2] += dt
            SIhist[t, 2] += si
            if zw
                if ss == 1 ZWhist[t, 2] += 1 end
                if ss == 0 ZThist[t, 2] += 1 end
                if ss == -1 ZLhist[t, 2] += 1 end
            end
            @show "NSPG+PGCCD+CPSI1" SUP Pred Sim Infv round(dt, digits=1)

            if L0LEARN_AVAILABLE
                # 3. L0Learn (Pure L0Learn with Internal CV, includes validation)
                trial_start = time()
                β, best_λ, SUP, Pred, Sim, Infv = pure_l0learn_solver(vars)
                dt = time() - trial_start
                SUPhist[t, 3] += SUP; Predhist[t, 3] += Pred; Simhist[t, 3] += Sim; Infhist[t, 3] += Infv; Timehist[t, 3] += dt
                @show "L0Learn" SUP Pred Sim Infv round(dt, digits=1)

                # 4. L0Learn val CD (Pure L0Learn with algorithm="CD")
                trial_start = time()
                β, best_λ, SUP, Pred, Sim, Infv = pure_l0learn_cd_solver(vars)
                dt = time() - trial_start
                SUPhist[t, 4] += SUP; Predhist[t, 4] += Pred; Simhist[t, 4] += Sim; Infhist[t, 4] += Infv; Timehist[t, 4] += dt
                @show "L0Learn (CD only)" SUP Pred Sim Infv round(dt, digits=1)

                # 5. PGCCD+CPSI1 (Smart Adaptive CV using L0LearnStep)
                trial_start = time()
                β, best_λ, SUP, Pred, Sim, Infv, zw, ss, si = smart_adaptive_cross_validation((x, f; kw...) -> SolverPSI1(L0LearnStep, x, f; kw...), vars; NSPG=false, use_refinement=false)
                dt = time() - trial_start
                SUPhist[t, 5] += SUP; Predhist[t, 5] += Pred; Simhist[t, 5] += Sim; Infhist[t, 5] += Infv; Timehist[t, 5] += dt
                SIhist[t, 5] += si
                if zw
                    if ss == 1 ZWhist[t, 5] += 1 end
                    if ss == 0 ZThist[t, 5] += 1 end
                    if ss == -1 ZLhist[t, 5] += 1 end
                end
                @show "L0Learn+CPSI1 val path" SUP Pred Sim Infv round(dt, digits=1)

            end

            @info "Trial $i/$T completed"
            # Explicit garbage collection to prevent memory leaks from RCall
            if L0LEARN_AVAILABLE
                try RCall.reval("gc(verbose=FALSE)") catch; end
            end
            GC.gc()
        end

        @info "Completed n=$n in $(round(time() - total_start, digits=1))s"
        last_completed_idx = t

        # Per-n Refinement Summary
        println("\n--- n=$n Results (Avg over $T trials) ---")
        println("Algorithm              | SUP   | Pred    | Sim   | Time (s) | Z-Chosen   | Z-Improv | Refinement (W/T/L)")
        println("-----------------------|-------|---------|-------|----------|------------|----------|-------------------")
        
        @printf("NSPG+CPSI1             | %-5.1f | %-7.4f | %-5.3f | %-8.1f | %3d / %-3d | %-8.4f | (%d/%d/%d)\n", SUPhist[t, 1] / T, Predhist[t, 1] / T, Simhist[t, 1] / T, Timehist[t, 1] / T, Int(ZWhist[t, 1] + ZThist[t, 1] + ZLhist[t, 1]), T, SIhist[t, 1] / T, Int(ZWhist[t, 1]), Int(ZThist[t, 1]), Int(ZLhist[t, 1]))
        @printf("NSPG+PGCCD+CPSI1       | %-5.1f | %-7.4f | %-5.3f | %-8.1f | %3d / %-3d | %-8.4f | (%d/%d/%d)\n", SUPhist[t, 2] / T, Predhist[t, 2] / T, Simhist[t, 2] / T, Timehist[t, 2] / T, Int(ZWhist[t, 2] + ZThist[t, 2] + ZLhist[t, 2]), T, SIhist[t, 2] / T, Int(ZWhist[t, 2]), Int(ZThist[t, 2]), Int(ZLhist[t, 2]))
        @printf("L0Learn                | %-5.1f | %-7.4f | %-5.3f | %-8.1f |    N/A     |    N/A   | N/A\n", SUPhist[t, 3] / T, Predhist[t, 3] / T, Simhist[t, 3] / T, Timehist[t, 3] / T)
        @printf("L0Learn (CD only)      | %-5.1f | %-7.4f | %-5.3f | %-8.1f |    N/A     |    N/A   | N/A\n", SUPhist[t, 4] / T, Predhist[t, 4] / T, Simhist[t, 4] / T, Timehist[t, 4] / T)
        @printf("L0Learn+CPSI1 val path | %-5.1f | %-7.4f | %-5.3f | %-8.1f | %3d / %-3d | %-8.4f | (%d/%d/%d)\n", SUPhist[t, 5] / T, Predhist[t, 5] / T, Simhist[t, 5] / T, Timehist[t, 5] / T, Int(ZWhist[t, 5] + ZThist[t, 5] + ZLhist[t, 5]), T, SIhist[t, 5] / T, Int(ZWhist[t, 5]), Int(ZThist[t, 5]), Int(ZLhist[t, 5]))
        println("-----------------------|-------|---------|-------|----------|------------|----------|-------------------\n")
        flush(stdout)

        # Check for early stopping
        avg_sims = Simhist[t, :] ./ T
        if all(x -> x >= 0.999, avg_sims)
            consecutive_perfect_recoveries += 1
            if consecutive_perfect_recoveries >= 2
                @info "Perfect recovery achieved for 2 consecutive n steps. Stopping early."
                break
            end
        else
            consecutive_perfect_recoveries = 0
        end
    end

    # Trim and Normalize
    ns_final = collect(ns_range)[1:last_completed_idx]
    SUPhist = SUPhist[1:last_completed_idx, :] ./ T
    Predhist = Predhist[1:last_completed_idx, :] ./ T
    Simhist = Simhist[1:last_completed_idx, :] ./ T
    Infhist = Infhist[1:last_completed_idx, :] ./ T
    Timehist = Timehist[1:last_completed_idx, :] ./ T
    SIhist = SIhist[1:last_completed_idx, :] ./ T

    # Display final results
    println("\n" * "="^60)
    println("FINAL RESULTS OVER SWEEP")
    println("="^60)
    
    # Save a text summary
    summary_file = "new_test_terminal_snr_summary.txt"
    open(summary_file, "w") do io
        println(io, "Algorithm,Sample_Size,SUP,Pred_MSE,Sim,Time")
        for (idx, n) in enumerate(ns_final)
            for a in 1:n_algos
                println(io, "$(algo_names[a]),$n,$(SUPhist[idx, a]),$(Predhist[idx, a]),$(Simhist[idx, a]),$(Timehist[idx, a])")
            end
        end
    end
    @info "Results summary saved to $summary_file"

    try
        # Attempt to plot
        @eval using Plots, StatsPlots, Plots.PlotMeasures
        plotname = "VariableN_L0_$(corr)-$(ρ)-$(p)-$(SNR)-$(k⃰)-$(T)-$(first(ns_range))_$(step(ns_range))_$(last(ns_range))"
        println("\nGenerating plots...")
        names = reshape(algo_names, 1, n_algos)
        pPred = plot(ns_final, Predhist, linewidth=3, labels=names, xlabel=L"n", ylabel=L"\frac{\left\Vert Ax-b\right\Vert^2}{\left\Vert b\right\Vert^2}", left_margin=15mm, dpi=600)
        savefig(pPred, "pnPred-$(plotname).png")
        pSim = plot(ns_final, Simhist, linewidth=3, labels=names, xlabel=L"n", ylabel=L"\frac{|S\cap S^\dagger|}{\max\{|S|,k^\dagger\}}", left_margin=15mm, dpi=600)
        savefig(pSim, "pnSim-$(plotname).png")
        pTime = plot(ns_final, Timehist, linewidth=3, labels=names, xlabel=L"n", ylabel="Time (s)", left_margin=15mm, dpi=600, legend=:topleft)
        savefig(pTime, "pnTime-$(plotname).png")
        pInf = plot(ns_final, Infhist, linewidth=3, labels=names, xlabel=L"n", ylabel=L"\left\Vert x-x^\dagger\right\Vert_\infty", left_margin=15mm, dpi=600)
        savefig(pInf, "pnInf-$(plotname).png")
        println("Plots saved locally.")
    catch e
        @warn "Plotting failed: $e"
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
