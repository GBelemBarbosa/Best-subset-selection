# spectral_cdss_experiment.jl
# Compare SolverPSI1 variants with NSPG_PGCCD using smart adaptive CV
#
# Variant A (baseline): Original SolverPSI1(NSPG_PGCCD, ...) — γₖ=0.0 after first
#   outer iter, so NSPG recomputes the Rayleigh-quotient initial step every time.
# Variant B (spectral): SolverPSI1_SpectralCDSS — computes a BB1 spectral step
#   between the current iterate (after PGCCD+PSI1) and the previous NSPG output,
#   so the curvature information from the PGCCD+PSI1 displacement is carried into
#   the next NSPG call.
#
# Usage: julia spectral_cdss_experiment.jl [corr] [ρ] [p] [SNR] [k⃰] [ns_start:ns_step:ns_end] [T] [tridiagonal]

# Load all base definitions (packages, parameters, algorithms, CV methods)
include("new_test_terminal.jl")

# ============================================================================
# New SolverPSI1 variant: spectral step from PGCCD+PSI1 displacement
# ============================================================================

"""
    SolverPSI1_SpectralCDSS(x⁰, funcs; γₖ=0.0, max_psi1_fails=50, kwargs...)

Like `SolverPSI1(NSPG_PGCCD, ...)` but instead of resetting `γₖ=0.0` after the
first outer iteration (which forces NSPG to recompute the Rayleigh-quotient
initial step from scratch), this variant computes a BB1 spectral step from the
displacement between the current iterate (after PGCCD+PSI1) and the previous
NSPG output:

    s = β_current − β_prev_spg
    γₖ = ⟨s, s⟩ / ⟨s, X'X s⟩

This carries curvature information across the PGCCD+PSI1 refinement into the
next NSPG call.
"""
function SolverPSI1_SpectralCDSS(x⁰, funcs_tuple; γₖ=0.0, max_psi1_fails=50, kwargs...)
    r!, r, ∇f!, f, F, ∇f_fn, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs_tuple

    β = x⁰
    kᵢ = kₒ = 0
    isPSI1 = false

    # Track previous NSPG output for spectral step computation
    β_prev_spg = nothing
    s = similar(β)       # Pre-allocate displacement vector
    XTXs = similar(β)    # Pre-allocate X'X s

    # Cycle detection (same as original SolverPSI1)
    seen_supports = Set{UInt64}()
    consecutive_fails = 0

    while !isPSI1 && kₒ < kₘₐₓ
        kₒ += 1

        # --- NSPG phase ---
        # First call: γₖ passed from CV (or 0.0 → NSPG computes Rayleigh quotient)
        # Subsequent calls: γₖ = BB1 step from PGCCD+PSI1 displacement
        β_spg, k = NSPG(β, funcs_tuple; γₖ=γₖ, kwargs...)
        kᵢ += k

        # --- PGCCD phase ---
        β_cdss, k2 = PGCCD(β_spg, funcs_tuple; kwargs...)
        kᵢ += k2
        β = β_cdss

        # --- PSI1 phase ---
        β, isPSI1 = PSI1(β, funcs_tuple)

        # --- Compute spectral step for next NSPG call ---
        # BB1: γₖ = ⟨s,s⟩ / ⟨s, X'X s⟩
        # where s = β_current (after PGCCD+PSI1) − β_prev_spg (previous NSPG output)
        if β_prev_spg !== nothing
            s .= β .- β_prev_spg
            ns = dot(s, s)
            if ns > 0
                mul!(XTXs, XTX, s)
                ys = dot(s, XTXs)   # s' X'X s
                if ys > 0
                    γₖ = ns / ys    # BB1 step
                else
                    γₖ = 0.0        # Fallback: let NSPG recompute
                end
            else
                γₖ = 0.0
            end
        else
            γₖ = 0.0  # First iteration: let NSPG compute Rayleigh quotient
        end

        # Save current NSPG output for next iteration's spectral step
        β_prev_spg = copy(β_spg)

        # --- Cycle detection (identical to original SolverPSI1) ---
        if !isPSI1
            consecutive_fails += 1
            support_hash = hash(findall(!iszero, β))
            if support_hash in seen_supports
                break
            end
            push!(seen_supports, support_hash)

            if consecutive_fails >= max_psi1_fails
                break
            end
        else
            consecutive_fails = 0
            empty!(seen_supports)
        end
    end

    return β, kᵢ, kₒ
end

# ============================================================================
# Experiment
# ============================================================================

function main_spectral()
    x⁰ = zeros(Float64, p)
    consecutive_perfect_recoveries = 0
    last_completed_idx = 0

    algo_names = [L"NSPG+PGCCD+CPSI1\ (reset\ \gamma)", L"NSPG+PGCCD+CPSI1\ (spectral\ \gamma)"]
    n_algos = length(algo_names)
    @info "Spectral PGCCD experiment" samples=ns_range trials=T algorithms=algo_names

    Predhist = zeros(length(ns_range), n_algos)
    SUPhist  = zeros(length(ns_range), n_algos)
    Infhist  = zeros(length(ns_range), n_algos)
    Simhist  = zeros(length(ns_range), n_algos)
    SIhist   = zeros(length(ns_range), n_algos)
    ZWhist   = zeros(Float64, length(ns_range), n_algos)
    ZThist   = zeros(Float64, length(ns_range), n_algos)
    ZLhist   = zeros(Float64, length(ns_range), n_algos)
    Timehist = zeros(length(ns_range), n_algos)

    for (t, n) in enumerate(ns_range)
        @info "Sample size n=$n" progress="$(t)/$(length(ns_range))"
        total_start = time()

        for i = 1:T
            vars = variables(corr=corr, ρ=ρ, n=n, p=p, SNR=SNR, k⃰=k⃰, is_tridiagonal=is_tridiagonal)

            # --- Algo 1: Original SolverPSI1(NSPG_PGCCD) — γₖ reset to 0 ---
            trial_start = time()
            β, best_λ, SUP, Pred, Sim, Infv, zw, ss, si = smart_adaptive_cross_validation(
                (x, f; kw...) -> SolverPSI1(NSPG_PGCCD, x, f; kw...),
                vars; NSPG=true, use_refinement=true)
            dt = time() - trial_start
            SUPhist[t, 1] += SUP; Predhist[t, 1] += Pred; Simhist[t, 1] += Sim
            Infhist[t, 1] += Infv; Timehist[t, 1] += dt; SIhist[t, 1] += si
            if zw
                if ss == 1  ZWhist[t, 1] += 1 end
                if ss == 0  ZThist[t, 1] += 1 end
                if ss == -1 ZLhist[t, 1] += 1 end
            end
            @show "Reset γ" SUP Pred Sim Infv round(dt, digits=1)

            # --- Algo 2: SolverPSI1_SpectralCDSS — BB1 spectral step ---
            trial_start = time()
            β, best_λ, SUP, Pred, Sim, Infv, zw, ss, si = smart_adaptive_cross_validation(
                (x, f; kw...) -> SolverPSI1_SpectralCDSS(x, f; kw...),
                vars; NSPG=true, use_refinement=true)
            dt = time() - trial_start
            SUPhist[t, 2] += SUP; Predhist[t, 2] += Pred; Simhist[t, 2] += Sim
            Infhist[t, 2] += Infv; Timehist[t, 2] += dt; SIhist[t, 2] += si
            if zw
                if ss == 1  ZWhist[t, 2] += 1 end
                if ss == 0  ZThist[t, 2] += 1 end
                if ss == -1 ZLhist[t, 2] += 1 end
            end
            @show "Spectral γ" SUP Pred Sim Infv round(dt, digits=1)

            @info "Trial $i/$T completed"
        end

        @info "Completed n=$n in $(round(time() - total_start, digits=1))s"

        last_completed_idx = t

        # Per-n Results Table
        println("\n--- n=$n Results (Avg over $T trials) ---")
        println("Algorithm            | SUP   | Pred    | Sim   | Time (s) | Z-Chosen   | Z-Improv | Refinement (W/T/L)")
        println("---------------------|-------|---------|-------|----------|------------|----------|-------------------")

        @printf("Reset γ (original)   | %-5.1f | %-7.4f | %-5.3f | %-8.1f | %3d / %-3d | %-8.4f | (%d/%d/%d)\n",
            SUPhist[t,1]/T, Predhist[t,1]/T, Simhist[t,1]/T, Timehist[t,1]/T,
            Int(ZWhist[t,1]+ZThist[t,1]+ZLhist[t,1]), T, SIhist[t,1]/T,
            Int(ZWhist[t,1]), Int(ZThist[t,1]), Int(ZLhist[t,1]))
        @printf("Spectral γ (BB1)     | %-5.1f | %-7.4f | %-5.3f | %-8.1f | %3d / %-3d | %-8.4f | (%d/%d/%d)\n",
            SUPhist[t,2]/T, Predhist[t,2]/T, Simhist[t,2]/T, Timehist[t,2]/T,
            Int(ZWhist[t,2]+ZThist[t,2]+ZLhist[t,2]), T, SIhist[t,2]/T,
            Int(ZWhist[t,2]), Int(ZThist[t,2]), Int(ZLhist[t,2]))
        println("---------------------|-------|---------|-------|----------|------------|----------|-------------------\n")
        flush(stdout)

        # Early stopping
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

    @info "All experiments finished"

    # Trim and Normalize
    ns_final = collect(ns_range)[1:last_completed_idx]

    SUPhist  = SUPhist[1:last_completed_idx, :]  ./ T
    Predhist = Predhist[1:last_completed_idx, :] ./ T
    Simhist  = Simhist[1:last_completed_idx, :]  ./ T
    Infhist  = Infhist[1:last_completed_idx, :]  ./ T
    Timehist = Timehist[1:last_completed_idx, :] ./ T
    SIhist   = SIhist[1:last_completed_idx, :]   ./ T
    ZWhist   = ZWhist[1:last_completed_idx, :]
    ZThist   = ZThist[1:last_completed_idx, :]
    ZLhist   = ZLhist[1:last_completed_idx, :]

    # Display results
    println("\n" * "="^60)
    println("RESULTS")
    println("="^60)
    println("\nSupport Size History (SUPhist):")
    display(SUPhist)

    names = reshape(algo_names, 1, :)
    plotname = "SpectralCDSS_$(corr)-$(ρ)-$(p)-$(SNR)-$(k⃰)-$(T)-$(first(ns_range))_$(step(ns_range))_$(last(ns_range))"
    specifics = is_tridiagonal ? "_tridiagonal" : "_standard"

    # Plot theme
    theme(:seaborn_bright)
    default(lw=3)

    println("\nGenerating plots...")

    pPred = plot(ns_final, Predhist, labels=names, xlabel=L"n",
        ylabel=L"\frac{\left\Vert Ax-b\right\Vert^2}{\left\Vert b\right\Vert^2}", left_margin=15mm, dpi=600)
    savefig(pPred, "pnPred-$(plotname)$(specifics).png")

    pSim = plot(ns_final, Simhist, labels=names, xlabel=L"n",
        ylabel=L"\frac{|S\cap S^\dagger|}{\max\{|S|,k^\dagger\}}", left_margin=15mm, dpi=600)
    savefig(pSim, "pnSim-$(plotname)$(specifics).png")

    pTime = plot(ns_final, Timehist, labels=names, xlabel=L"n",
        ylabel="Time (s)", left_margin=15mm, dpi=600, legend=:topleft)
    savefig(pTime, "pnTime-$(plotname)$(specifics).png")

    pInf = plot(ns_final, Infhist, labels=names, xlabel=L"n",
        ylabel=L"\left\Vert x-x^\dagger\right\Vert_\infty", left_margin=15mm, dpi=600)
    savefig(pInf, "pnInf-$(plotname)$(specifics).png")

    pSI = plot(ns_final, SIhist, labels=names, xlabel=L"n",
        ylabel="Avg Sim Improvement", title="Similarity Improvement Comparison",
        left_margin=15mm, dpi=600)
    savefig(pSI, "pnSimImprov-$(plotname)$(specifics).png")

    # Refinement Plots
    choices_reset    = ZWhist[:,1] .+ ZThist[:,1] .+ ZLhist[:,1]
    choices_spectral = ZWhist[:,2] .+ ZThist[:,2] .+ ZLhist[:,2]

    pRef_reset = plot(ns_final, [choices_reset ZWhist[:,1] ZLhist[:,1] SIhist[:,1].*T],
        label=["Choices" "Wins" "Losses" "Total Sim Improv"],
        title="Reset γ Refinement",
        xlabel="n", ylabel="Metric", left_margin=15mm, dpi=600)

    pRef_spectral = plot(ns_final, [choices_spectral ZWhist[:,2] ZLhist[:,2] SIhist[:,2].*T],
        label=["Choices" "Wins" "Losses" "Total Sim Improv"],
        title="Spectral γ Refinement",
        xlabel="n", ylabel="Metric", left_margin=15mm, dpi=600)

    pRef = plot(pRef_reset, pRef_spectral, layout=(2,1), size=(800, 800))
    savefig(pRef, "pnRefinement-$(plotname)$(specifics).png")

    # Final Refinement Averages
    avg_improve_reset    = sum(SIhist[:, 1]) / last_completed_idx
    avg_improve_spectral = sum(SIhist[:, 2]) / last_completed_idx
    println("\n" * "="^60)
    println("REFINEMENT SIMILARITY IMPROVEMENT AVERAGES")
    println("="^60)
    @printf("Reset γ (original): %.6f\n", avg_improve_reset)
    @printf("Spectral γ (BB1):   %.6f\n", avg_improve_spectral)
    println("="^60)

    println("Plots saved.")

    # Aggregated Refinement Statistics
    println("\n" * "="^60)
    println("REFINEMENT STATISTICS (Aggregated)")
    println("="^60)
    total_zw1 = sum(ZWhist[:,1]); total_zt1 = sum(ZThist[:,1]); total_zl1 = sum(ZLhist[:,1])
    total_zw2 = sum(ZWhist[:,2]); total_zt2 = sum(ZThist[:,2]); total_zl2 = sum(ZLhist[:,2])

    @printf("Reset γ (original): %d Wins / %d Tied / %d Losses\n", total_zw1, total_zt1, total_zl1)
    @printf("Spectral γ (BB1):   %d Wins / %d Tied / %d Losses\n", total_zw2, total_zt2, total_zl2)
    println("="^60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_spectral()
end
