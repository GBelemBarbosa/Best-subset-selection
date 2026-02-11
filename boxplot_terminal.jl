# Terminal-running boxplot experiment
# Usage: julia boxplot_terminal.jl [corr] [ρ] [n] [p] [SNR] [k⃰] [T] [mode]
# mode: "cpsi1" (with CPSI1, Section 5.3) or "nocpsi1" (pure algorithms, Section 5.2)
# Example: julia boxplot_terminal.jl exp 0.5 500 2000 10 100 50 cpsi1

@info "Loading packages..."
using Random
using Distributions
using LinearAlgebra
using BenchmarkTools, Profile, TimerOutputs
using Plots, StatsPlots, Plots.PlotMeasures
using LaTeXStrings
using ThreadsX
using Printf

@info "Packages loaded successfully"

# ============================================================================
# Parameters (command-line arguments)
# ============================================================================

corr = length(ARGS) >= 1 ? ARGS[1] : "exp"
ρ = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.5
n = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 500
p = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 2000
SNR = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 10.0
k⃰ = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 100
T = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 50
mode = length(ARGS) >= 8 ? ARGS[8] : "cpsi1"

@assert mode in ("cpsi1", "nocpsi1") "mode must be 'cpsi1' or 'nocpsi1', got '$mode'"

@info "Parameters" corr ρ n p SNR k⃰ T mode
flush(stdout)

kₘₐₓ = 1000
ϵ = 10^-7

# ============================================================================
# Variable Generation (from new_test_terminal.jl)
# ============================================================================

function generate_data(; corr="exp", ρ=0.9, n=250, p=1000, SNR=5, k⃰=20)
    Σ = corr == "exp" ? [ρ^abs(i - j) for i = 1:p, j = 1:p] : [1 - (1 - ρ) * (i != j) for i = 1:p, j = 1:p]
    d = MvNormal(zeros(p), Σ)
    X = rand(d, n)'
    for i = 1:p
        X[:, i] /= norm(X[:, i])
    end
    β⃰ = [1.0 * (mod(i, Int(p / k⃰)) == 0) for i = 1:p]
    σ = sqrt(norm(X * β⃰)^2 / (n * SNR))
    y = X * β⃰ + randn(n) * σ

    XTX = X'X

    return X, y, XTX, p, β⃰, k⃰
end

# ============================================================================
# Function Definitions (from new_test_terminal.jl)
# ============================================================================

function make_funcs(X, y, XTX, λ₀)
    HT = sqrt(2 * λ₀)
    r!(r, β) = (mul!(r, X, β); r .-= y)
    r(β) = X * β - y
    f(r) = norm(r)^2 / 2
    h(β) = λ₀ * norm(β, 0)
    F(r, β) = f(r) + h(β)
    ∇f!(∇f, r) = mul!(∇f, X', r)
    ∇f(r) = X'r
    proxl0(x) = (abs(x) >= HT) * x
    proxl0(x, τ) = (abs.(x) .>= sqrt(2λ₀ * τ)) .* x
    proxl0VM(x, Uₖ) = (abs.(x) .>= sqrt.(2λ₀ .* Uₖ)) .* x

    # In-place versions for reduced allocations
    function proxl0!(out, x, τ)
        thresh = sqrt(2λ₀ * τ)
        @inbounds @simd for i in eachindex(out)
            out[i] = abs(x[i]) >= thresh ? x[i] : zero(eltype(x))
        end
        return out
    end

    function proxl0VM!(out, x, Uₖ)
        @inbounds @simd for i in eachindex(out)
            out[i] = abs(x[i]) >= sqrt(2λ₀ * Uₖ[i]) ? x[i] : zero(eltype(x))
        end
        return out
    end

    return r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX
end

# ============================================================================
# VMSPG Algorithm (from new_test_terminal.jl)
# ============================================================================

function VMSPG(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64), µ=10^-3, kwargs...)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs

    n_vars = length(x⁰)
    xᵏ = copy(x⁰)
    xᵏ⁻¹ = copy(x⁰)
    sᵏ = similar(x⁰)
    yᵏ = similar(x⁰)
    ∇fxᵏ = similar(x⁰)
    ∇fxᵏ⁻¹ = similar(x⁰)
    rᵏ = similar(x⁰, size(X, 1))
    temp = similar(x⁰)

    r!(rᵏ, xᵏ)
    ∇f!(∇fxᵏ, rᵏ)
    Fxᵏ⁻¹ = Fxᵏ = F(rᵏ, xᵏ)

    # Initial step
    temp .= x⁰ .- ∇fxᵏ .* 10^-5
    r!(rᵏ, temp)
    yᵏ .= ∇fxᵏ .- X'rᵏ
    yᵏTsᵏ = dot(yᵏ, ∇fxᵏ)
    nsᵏ = γₖ¹ = dot(∇fxᵏ, ∇fxᵏ) * 10^-5 / yᵏTsᵏ
    γₖ² = yᵏTsᵏ * 10^-5 / dot(yᵏ, yᵏ)

    Uₖ = similar(x⁰)
    Uₖ₋₁ = similar(x⁰)
    @inbounds for i in eachindex(Uₖ)
        val = 10^-5 * ∇fxᵏ[i] * ∇fxᵏ[i] / (∇fxᵏ[i] * yᵏ[i])
        Uₖ[i] = min(max(val, γₖ²), γₖ¹)
    end
    copyto!(Uₖ₋₁, Uₖ)

    lastₘ = fill(Fxᵏ, m)

    r!(rᵏ, xᵏ)

    for k = 1:kₘₐₓ
        Fxₗ₍ₖ₎ = maximum(lastₘ)

        while true
            temp .= xᵏ⁻¹ .- Uₖ .* ∇fxᵏ
            proxl0VM!(xᵏ, temp, Uₖ)
            r!(rᵏ, xᵏ)
            Fxᵏ = F(rᵏ, xᵏ)
            sᵏ .= xᵏ .- xᵏ⁻¹
            nsᵏ = dot(sᵏ, sᵏ)

            sum_diag = zero(eltype(sᵏ))
            @inbounds @simd for i in eachindex(sᵏ)
                sum_diag += sᵏ[i] * sᵏ[i] / Uₖ[i]
            end
            if Fxᵏ + δ * sum_diag / 2 <= Fxₗ₍ₖ₎
                break
            end

            BLAS.scal!(τ, Uₖ)
            if any(isnan, Uₖ) || any(x -> x < γₘᵢₙ, Uₖ)
                break
            end
        end

        if abs(Fxᵏ⁻¹ - Fxᵏ) / Fxᵏ <= ϵ
            return xᵏ, k
        end

        popfirst!(lastₘ)
        push!(lastₘ, Fxᵏ)
        copyto!(xᵏ⁻¹, xᵏ)
        Fxᵏ⁻¹ = Fxᵏ
        copyto!(∇fxᵏ⁻¹, ∇fxᵏ)
        ∇f!(∇fxᵏ, rᵏ)
        nsᵏ = dot(sᵏ, sᵏ)
        yᵏ .= ∇fxᵏ .- ∇fxᵏ⁻¹
        nyᵏ = dot(yᵏ, yᵏ)

        yᵏTsᵏ = dot(yᵏ, sᵏ)
        γₖ¹ = yᵏTsᵏ > 0 ? nsᵏ / yᵏTsᵏ : sqrt(nsᵏ / nyᵏ)
        γₖ² = yᵏTsᵏ > 0 ? yᵏTsᵏ / nyᵏ : 1 / γₖ¹

        Uₖ₋₁, Uₖ = Uₖ, Uₖ₋₁
        @inbounds @simd for i in eachindex(Uₖ)
            val = (sᵏ[i] * sᵏ[i] + µ * Uₖ₋₁[i]) / (sᵏ[i] * yᵏ[i] + µ)
            Uₖ[i] = min(max(val, γₖ²), γₖ¹)
        end
    end

    return xᵏ, kₘₐₓ
end

# ============================================================================
# SPGH Algorithm (from new_test_terminal.jl)
# ============================================================================

function SPGH(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64), kwargs...)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs

    xᵏ = copy(x⁰)
    xᵏ⁻¹ = copy(x⁰)
    sᵏ = similar(x⁰)
    yᵏ = similar(x⁰)
    ∇fxᵏ = similar(x⁰)
    ∇fxᵏ⁻¹ = similar(x⁰)
    rᵏ = similar(x⁰, size(X, 1))
    temp = similar(x⁰)

    r!(rᵏ, xᵏ)
    ∇f!(∇fxᵏ, rᵏ)
    Fxᵏ⁻¹ = Fxᵏ = F(rᵏ, xᵏ)

    temp .= x⁰ .- ∇fxᵏ .* 10^-5
    r!(rᵏ, temp)
    yᵏ .= ∇fxᵏ .- X'rᵏ
    yᵏTsᵏ = dot(yᵏ, ∇fxᵏ)
    nsᵏ = γₖ¹ = dot(∇fxᵏ, ∇fxᵏ) * 10^-5 / yᵏTsᵏ
    γₖ² = yᵏTsᵏ * 10^-5 / dot(yᵏ, yᵏ)
    γₖ = γₖ¹ < 2 * γₖ² ? γₖ² : γₖ¹ - γₖ² / 2
    lastₘ = fill(Fxᵏ, m)

    r!(rᵏ, xᵏ)

    for k = 1:kₘₐₓ
        Fxₗ₍ₖ₎ = maximum(lastₘ)

        while true
            temp .= xᵏ⁻¹ .- γₖ .* ∇fxᵏ
            proxl0!(xᵏ, temp, γₖ)
            r!(rᵏ, xᵏ)
            Fxᵏ = F(rᵏ, xᵏ)
            sᵏ .= xᵏ .- xᵏ⁻¹
            nsᵏ = dot(sᵏ, sᵏ)

            if Fxᵏ + δ * nsᵏ / (2 * γₖ) <= Fxₗ₍ₖ₎
                break
            end

            γₖ *= τ
            if isnan(γₖ) || γₖ < γₘᵢₙ
                break
            end
        end

        if abs(Fxᵏ⁻¹ - Fxᵏ) / Fxᵏ <= ϵ
            return xᵏ, k
        end

        popfirst!(lastₘ)
        push!(lastₘ, Fxᵏ)
        copyto!(xᵏ⁻¹, xᵏ)
        Fxᵏ⁻¹ = Fxᵏ
        copyto!(∇fxᵏ⁻¹, ∇fxᵏ)
        ∇f!(∇fxᵏ, rᵏ)
        yᵏ .= ∇fxᵏ .- ∇fxᵏ⁻¹
        nyᵏ = dot(yᵏ, yᵏ)
        yᵏTsᵏ = dot(yᵏ, sᵏ)
        γₖ¹ = yᵏTsᵏ > 0 ? nsᵏ / yᵏTsᵏ : sqrt(nsᵏ / nyᵏ)
        γₖ² = yᵏTsᵏ > 0 ? yᵏTsᵏ / nyᵏ : 1 / γₖ¹
        γₖ = γₖ¹ < 2 * γₖ² ? γₖ² : γₖ¹ - γₖ² / 2
    end

    return xᵏ, kₘₐₓ
end

# ============================================================================
# SPG Algorithm (from new_test_terminal.jl)
# ============================================================================

function SPG(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64), γₖ=0.0, kwargs...)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs

    xᵏ = copy(x⁰)
    xᵏ⁻¹ = copy(x⁰)
    sᵏ = similar(x⁰)
    yᵏ = similar(x⁰)
    ∇fxᵏ = similar(x⁰)
    ∇fxᵏ⁻¹ = similar(x⁰)
    rᵏ = similar(x⁰, size(X, 1))
    temp = similar(x⁰)

    r!(rᵏ, xᵏ)
    ∇f!(∇fxᵏ, rᵏ)
    Fxᵏ⁻¹ = Fxᵏ = F(rᵏ, xᵏ)

    if γₖ == 0.0
        temp .= x⁰ .- ∇fxᵏ .* 10^-5
        r!(rᵏ, temp)
        yᵏ .= ∇fxᵏ .- X'rᵏ
        nsᵏ = γₖ = dot(∇fxᵏ, ∇fxᵏ) * 10^-5 / dot(yᵏ, ∇fxᵏ)
    else
        XTX∇fxᵏ = XTX * ∇fxᵏ
        γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
    end
    nsᵏ = γₖ
    lastₘ = fill(Fxᵏ, m)

    r!(rᵏ, xᵏ)

    for k = 1:kₘₐₓ
        Fxₗ₍ₖ₎ = maximum(lastₘ)

        while true
            temp .= xᵏ⁻¹ .- γₖ .* ∇fxᵏ
            proxl0!(xᵏ, temp, γₖ)
            r!(rᵏ, xᵏ)
            Fxᵏ = F(rᵏ, xᵏ)
            sᵏ .= xᵏ .- xᵏ⁻¹
            nsᵏ = dot(sᵏ, sᵏ)

            if Fxᵏ + δ * nsᵏ / (2 * γₖ) <= Fxₗ₍ₖ₎
                break
            end

            γₖ *= τ
            if isnan(γₖ) || γₖ < γₘᵢₙ
                break
            end
        end

        if abs(Fxᵏ⁻¹ - Fxᵏ) / Fxᵏ <= ϵ
            return xᵏ, k
        end

        popfirst!(lastₘ)
        push!(lastₘ, Fxᵏ)
        copyto!(xᵏ⁻¹, xᵏ)
        Fxᵏ⁻¹ = Fxᵏ
        copyto!(∇fxᵏ⁻¹, ∇fxᵏ)
        ∇f!(∇fxᵏ, rᵏ)

        yᵏ .= ∇fxᵏ .- ∇fxᵏ⁻¹
        γₖ = nsᵏ / dot(yᵏ, sᵏ)
        if γₖ > γₘₐₓ || γₖ < γₘᵢₙ
            γₖ = sqrt(nsᵏ / dot(yᵏ, yᵏ))
        end
    end

    return xᵏ, kₘₐₓ
end

# ============================================================================
# CDSS Algorithm (from new_test_terminal.jl)
# ============================================================================

function has_same_support(x, y)
    for i in eachindex(x)
        if iszero(x[i]) != iszero(y[i])
            return false
        end
    end
    return true
end

function CDSS(x⁰, funcs; sortperc=1 / 4, ActiveSetNum=10, kwargs...)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs

    xᵏ = copy(x⁰)
    rᵏ = -r(xᵏ)
    Fxᵏ⁻¹ = F(rᵏ, xᵏ)

    n_vars = length(x⁰)
    ksort = round(Int64, n_vars * sortperc)

    greedy = partialsortperm(abs.(∇f(rᵏ)), 1:ksort, rev=true)
    greedy = vcat(greedy, setdiff(1:n_vars, greedy))

    xᵏ⁻¹ = copy(xᵏ)
    SameSuppCounter = 0
    Stabilized = false
    Order = greedy

    for k = 1:kₘₐₓ
        if !Stabilized
            if has_same_support(xᵏ, xᵏ⁻¹)
                SameSuppCounter += 1
                if SameSuppCounter == ActiveSetNum - 1
                    Order = findall(!iszero, xᵏ)
                    Stabilized = true
                end
            else
                SameSuppCounter = 0
            end
        end

        copyto!(xᵏ⁻¹, xᵏ)

        @inbounds for i in Order
            xi = proxl0(dot(rᵏ, view(X, :, i)) + xᵏ[i])
            if xi != xᵏ[i]
                BLAS.axpy!(xᵏ[i] - xi, view(X, :, i), rᵏ)
                xᵏ[i] = xi
            end
        end

        Fxᵏ = F(rᵏ, xᵏ)
        if (Fxᵏ⁻¹ - Fxᵏ) / Fxᵏ <= ϵ
            if Stabilized
                optimality_violated = false
                @inbounds for i in 1:n_vars
                    xi = proxl0(dot(rᵏ, view(X, :, i)) + xᵏ[i])
                    if xi != xᵏ[i]
                        BLAS.axpy!(xᵏ[i] - xi, view(X, :, i), rᵏ)
                        xᵏ[i] = xi
                        optimality_violated = true
                    end
                end

                if !optimality_violated
                    return xᵏ, k
                else
                    Stabilized = false
                    SameSuppCounter = 0
                    Order = greedy
                    Fxᵏ⁻¹ = F(rᵏ, xᵏ)
                    continue
                end
            else
                return xᵏ, k
            end
        end

        Fxᵏ⁻¹ = Fxᵏ
    end

    return xᵏ, kₘₐₓ
end

# ============================================================================
# Combined Algorithms
# ============================================================================

function SPGpCDSS(x⁰, funcs; γₖ=0.0, kwargs...)
    x, k = SPG(x⁰, funcs; γₖ=γₖ, kwargs...)
    x, k2 = CDSS(x, funcs; kwargs...)
    return x, k + k2
end

# ============================================================================
# PSI1 Algorithm (from new_test_terminal.jl)
# ============================================================================

function PSI1(xˡ, funcs)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs

    r⃰ = -X'r(xˡ)

    nonzero_indices = findall(!iszero, xˡ)
    zero_indices = findall(iszero, xˡ)

    for i in nonzero_indices
        jₘₐₓ = 0
        v⃰ₘₐₓ = 0.0

        @inbounds for j in zero_indices
            v⃰ = proxl0(r⃰[j] + XTX[i, j] * xˡ[i])

            if abs(v⃰) > abs(v⃰ₘₐₓ)
                v⃰ₘₐₓ = v⃰
                jₘₐₓ = j
            end
        end

        if abs(v⃰ₘₐₓ) > abs(xˡ[i])
            xˡ[i] = 0.0
            xˡ[jₘₐₓ] = v⃰ₘₐₓ

            return xˡ, false
        end
    end

    return xˡ, true
end

# ============================================================================
# Solver with PSI1
# ============================================================================

function SolverPSI1(solver, x⁰, funcs; γₖ=0.0, max_psi1_fails=50, kwargs...)
    β = x⁰
    kᵢ = kₒ = 0
    isPSI1 = false

    seen_supports = Set{UInt64}()
    consecutive_fails = 0

    while !isPSI1 && kₒ < kₘₐₓ
        kₒ += 1
        β, k = solver(β, funcs; γₖ=γₖ, kwargs...)
        kᵢ += k
        γₖ = 0.0  # Only the first call uses the pre-computed γₖ
        β, isPSI1 = PSI1(β, funcs)

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
# Main Experiment
# ============================================================================

# Format SNR without trailing .0 for integer values
format_snr(x) = isinteger(x) ? string(Int(x)) : string(x)

function main()
    config_name = "$(corr)-$(ρ)-$(n)-$(p)-$(format_snr(SNR))-$(k⃰)"
    outdir = "graphs/Boxplot/$(mode)/$(config_name)"
    mkpath(outdir)

    @info "Output directory: $outdir"

    if mode == "nocpsi1"
        # ================================================================
        # Section 5.2: Pure algorithms (no CPSI1)
        # Algorithms: PGCCD, NSPG, NSPGH, VMNSPG
        # Metrics: F, SUP (support similarity), K (iterations)
        # ================================================================
        algo_names = [L"PGCCD" L"NSPG" L"NSPGH" L"VMNSPG"]
        n_algos = 4

        Fhist = zeros(T, n_algos)
        SUPhist = zeros(T, n_algos)
        Khist = zeros(T, n_algos)

        @info "Running NO-CPSI1 experiment" config=config_name algorithms=algo_names trials=T
        total_start = time()

        for t = 1:T
            # Generate fresh data for each trial
            X, y, XTX, p_local, β⃰_local, k⃰_local = generate_data(corr=corr, ρ=ρ, n=n, p=p, SNR=SNR, k⃰=k⃰)

            # Fixed λ₀ = 0.5 * ‖X'y‖²∞ / (2 * Lf)
            Lf = eigmax(X * X')
            λ₀ = 0.5 * norm(X'y, Inf)^2 / (2 * Lf)
            fns = make_funcs(X, y, XTX, λ₀)

            r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, _, _ = fns
            suppsim(β) = count(i -> !iszero(β⃰_local[i]) && !iszero(β[i]), 1:p_local) / max(k⃰_local, norm(β, 0))

            # Random sparse initialization
            x⁰ = zeros(Float64, p_local)
            supp = randperm(p_local)[1:k⃰_local]
            x⁰[supp] = rand(Uniform(0, 1), k⃰_local)

            # 1. PGCCD
            β, k = CDSS(copy(x⁰), fns)
            SUP = suppsim(β); Fval = F(r(β), β)
            SUPhist[t, 1] = SUP; Fhist[t, 1] = Fval; Khist[t, 1] = k
            @show "PGCCD" SUP Fval k

            # 2. NSPG
            β, k = SPG(copy(x⁰), fns)
            SUP = suppsim(β); Fval = F(r(β), β)
            SUPhist[t, 2] = SUP; Fhist[t, 2] = Fval; Khist[t, 2] = k
            @show "NSPG" SUP Fval k

            # 3. NSPGH
            β, k = SPGH(copy(x⁰), fns)
            SUP = suppsim(β); Fval = F(r(β), β)
            SUPhist[t, 3] = SUP; Fhist[t, 3] = Fval; Khist[t, 3] = k
            @show "NSPGH" SUP Fval k

            # 4. VMNSPG
            β, k = VMSPG(copy(x⁰), fns)
            SUP = suppsim(β); Fval = F(r(β), β)
            SUPhist[t, 4] = SUP; Fhist[t, 4] = Fval; Khist[t, 4] = k
            @show "VMNSPG" SUP Fval k

            @info "Trial $t/$T completed"
            flush(stdout)
        end

        @info "All trials completed in $(round(time() - total_start, digits=1))s"

        # Print summary table
        println("\n" * "="^60)
        println("RESULTS ($(config_name), mode=nocpsi1, T=$T)")
        println("="^60)
        println("Algorithm | F(x) mean   | SUP mean | K mean")
        println("----------|-------------|----------|-------")
        algo_labels = ["PGCCD", "NSPG", "NSPGH", "VMNSPG"]
        for a = 1:n_algos
            @printf("%-9s | %-11.4f | %-8.4f | %-6.1f\n",
                algo_labels[a],
                mean(Fhist[:, a]), mean(SUPhist[:, a]), mean(Khist[:, a]))
        end
        println("="^60)

        # Generate boxplots
        theme(:seaborn_bright)
        default(lw=2)

        pF = boxplot(algo_names, Fhist, legend=false, ylabel=L"F(x)", dpi=600)
        savefig(pF, joinpath(outdir, "pF-$(config_name).png"))

        pSUP = boxplot(algo_names, SUPhist, legend=false,
            ylabel=L"\frac{|S\cap S^\dagger|}{\max\{|S|,k^\dagger\}}", left_margin=5mm, dpi=600)
        savefig(pSUP, joinpath(outdir, "pSUP-$(config_name).png"))

        pK = boxplot(algo_names, Khist, legend=false, ylabel="Iterations", dpi=600)
        savefig(pK, joinpath(outdir, "pK-$(config_name).png"))

    else
        # ================================================================
        # Section 5.3: With CPSI1 wrapper
        # Algorithms: PGCCD+CPSI1, NSPG+CPSI1, NSPG+PGCCD+CPSI1, NSPG (no CPSI1)
        # Metrics: F, SUP, K (inner iters), Kout (outer iters)
        # ================================================================
        algo_names = [L"PGCCD" L"NSPG" L"NSPG+PGCCD" L"NSPG\ (no\ CPSI)"]
        n_algos = 4

        Fhist = zeros(T, n_algos)
        SUPhist = zeros(T, n_algos)
        Khist = zeros(T, n_algos)
        Kouthist = zeros(T, n_algos)

        @info "Running CPSI1 experiment" config=config_name algorithms=algo_names trials=T
        total_start = time()

        for t = 1:T
            X, y, XTX, p_local, β⃰_local, k⃰_local = generate_data(corr=corr, ρ=ρ, n=n, p=p, SNR=SNR, k⃰=k⃰)

            Lf = eigmax(X * X')
            λ₀ = 0.5 * norm(X'y, Inf)^2 / (2 * Lf)
            fns = make_funcs(X, y, XTX, λ₀)

            r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, _, _ = fns
            suppsim(β) = count(i -> !iszero(β⃰_local[i]) && !iszero(β[i]), 1:p_local) / max(k⃰_local, norm(β, 0))

            # Random sparse initialization
            x⁰ = zeros(Float64, p_local)
            supp = randperm(p_local)[1:k⃰_local]
            x⁰[supp] = rand(Uniform(0, 1), k⃰_local)

            # 1. PGCCD+CPSI1
            β, kᵢ, kₒ = SolverPSI1(CDSS, copy(x⁰), fns)
            SUP = suppsim(β); Fval = F(r(β), β)
            SUPhist[t, 1] = SUP; Fhist[t, 1] = Fval; Khist[t, 1] = kᵢ; Kouthist[t, 1] = kₒ
            @show "PGCCD+CPSI1" SUP Fval kᵢ kₒ

            # 2. NSPG+CPSI1
            β, kᵢ, kₒ = SolverPSI1(SPG, copy(x⁰), fns)
            SUP = suppsim(β); Fval = F(r(β), β)
            SUPhist[t, 2] = SUP; Fhist[t, 2] = Fval; Khist[t, 2] = kᵢ; Kouthist[t, 2] = kₒ
            @show "NSPG+CPSI1" SUP Fval kᵢ kₒ

            # 3. NSPG+PGCCD+CPSI1
            β, kᵢ, kₒ = SolverPSI1(SPGpCDSS, copy(x⁰), fns)
            SUP = suppsim(β); Fval = F(r(β), β)
            SUPhist[t, 3] = SUP; Fhist[t, 3] = Fval; Khist[t, 3] = kᵢ; Kouthist[t, 3] = kₒ
            @show "NSPG+PGCCD+CPSI1" SUP Fval kᵢ kₒ

            # 4. NSPG (no CPSI1) — plain SPG
            β, k = SPG(copy(x⁰), fns)
            SUP = suppsim(β); Fval = F(r(β), β)
            SUPhist[t, 4] = SUP; Fhist[t, 4] = Fval; Khist[t, 4] = k; Kouthist[t, 4] = 0
            @show "NSPG (no CPSI)" SUP Fval k

            @info "Trial $t/$T completed"
            flush(stdout)
        end

        @info "All trials completed in $(round(time() - total_start, digits=1))s"

        # Print summary table
        println("\n" * "="^60)
        println("RESULTS ($(config_name), mode=cpsi1, T=$T)")
        println("="^60)
        println("Algorithm       | F(x) mean   | SUP mean | K mean | Kout mean")
        println("----------------|-------------|----------|--------|----------")
        algo_labels = ["PGCCD", "NSPG", "NSPG+PGCCD", "NSPG(noCPSI)"]
        for a = 1:n_algos
            @printf("%-15s | %-11.4f | %-8.4f | %-6.1f | %-6.1f\n",
                algo_labels[a],
                mean(Fhist[:, a]), mean(SUPhist[:, a]), mean(Khist[:, a]), mean(Kouthist[:, a]))
        end
        println("="^60)

        # Generate boxplots
        theme(:seaborn_bright)
        default(lw=2)

        pF = boxplot(algo_names, Fhist, legend=false, ylabel=L"F(x)", dpi=600)
        savefig(pF, joinpath(outdir, "pF-$(config_name).png"))

        pSUP = boxplot(algo_names, SUPhist, legend=false,
            ylabel=L"\frac{|S\cap S^\dagger|}{\max\{|S|,k^\dagger\}}", left_margin=5mm, dpi=600)
        savefig(pSUP, joinpath(outdir, "pSUP-$(config_name).png"))

        pK = boxplot(algo_names, Khist, legend=false, ylabel="Iterations", dpi=600)
        savefig(pK, joinpath(outdir, "pK-$(config_name).png"))

        pKout = boxplot(algo_names, Kouthist, legend=false, ylabel="Outer iterations", dpi=600)
        savefig(pKout, joinpath(outdir, "pKout-$(config_name).png"))
    end

    println("\nPlots saved to $outdir")
    flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
