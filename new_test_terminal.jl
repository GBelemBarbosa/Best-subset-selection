# Terminal-running version of new_test.pluto.jl
# Converted from Pluto notebook to standard Julia script

@info "Loading packages..."
using Random
using Distributions
using LinearAlgebra
using BenchmarkTools, Profile, TimerOutputs
using Plots, StatsPlots, Plots.PlotMeasures
using LaTeXStrings
using ThreadsX
@info "Packages loaded successfully"

# ============================================================================
# Parameters
# ============================================================================

corr = "exp"
ρ = 0.9
p = 1000
SNR = 5
k⃰ = 20

@info "Parameters" corr ρ p SNR k⃰

kₘₐₓ = 1000
ϵ = 10^-7

# ============================================================================
# Variable Generation
# ============================================================================

function variables(; corr="exp", ρ=0.9, n=250, p=1000, SNR=5, k⃰=20)
    Σ = corr == "exp" ? [ρ^abs(i - j) for i = 1:p, j = 1:p] : [1 - (1 - ρ) * (i != j) for i = 1:p, j = 1:p]
    d = MvNormal(zeros(p), Σ)
    X = rand(d, n)'
    for i = 1:p
        X[:, i] /= norm(view(X, :, i))
    end
    β⃰ = [1.0 * (mod(i, Int(p / k⃰)) == 0) for i = 1:p]

    σ = sqrt(norm(X * β⃰)^2 / (n * SNR))

    y = X * β⃰ + randn(n) * σ
    yval = X * β⃰ + randn(n) * σ

    XTX = X'X

    return X, y, yval, XTX, p, β⃰, k⃰
end

# ============================================================================
# Function Definitions
# ============================================================================

function funcs(X, y, yval, XTX, p, β⃰, k⃰, λ₀)
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
# VMSPG Algorithm
# ============================================================================

function VMSPG(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64), µ=10^-3)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs

    n_vars = length(x⁰)

    # Pre-allocate all work arrays
    xᵏ = copy(x⁰)
    xᵏ⁻¹ = copy(x⁰)
    sᵏ = similar(x⁰)
    yᵏ = similar(x⁰)
    ∇fxᵏ = similar(x⁰)
    ∇fxᵏ⁻¹ = similar(x⁰)
    temp = similar(x⁰)
    Uₖ = similar(x⁰)
    Uₖ₋₁ = similar(x⁰)

    rᵏ = r(xᵏ)
    ∇f!(∇fxᵏ, rᵏ)
    Fxᵏ⁻¹ = Fxᵏ = F(rᵏ, xᵏ)

    # Initial step computation
    temp .= x⁰ .- ∇fxᵏ .* 10^-5
    r!(rᵏ, temp)
    yᵏ .= ∇fxᵏ .- ∇f(rᵏ)
    yᵏTsᵏ = dot(yᵏ, ∇fxᵏ)
    nsᵏ = γₖ¹ = dot(∇fxᵏ, ∇fxᵏ) * 10^-5 / yᵏTsᵏ
    γₖ² = yᵏTsᵏ * 10^-5 / dot(yᵏ, yᵏ)

    # Initialize Uₖ in-place
    @inbounds for i in eachindex(Uₖ)
        val = 10^-5 * ∇fxᵏ[i] * ∇fxᵏ[i] / (∇fxᵏ[i] * yᵏ[i])
        Uₖ[i] = min(max(val, γₖ²), γₖ¹)
    end
    copyto!(Uₖ₋₁, Uₖ)

    lastₘ = fill(Fxᵏ, m)

    for k = 1:kₘₐₓ
        Fxₗ₍ₖ₎ = maximum(lastₘ)

        while true
            # xᵏ = proxl0VM(xᵏ⁻¹ - Uₖ .* ∇fxᵏ, Uₖ)
            temp .= xᵏ⁻¹ .- Uₖ .* ∇fxᵏ
            proxl0VM!(xᵏ, temp, Uₖ)
            r!(rᵏ, xᵏ)
            Fxᵏ = F(rᵏ, xᵏ)
            sᵏ .= xᵏ .- xᵏ⁻¹

            # Compute dot(sᵏ, sᵏ ./ Uₖ) without allocation
            ss_div_U = zero(eltype(sᵏ))
            @inbounds @simd for i in eachindex(sᵏ)
                ss_div_U += sᵏ[i] * sᵏ[i] / Uₖ[i]
            end

            if Fxᵏ + δ * ss_div_U / 2 <= Fxₗ₍ₖ₎
                break
            end

            BLAS.scal!(τ, Uₖ)

            # Check bounds without allocation
            should_break = false
            @inbounds for i in eachindex(Uₖ)
                if isnan(Uₖ[i]) || Uₖ[i] < γₘᵢₙ
                    should_break = true
                    break
                end
            end
            if should_break
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

        # Update Uₖ in-place with buffer swap
        Uₖ₋₁, Uₖ = Uₖ, Uₖ₋₁
        @inbounds @simd for i in eachindex(Uₖ)
            val = (sᵏ[i] * sᵏ[i] + µ * Uₖ₋₁[i]) / (sᵏ[i] * yᵏ[i] + µ)
            Uₖ[i] = min(max(val, γₖ²), γₖ¹)
        end
    end

    return xᵏ, kₘₐₓ
end

# ============================================================================
# SPGH Algorithm
# ============================================================================

function SPGH(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64))
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs

    # Pre-allocate all work arrays
    xᵏ = copy(x⁰)
    xᵏ⁻¹ = copy(x⁰)
    sᵏ = similar(x⁰)
    yᵏ = similar(x⁰)
    ∇fxᵏ = similar(x⁰)
    ∇fxᵏ⁻¹ = similar(x⁰)
    temp = similar(x⁰)

    rᵏ = r(xᵏ)
    ∇f!(∇fxᵏ, rᵏ)
    Fxᵏ⁻¹ = Fxᵏ = F(rᵏ, xᵏ)

    # Initial step computation
    temp .= x⁰ .- ∇fxᵏ .* 10^-5
    r!(rᵏ, temp)
    yᵏ .= ∇fxᵏ .- ∇f(rᵏ)
    yᵏTsᵏ = dot(yᵏ, ∇fxᵏ)
    nsᵏ = γₖ¹ = dot(∇fxᵏ, ∇fxᵏ) * 10^-5 / yᵏTsᵏ
    γₖ² = yᵏTsᵏ * 10^-5 / dot(yᵏ, yᵏ)
    γₖ = γₖ¹ < 2 * γₖ² ? γₖ² : γₖ¹ - γₖ² / 2
    lastₘ = fill(Fxᵏ, m)

    for k = 1:kₘₐₓ
        Fxₗ₍ₖ₎ = maximum(lastₘ)

        while true
            # xᵏ = proxl0(xᵏ⁻¹ - γₖ * ∇fxᵏ, γₖ)
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
# SPG Algorithm
# ============================================================================

function SPG(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64), γₖ=0.0)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs

    # Pre-allocate all work arrays
    xᵏ = copy(x⁰)
    xᵏ⁻¹ = copy(x⁰)
    sᵏ = similar(x⁰)
    yᵏ = similar(x⁰)
    ∇fxᵏ = similar(x⁰)
    ∇fxᵏ⁻¹ = similar(x⁰)
    temp = similar(x⁰)

    rᵏ = r(xᵏ)
    ∇f!(∇fxᵏ, rᵏ)
    Fxᵏ⁻¹ = Fxᵏ = F(rᵏ, xᵏ)

    if iszero(γₖ)
        temp .= x⁰ .- ∇fxᵏ .* 10^-5
        r!(rᵏ, temp)
        yᵏ .= ∇fxᵏ .- ∇f(rᵏ)
        γₖ = dot(∇fxᵏ, ∇fxᵏ) * 10^-5 / dot(yᵏ, ∇fxᵏ)
    end
    nsᵏ = γₖ
    lastₘ = fill(Fxᵏ, m)

    for k = 1:kₘₐₓ
        Fxₗ₍ₖ₎ = maximum(lastₘ)

        while true
            # xᵏ = proxl0(xᵏ⁻¹ - γₖ * ∇fxᵏ, γₖ)
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
# CDSS Algorithm
# ============================================================================

function CDSS(x⁰, funcs; sortperc=1 / 4)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs

    xᵏ = copy(x⁰)

    rᵏ = -r(xᵏ)

    Fxᵏ⁻¹ = F(rᵏ, xᵏ)

    ksort = round(Int64, length(x⁰) * sortperc)
    greedy = partialsortperm(abs.(∇f(rᵏ)), 1:ksort, rev=true)
    greedy = vcat(greedy, setdiff(1:p, greedy))

    for k = 1:kₘₐₓ
        @inbounds for i in greedy
            xi = proxl0(dot(rᵏ, view(X, :, i)) + xᵏ[i])
            if xi != xᵏ[i]
                BLAS.axpy!(xᵏ[i] - xi, view(X, :, i), rᵏ)
                xᵏ[i] = xi
            end
        end

        Fxᵏ = F(rᵏ, xᵏ)
        if (Fxᵏ⁻¹ - Fxᵏ) / Fxᵏ <= ϵ
            return xᵏ, k
        end
        Fxᵏ⁻¹ = Fxᵏ
    end

    return xᵏ, kₘₐₓ
end

# ============================================================================
# SPG + CDSS Combined
# ============================================================================

function SPGpCDSS(x⁰, funcs)
    x, k = SPG(x⁰, funcs)
    x, k2 = CDSS(x, funcs)

    return x, k + k2
end

# ============================================================================
# PSI1 Algorithm
# ============================================================================

function PSI1(xˡ, funcs)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs

    r⃰ = -X'r(xˡ)

    # Cache index arrays to avoid repeated allocation
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

function SolverPSI1(solver, x⁰, funcs; γₖ=0.0)
    β = x⁰
    kᵢ = kₒ = 0
    isPSI1 = false

    while !isPSI1 && kₒ < kₘₐₓ
        kₒ += 1
        β, k = solver(β, funcs)
        kᵢ += k
        β, isPSI1 = PSI1(β, funcs)
    end

    return β, kᵢ, kₒ
end

# ============================================================================
# Cross Validation
# ============================================================================

function cross_validation(solver, vars; λ_min_ratio=10^-5, SPG=false)
    X, y, yval, XTX, p, β⃰, k⃰ = vars
    suppsim(β) = count(i -> !iszero(β⃰[i]) && !iszero(β[i]), 1:p) / max(k⃰, norm(β, 0))
    predval(β) = norm(X * β .- yval)^2 / norm(yval)^2

    mse_results = Float64[]
    β = β_best = zeros(p)
    best = best_λ = Inf

    ∇fxᵏ = -X'y
    correlations = abs.(∇fxᵏ)

    if SPG
        #γₖ = dot(∇fxᵏ, ∇fxᵏ) * 10^-5 / dot(∇fxᵏ + X' * (y + X * ∇fxᵏ * 10^-5), ∇fxᵏ)
        #λ = 1.01 * γₖ * ThreadsX.maximum(correlations)^2 / 2
        λ = 1.01 * ThreadsX.maximum(correlations)^2 / 2
    else
        λ = 1.01 * ThreadsX.maximum(correlations)^2 / 2
    end
    λ_min = λ * λ_min_ratio

    i = 1
    while λ > λ_min
        # Run solver
        β, kᵢ, kₒ = solver(β, funcs(X, y, yval, XTX, p, β⃰, k⃰, λ))

        push!(mse_results, norm(yval .- X * β))
        if mse_results[end] < best
            β_best = β
            best = mse_results[i]
            best_λ = λ
        end

        #β = zeros(p)

        if SPG
            #∇fxᵏ = X' * (X * β - y)
            #γₖ = dot(∇fxᵏ, ∇fxᵏ) * 10^-5 / dot(∇fxᵏ + X' * (y - X * (β - ∇fxᵏ * 10^-5)), ∇fxᵏ)
            #λ = norm(β, 0) != p ? 0.9 * γₖ * min(λ, ThreadsX.maximum(abs(∇fxᵏ[j]) for j = 1:p if iszero(β[j]))^2 / 2) : 0.0
            λ = norm(β, 0) != p ? 0.9 * min(λ, ThreadsX.maximum(abs(dot(view(X, :, j), X * β - y)) for j = 1:p if iszero(β[j]))^2 / 2) : 0.0
        else
            λ = norm(β, 0) != p ? 0.9 * min(λ, ThreadsX.maximum(abs(dot(view(X, :, j), X * β - y)) for j = 1:p if iszero(β[j]))^2 / 2) : 0.0
        end
        i += 1
    end

    β_best, kᵢ, kₒ = solver(zeros(p), funcs(X, y, yval, XTX, p, β⃰, k⃰, best_λ))

    return β_best, best_λ, norm(β_best, 0), predval(β_best), suppsim(β_best), norm(β_best - β⃰, Inf)
end

# ============================================================================
# Inverse Cross Validation (lambda grows instead of shrinks)
# ============================================================================

function inverse_cross_validation(solver, vars; λ_max_ratio=floatmax(), SPG=false)
    X, y, yval, XTX, p, β⃰, k⃰ = vars
    suppsim(β) = count(i -> !iszero(β⃰[i]) && !iszero(β[i]), 1:p) / max(k⃰, norm(β, 0))
    predval(β) = norm(X * β .- yval)^2 / norm(yval)^2

    mse_results = Float64[]
    β = β_best = zeros(p)
    best = best_λ = Inf

    # Compute initial residual correlations (residual = y when β = 0)
    ∇fxᵏ = -X'y
    correlations = abs.(∇fxᵏ)

    # Initial λ: 1.01^-1 times the smallest nonzero correlation squared / 2
    min_corr = ThreadsX.minimum(correlations)

    if SPG
        γₖ = dot(∇fxᵏ, ∇fxᵏ) * 10^-5 / dot(∇fxᵏ + X' * (y + X * ∇fxᵏ * 10^-5), ∇fxᵏ)
        λ = (1.01^-1) * γₖ * min_corr^2 / 2
        #λ = (1.01^-1) * min_corr^2 / 2
    else
        λ = (1.01^-1) * min_corr^2 / 2
    end

    λ_max = λ * λ_max_ratio

    i = 1
    while λ < λ_max
        # Ensure λ stays positive to avoid sqrt domain errors
        λ = max(λ, eps())

        # Run solver
        β, kᵢ, kₒ = solver(β, funcs(X, y, yval, XTX, p, β⃰, k⃰, λ))

        push!(mse_results, norm(yval .- X * β))
        is_new_best = mse_results[end] < best
        if is_new_best
            β_best = copy(β)
            best = mse_results[i]
            best_λ = λ
        end

        #β = zeros(p)

        # Check for edge cases: all zeros or all nonzero
        if norm(β, 0) == 0
            break  # All components are zero, no nonzero entries to compute minimum from
        end

        if SPG
            ∇fxᵏ = X' * (X * β - y)
            γₖ = dot(∇fxᵏ, ∇fxᵏ) * 10^-5 / dot(∇fxᵏ + X' * (y - X * (β - ∇fxᵏ * 10^-5)), ∇fxᵏ)
            λ = (0.9^-1) * max(λ, ThreadsX.minimum(abs(β[j] - γₖ * ∇fxᵏ[j]) for j = 1:p if !iszero(β[j]))^2 / 2) / γₖ
            #λ = (0.9^-1) * max(λ, ThreadsX.minimum(abs(β[j] - dot(view(X, :, j), X * β - y)) for j = 1:p if !iszero(β[j]))^2 / 2)
        else
            λ = (0.9^-1) * max(λ, ThreadsX.minimum(abs(β[j] - dot(view(X, :, j), X * β - y)) for j = 1:p if !iszero(β[j]))^2 / 2)
        end

        i += 1
    end

    # Final refinement with best λ
    β_best, kᵢ, kₒ = solver(zeros(p), funcs(X, y, yval, XTX, p, β⃰, k⃰, best_λ))

    return β_best, best_λ, norm(β_best, 0), predval(β_best), suppsim(β_best), norm(β_best - β⃰, Inf)
end

# ============================================================================
# Main Experiment
# ============================================================================

function main()
    x⁰ = zeros(Float64, p)
    ns = 100:100:1000
    T = 5

    @info "Experiment configuration" samples = ns trials = T algorithms = ["Greedy CD", "NSPG", "NSPG+CD"]

    Predhist = zeros(length(ns), 3)
    SUPhist = zeros(length(ns), 3)
    Infhist = zeros(length(ns), 3)
    Simhist = zeros(length(ns), 3)

    for (t, n) in enumerate(ns)
        @info "Sample size n=$n" progress = "$(t)/$(length(ns))"
        total_start = time()

        for i = 1:T
            vars = variables(corr=corr, ρ=ρ, n=n, p=p, SNR=SNR, k⃰=k⃰)

            trial_start = time()
            β, best_λ, SUP, Pred, Sim, Infv = cross_validation((x, f) -> SolverPSI1(CDSS, x, f), vars)
            SUPhist[t, 1] += SUP
            Predhist[t, 1] += Pred
            Simhist[t, 1] += Sim
            Infhist[t, 1] += Infv
            @show SUP Pred Sim Infv round(time() - trial_start, digits=1)

            trial_start = time()
            β, best_λ, SUP, Pred, Sim, Infv = cross_validation((x, f) -> SolverPSI1(SPG, x, f), vars, SPG=true)
            SUPhist[t, 2] += SUP
            Predhist[t, 2] += Pred
            Simhist[t, 2] += Sim
            Infhist[t, 2] += Infv
            @show SUP Pred Sim Infv round(time() - trial_start, digits=1)

            trial_start = time()
            β, best_λ, SUP, Pred, Sim, Infv = cross_validation((x, f) -> SolverPSI1(SPGpCDSS, x, f), vars, SPG=true)
            SUPhist[t, 3] += SUP
            Predhist[t, 3] += Pred
            Simhist[t, 3] += Sim
            Infhist[t, 3] += Infv
            @show SUP Pred Sim Infv round(time() - trial_start, digits=1)

            @info "Trial $i/$T completed"
        end

        @info "Completed n=$n in $(round(time() - total_start, digits=1))s"
    end

    @info "All experiments finished"

    SUPhist ./= T
    Predhist ./= T
    Simhist ./= T
    Infhist ./= T

    # Display results
    println("\n" * "="^60)
    println("RESULTS")
    println("="^60)
    println("\nSupport Size History (SUPhist):")
    display(SUPhist)

    # Plot settings
    names = ["Greedy CD" "NSPG" "NSPG+CD"]
    plotname = "$(corr)-$(ρ)-$(p)-$(SNR)-$(k⃰)-$(T)-$(first(ns))_$(step(ns))_$(last(ns))"
    specifics = "_zerolast_not"

    # Create and save plots
    println("\nGenerating plots...")

    pPred = plot(ns, Predhist, labels=names, xlabel=L"n", ylabel=L"\frac{\Vert Ax-b\Vert^2}{\Vert b\Vert^2}", left_margin=5mm, dpi=600)
    savefig(pPred, "pnPred-$(plotname)$(specifics).png")

    pSim = plot(ns, Simhist, labels=names, xlabel=L"n", ylabel=L"\frac{|Supp(x)\cap Supp(x^\dagger)|}{\max\{|Supp(x)|,k^\dagger\}}", left_margin=5mm, dpi=600)
    savefig(pSim, "pnSim-$(plotname)$(specifics).png")

    pInf = plot(ns, Infhist, labels=names, xlabel=L"n", ylabel=L"\Vert x-x^\dagger\Vert_\infty", dpi=600)
    savefig(pInf, "pnInf-$(plotname)$(specifics).png")

    pSUP = plot(ns, SUPhist, labels=names, xlabel=L"n", ylabel=L"\Vert x\Vert_0", dpi=600)
    savefig(pSUP, "pnSUP-$(plotname)$(specifics).png")

    @info "Plots saved" files = ["pnPred-$(plotname)$(specifics).png",
        "pnSim-$(plotname)$(specifics).png",
        "pnInf-$(plotname)$(specifics).png",
        "pnSUP-$(plotname)$(specifics).png"]

    println("\n" * "="^60)
    println("All experiments completed successfully!")
    println("="^60)

    return SUPhist, Predhist, Simhist, Infhist
end

# Run the main function
main()