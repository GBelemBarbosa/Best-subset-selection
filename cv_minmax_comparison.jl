# CV Min/Max Comparison Debug Script
# Compares 4 CV strategies: inverse and smart, with and without min/max lambda clamping
# The 0.9 or 0.9^-1 multiplier is OUTSIDE the min/max operation
# Using SPG algorithm with SPG=true

@info "Loading packages..."
flush(stdout)
using Random
using Distributions
using LinearAlgebra
using BenchmarkTools, Profile, TimerOutputs
using Plots, StatsPlots, Plots.PlotMeasures
using LaTeXStrings
using ThreadsX
@info "Packages loaded successfully"
flush(stdout)

# ============================================================================
# Parameters
# ============================================================================
corr = "exp"
ρ = 0.5
p = 2000
SNR = 10.0
k⃰ = 100
n = 1600

kₘₐₓ = 1000
ϵ = 10^-7

@info "Parameters" corr ρ p SNR k⃰ n algo = "SPG" SPG = true
flush(stdout)

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

    function proxl0!(out, x, τ)
        thresh = sqrt(2λ₀ * τ)
        @inbounds @simd for i in eachindex(out)
            out[i] = abs(x[i]) >= thresh ? x[i] : zero(eltype(x))
        end
        return out
    end

    function proxl0VM!(out, x, Uₖ)
        @inbounds for i in eachindex(out)
            out[i] = abs(x[i]) >= sqrt(2λ₀ * Uₖ[i]) ? x[i] : zero(eltype(x))
        end
        return out
    end

    return r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX
end

# ============================================================================
# SPG Algorithm
# ============================================================================

function SPG(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64), γₖ=0.0, kwargs...)
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
        # Initial SPG step via BB1: γₖ = ⟨sᵏ,sᵏ⟩/⟨yᵏ,sᵏ⟩
        XTX∇fxᵏ = XTX * ∇fxᵏ
        γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
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
# PSI1 Algorithm
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
        γₖ = 0.0  # Only first call uses pre-computed γₖ
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
# Inverse Cross Validation - WITH min/max clamping
# The 0.9^-1 multiplier is OUTSIDE the min/max
# ============================================================================

function inverse_cv_with_minmax(solver, vars; λ_max_ratio=1e30, SPG=true, stagnation_handling=true)
    X, y, yval, XTX, p, β⃰, k⃰ = vars
    suppsim(β) = count(i -> !iszero(β⃰[i]) && !iszero(β[i]), 1:p) / max(k⃰, norm(β, 0))
    predval(β) = norm(X * β .- yval)^2 / norm(yval)^2

    β = β_best = zeros(p)
    best = best_λ = Inf

    ∇fxᵏ = -X'y
    correlations = abs.(∇fxᵏ)
    min_corr = ThreadsX.minimum(correlations)

    if SPG
        XTX∇fxᵏ = X' * X * ∇fxᵏ
        γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
    else
        γₖ = 1.0
    end

    λ = (1.01^-1) * γₖ * min_corr^2 / 2
    λ_max = λ * λ_max_ratio

    i = 1
    stagnant_count = 0
    prev_λ = λ

    while λ < λ_max && (!iszero(norm(β, 0)) || i == 1)
        λ = max(λ, eps())

        β, kᵢ, kₒ = solver(β, funcs(X, y, yval, XTX, p, β⃰, k⃰, λ); γₖ=SPG ? γₖ : 0.0)

        mse = norm(yval .- X * β)
        if mse < best
            β_best = copy(β)
            best = mse
            best_λ = λ
        end

        if norm(β, 0) == 0
            break
        end

        if SPG
            ∇fxᵏ = X' * (X * β - y)
            XTX∇fxᵏ = X' * X * ∇fxᵏ
            γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
        end

        # Compute new λ with min/max clamping to previous λ
        # Formula: new_λ = 0.9^-1 * max(previous_λ, raw_λ)
        raw_λ = ThreadsX.minimum(abs(β[j] - γₖ * dot(view(X, :, j), X * β - y)) for j = 1:p if !iszero(β[j]))^2 / (2 * γₖ)
        computed_λ = (0.9^-1) * max(prev_λ, raw_λ)  # min/max: take max with previous λ, THEN multiply

        if stagnation_handling
            if abs(computed_λ - prev_λ) / max(prev_λ, eps()) < 0.01
                stagnant_count += 1
                new_λ = computed_λ < λ ? λ : computed_λ
                if stagnant_count >= 3
                    new_λ = 1.1 * λ
                    stagnant_count = 0
                end
            else
                stagnant_count = 0
                new_λ = computed_λ
            end
            prev_λ = λ
            λ = new_λ
        else
            prev_λ = λ
            λ = computed_λ
        end
        i += 1
    end

    β_best, kᵢ, kₒ = solver(β_best, funcs(X, y, yval, XTX, p, β⃰, k⃰, best_λ); γₖ=SPG ? γₖ : 0.0)
    β_best_0, kᵢ, kₒ = solver(zeros(p), funcs(X, y, yval, XTX, p, β⃰, k⃰, best_λ); γₖ=SPG ? γₖ : 0.0)
    β_best = predval(β_best) > predval(β_best_0) ? β_best_0 : β_best
    return β_best, best_λ, norm(β_best, 0), predval(β_best), suppsim(β_best), norm(β_best - β⃰, Inf)
end

# ============================================================================
# Inverse Cross Validation - WITHOUT min/max clamping
# ============================================================================

function inverse_cv_without_minmax(solver, vars; λ_max_ratio=1e30, SPG=true, stagnation_handling=true)
    X, y, yval, XTX, p, β⃰, k⃰ = vars
    suppsim(β) = count(i -> !iszero(β⃰[i]) && !iszero(β[i]), 1:p) / max(k⃰, norm(β, 0))
    predval(β) = norm(X * β .- yval)^2 / norm(yval)^2

    β = β_best = zeros(p)
    best = best_λ = Inf

    ∇fxᵏ = -X'y
    correlations = abs.(∇fxᵏ)
    min_corr = ThreadsX.minimum(correlations)

    if SPG
        XTX∇fxᵏ = X' * X * ∇fxᵏ
        γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
    else
        γₖ = 1.0
    end

    λ = (1.01^-1) * γₖ * min_corr^2 / 2
    λ_max = λ * λ_max_ratio

    i = 1
    stagnant_count = 0
    prev_λ = λ

    while λ < λ_max && (!iszero(norm(β, 0)) || i == 1)
        λ = max(λ, eps())

        β, kᵢ, kₒ = solver(β, funcs(X, y, yval, XTX, p, β⃰, k⃰, λ); γₖ=SPG ? γₖ : 0.0)

        mse = norm(yval .- X * β)
        if mse < best
            β_best = copy(β)
            best = mse
            best_λ = λ
        end

        if norm(β, 0) == 0
            break
        end

        if SPG
            ∇fxᵏ = X' * (X * β - y)
            XTX∇fxᵏ = X' * X * ∇fxᵏ
            γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
        end

        # Compute new λ WITHOUT min/max clamping - just use the raw formula
        raw_λ = ThreadsX.minimum(abs(β[j] - γₖ * dot(view(X, :, j), X * β - y)) for j = 1:p if !iszero(β[j]))^2 / (2 * γₖ)
        computed_λ = (0.9^-1) * raw_λ  # No min/max, just multiply

        if stagnation_handling
            if abs(computed_λ - prev_λ) / max(prev_λ, eps()) < 0.01
                stagnant_count += 1
                new_λ = computed_λ < λ ? λ : computed_λ
                if stagnant_count >= 3
                    new_λ = 1.1 * λ
                    stagnant_count = 0
                end
            else
                stagnant_count = 0
                new_λ = computed_λ
            end
            prev_λ = λ
            λ = new_λ
        else
            prev_λ = λ
            λ = computed_λ
        end
        i += 1
    end

    β_best, kᵢ, kₒ = solver(β_best, funcs(X, y, yval, XTX, p, β⃰, k⃰, best_λ); γₖ=SPG ? γₖ : 0.0)
    β_best_0, kᵢ, kₒ = solver(zeros(p), funcs(X, y, yval, XTX, p, β⃰, k⃰, best_λ); γₖ=SPG ? γₖ : 0.0)
    β_best = predval(β_best) > predval(β_best_0) ? β_best_0 : β_best
    return β_best, best_λ, norm(β_best, 0), predval(β_best), suppsim(β_best), norm(β_best - β⃰, Inf)
end

# ============================================================================
# Smart Adaptive Cross Validation - WITH min/max clamping
# The 0.9 or 0.9^-1 multiplier is OUTSIDE the min/max
# ============================================================================

function smart_cv_with_minmax(solver, vars; λ_min_ratio=10^-5, λ_max_ratio=floatmax(), SPG=true, stagnation_handling=true)
    X, y, yval, XTX, p, β⃰, k⃰ = vars
    suppsim(β) = count(i -> !iszero(β⃰[i]) && !iszero(β[i]), 1:p) / max(k⃰, norm(β, 0))
    predval(β) = norm(X * β .- yval)^2 / norm(yval)^2
    calc_mse(β) = norm(yval .- X * β)

    # --- 1. PROBE PHASE ---
    ∇fxᵏ = -X'y
    correlations = abs.(∇fxᵏ)
    max_corr = ThreadsX.maximum(correlations)
    min_corr = ThreadsX.minimum(correlations)

    if SPG
        XTX∇fxᵏ = X' * X * ∇fxᵏ
        γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
    else
        γₖ = 1.0
    end

    λ_low = (1.01^-1) * γₖ * min_corr^2 / 2
    λ_high = 1.01 * γₖ * max_corr^2 / 2

    β_low, _, _ = solver(zeros(p), funcs(X, y, yval, XTX, p, β⃰, k⃰, λ_low); γₖ=SPG ? γₖ : 0.0)
    mse_low = calc_mse(β_low)

    β_high, _, _ = solver(zeros(p), funcs(X, y, yval, XTX, p, β⃰, k⃰, λ_high); γₖ=SPG ? γₖ : 0.0)
    mse_high = calc_mse(β_high)

    if mse_low < mse_high
        current_λ = λ_low
        current_β = β_low
        current_mse = mse_low
        direction = :increase
        λ_limit = λ_high * λ_max_ratio
    else
        current_λ = λ_high
        current_β = β_high
        current_mse = mse_high
        direction = :decrease
        λ_limit = λ_low * λ_min_ratio
    end

    best_mse = current_mse
    best_λ = current_λ
    best_β = copy(current_β)

    stagnant_count = 0
    prev_computed_λ = current_λ

    # --- 2. DESCENT PHASE ---
    i = 2
    direction_flip_count = 0

    while true
        if SPG
            ∇fxᵏ = X' * (X * current_β - y)
            XTX∇fxᵏ = X' * X * ∇fxᵏ
            γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
        end

        # A. Propose Next Lambda WITH min/max clamping
        if direction == :increase
            SUP = Int(norm(current_β, 0))
            rᵏ = X * current_β - y
            val = SUP == 0 ? 0.0 : ThreadsX.minimum(abs(current_β[j] - γₖ * dot(view(X, :, j), rᵏ)) for j = 1:p if !iszero(current_β[j]))
            raw_λ = val^2 / (2 * γₖ)
            # min/max: max with previous λ, THEN multiply by 0.9^-1
            computed_λ = (0.9^-1) * max(current_λ, raw_λ)
        else # :decrease
            SUP = Int(norm(current_β, 0))
            rᵏ = X * current_β - y
            val = SUP == p ? 0.0 : ThreadsX.maximum(abs(dot(view(X, :, j), rᵏ)) for j = 1:p if iszero(current_β[j]))
            raw_λ = γₖ * val^2 / 2
            # min/max: min with previous λ, THEN multiply by 0.9
            computed_λ = 0.9 * min(current_λ, raw_λ)
        end

        # Stagnation Handling
        if stagnation_handling
            if abs(computed_λ - prev_computed_λ) / max(prev_computed_λ, eps()) < 0.01
                stagnant_count += 1

                if direction == :increase
                    computed_λ = computed_λ < current_λ ? current_λ : computed_λ
                    if stagnant_count >= 3
                        computed_λ = current_λ * 1.1
                        stagnant_count = 0
                    end
                else
                    computed_λ = computed_λ > current_λ ? current_λ : computed_λ
                    if stagnant_count >= 3
                        computed_λ = current_λ * 0.9
                        stagnant_count = 0
                    end
                end
            else
                stagnant_count = 0
            end
        end
        prev_computed_λ = computed_λ

        # Safety clamps
        if direction == :increase && computed_λ > λ_limit
            break
        end
        if direction == :decrease && computed_λ < λ_limit
            break
        end

        # B. Probe Next Step
        probe_λ = computed_λ
        probe_β, _, _ = solver(current_β, funcs(X, y, yval, XTX, p, β⃰, k⃰, probe_λ); γₖ=SPG ? γₖ : 0.0)
        probe_mse = calc_mse(probe_β)

        # C. Check Improvement
        min_improvement = 0.01
        if probe_mse >= current_mse
            alt_direction = (direction == :increase) ? :decrease : :increase

            if alt_direction == :increase
                SUP = Int(norm(current_β, 0))
                rᵏ = X * current_β - y
                val = SUP == 0 ? 0.0 : ThreadsX.minimum(abs(current_β[j] - γₖ * dot(view(X, :, j), rᵏ)) for j = 1:p if !iszero(current_β[j]))
                raw_λ = val^2 / (2 * γₖ)
                alt_λ = (0.9^-1) * max(current_λ, raw_λ)
            else
                SUP = Int(norm(current_β, 0))
                rᵏ = X * current_β - y
                val = SUP == p ? 0.0 : ThreadsX.maximum(abs(dot(view(X, :, j), rᵏ)) for j = 1:p if iszero(current_β[j]))
                raw_λ = γₖ * val^2 / 2
                alt_λ = 0.9 * min(current_λ, raw_λ)
            end

            alt_β, _, _ = solver(current_β, funcs(X, y, yval, XTX, p, β⃰, k⃰, alt_λ); γₖ=SPG ? γₖ : 0.0)
            alt_mse = calc_mse(alt_β)

            if alt_mse < probe_mse * (1 - min_improvement)
                probe_λ = alt_λ
                probe_β = alt_β
                probe_mse = alt_mse
                direction = alt_direction

                direction_flip_count += 1
                if direction_flip_count >= 2
                    break
                end
                λ_limit = (direction == :increase) ? λ_high * λ_max_ratio : λ_low * λ_min_ratio

                prev_computed_λ = probe_λ
            end
        end

        current_λ = probe_λ
        current_β = probe_β
        current_mse = probe_mse
        if current_mse < best_mse
            best_mse = current_mse
            best_λ = current_λ
            best_β = copy(current_β)
        end

        if norm(current_β, 0) == 0
            break
        end

        i += 1
    end

    β_best, kᵢ, kₒ = solver(best_β, funcs(X, y, yval, XTX, p, β⃰, k⃰, best_λ); γₖ=SPG ? γₖ : 0.0)
    β_best_0, kᵢ, kₒ = solver(zeros(p), funcs(X, y, yval, XTX, p, β⃰, k⃰, best_λ); γₖ=SPG ? γₖ : 0.0)
    best_β = predval(β_best) > predval(β_best_0) ? β_best_0 : β_best

    return best_β, best_λ, norm(best_β, 0), predval(best_β), suppsim(best_β), norm(best_β - β⃰, Inf)
end

# ============================================================================
# Smart Adaptive Cross Validation - WITHOUT min/max clamping
# ============================================================================

function smart_cv_without_minmax(solver, vars; λ_min_ratio=10^-5, λ_max_ratio=floatmax(), SPG=true, stagnation_handling=true)
    X, y, yval, XTX, p, β⃰, k⃰ = vars
    suppsim(β) = count(i -> !iszero(β⃰[i]) && !iszero(β[i]), 1:p) / max(k⃰, norm(β, 0))
    predval(β) = norm(X * β .- yval)^2 / norm(yval)^2
    calc_mse(β) = norm(yval .- X * β)

    # --- 1. PROBE PHASE ---
    ∇fxᵏ = -X'y
    correlations = abs.(∇fxᵏ)
    max_corr = ThreadsX.maximum(correlations)
    min_corr = ThreadsX.minimum(correlations)

    if SPG
        XTX∇fxᵏ = X' * X * ∇fxᵏ
        γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
    else
        γₖ = 1.0
    end

    λ_low = (1.01^-1) * γₖ * min_corr^2 / 2
    λ_high = 1.01 * γₖ * max_corr^2 / 2

    β_low, _, _ = solver(zeros(p), funcs(X, y, yval, XTX, p, β⃰, k⃰, λ_low); γₖ=SPG ? γₖ : 0.0)
    mse_low = calc_mse(β_low)

    β_high, _, _ = solver(zeros(p), funcs(X, y, yval, XTX, p, β⃰, k⃰, λ_high); γₖ=SPG ? γₖ : 0.0)
    mse_high = calc_mse(β_high)

    if mse_low < mse_high
        current_λ = λ_low
        current_β = β_low
        current_mse = mse_low
        direction = :increase
        λ_limit = λ_high * λ_max_ratio
    else
        current_λ = λ_high
        current_β = β_high
        current_mse = mse_high
        direction = :decrease
        λ_limit = λ_low * λ_min_ratio
    end

    best_mse = current_mse
    best_λ = current_λ
    best_β = copy(current_β)

    stagnant_count = 0
    prev_computed_λ = current_λ

    # --- 2. DESCENT PHASE ---
    i = 2
    direction_flip_count = 0

    while true
        if SPG
            ∇fxᵏ = X' * (X * current_β - y)
            XTX∇fxᵏ = X' * X * ∇fxᵏ
            γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
        end

        # A. Propose Next Lambda WITHOUT min/max clamping
        if direction == :increase
            SUP = Int(norm(current_β, 0))
            rᵏ = X * current_β - y
            val = SUP == 0 ? 0.0 : ThreadsX.minimum(abs(current_β[j] - γₖ * dot(view(X, :, j), rᵏ)) for j = 1:p if !iszero(current_β[j]))
            # No min/max, just the formula with multiplier
            computed_λ = (0.9^-1) * val^2 / (2 * γₖ)
        else # :decrease
            SUP = Int(norm(current_β, 0))
            rᵏ = X * current_β - y
            val = SUP == p ? 0.0 : ThreadsX.maximum(abs(dot(view(X, :, j), rᵏ)) for j = 1:p if iszero(current_β[j]))
            # No min/max, just the formula with multiplier
            computed_λ = 0.9 * γₖ * val^2 / 2
        end

        # Stagnation Handling
        if stagnation_handling
            if abs(computed_λ - prev_computed_λ) / max(prev_computed_λ, eps()) < 0.01
                stagnant_count += 1

                if direction == :increase
                    computed_λ = computed_λ < current_λ ? current_λ : computed_λ
                    if stagnant_count >= 3
                        computed_λ = current_λ * 1.1
                        stagnant_count = 0
                    end
                else
                    computed_λ = computed_λ > current_λ ? current_λ : computed_λ
                    if stagnant_count >= 3
                        computed_λ = current_λ * 0.9
                        stagnant_count = 0
                    end
                end
            else
                stagnant_count = 0
            end
        end
        prev_computed_λ = computed_λ

        # Safety clamps
        if direction == :increase && computed_λ > λ_limit
            break
        end
        if direction == :decrease && computed_λ < λ_limit
            break
        end

        # B. Probe Next Step
        probe_λ = computed_λ
        probe_β, _, _ = solver(current_β, funcs(X, y, yval, XTX, p, β⃰, k⃰, probe_λ); γₖ=SPG ? γₖ : 0.0)
        probe_mse = calc_mse(probe_β)

        # C. Check Improvement
        min_improvement = 0.01
        if probe_mse >= current_mse
            alt_direction = (direction == :increase) ? :decrease : :increase

            if alt_direction == :increase
                SUP = Int(norm(current_β, 0))
                rᵏ = X * current_β - y
                val = SUP == 0 ? 0.0 : ThreadsX.minimum(abs(current_β[j] - γₖ * dot(view(X, :, j), rᵏ)) for j = 1:p if !iszero(current_β[j]))
                alt_λ = (0.9^-1) * val^2 / (2 * γₖ)
            else
                SUP = Int(norm(current_β, 0))
                rᵏ = X * current_β - y
                val = SUP == p ? 0.0 : ThreadsX.maximum(abs(dot(view(X, :, j), rᵏ)) for j = 1:p if iszero(current_β[j]))
                alt_λ = 0.9 * γₖ * val^2 / 2
            end

            alt_β, _, _ = solver(current_β, funcs(X, y, yval, XTX, p, β⃰, k⃰, alt_λ); γₖ=SPG ? γₖ : 0.0)
            alt_mse = calc_mse(alt_β)

            if alt_mse < probe_mse * (1 - min_improvement)
                probe_λ = alt_λ
                probe_β = alt_β
                probe_mse = alt_mse
                direction = alt_direction

                direction_flip_count += 1
                if direction_flip_count >= 2
                    break
                end
                λ_limit = (direction == :increase) ? λ_high * λ_max_ratio : λ_low * λ_min_ratio

                prev_computed_λ = probe_λ
            end
        end

        current_λ = probe_λ
        current_β = probe_β
        current_mse = probe_mse
        if current_mse < best_mse
            best_mse = current_mse
            best_λ = current_λ
            best_β = copy(current_β)
        end

        if norm(current_β, 0) == 0
            break
        end

        i += 1
    end

    β_best, kᵢ, kₒ = solver(best_β, funcs(X, y, yval, XTX, p, β⃰, k⃰, best_λ); γₖ=SPG ? γₖ : 0.0)
    β_best_0, kᵢ, kₒ = solver(zeros(p), funcs(X, y, yval, XTX, p, β⃰, k⃰, best_λ); γₖ=SPG ? γₖ : 0.0)
    best_β = predval(β_best) > predval(β_best_0) ? β_best_0 : β_best

    return best_β, best_λ, norm(best_β, 0), predval(best_β), suppsim(best_β), norm(best_β - β⃰, Inf)
end

# ============================================================================
# Main Comparison Experiment
# ============================================================================

function main()
    T = 5  # 5 runs

    strategies = [
        "Inverse + MinMax",
        "Inverse - MinMax",
        "Smart + MinMax",
        "Smart - MinMax"
    ]

    cv_funcs = [
        inverse_cv_with_minmax,
        inverse_cv_without_minmax,
        smart_cv_with_minmax,
        smart_cv_without_minmax
    ]

    @info "Experiment configuration" n = n p = p corr = corr ρ = ρ SNR = SNR k⃰ = k⃰ trials = T algo = "SPG" SPG_mode = true strategies
    flush(stdout)

    # Storage: times and similarities for each strategy across trials
    times = zeros(T, 4)
    similarities = zeros(T, 4)
    SUPs = zeros(T, 4)
    Preds = zeros(T, 4)

    println("\n" * "="^80)
    println("CV MIN/MAX COMPARISON - $T RUNS (using SPG with SPG=true)")
    println("Setup: corr=$corr, ρ=$ρ, p=$p, SNR=$SNR, k⃰=$k⃰, n=$n")
    println("="^80 * "\n")

    for trial = 1:T
        println("\n--- Trial $trial/$T ---")
        flush(stdout)

        vars = variables(corr=corr, ρ=ρ, n=n, p=p, SNR=SNR, k⃰=k⃰)

        for (s, (name, cv_func)) in enumerate(zip(strategies, cv_funcs))
            trial_start = time()
            β, _, SUP, Pred, Sim, Infv = cv_func(
                (x, f; kwargs...) -> SolverPSI1(SPG, x, f; kwargs...),  # Using SPG algorithm
                vars;
                SPG=true  # SPG mode enabled
            )
            elapsed = time() - trial_start

            times[trial, s] = elapsed
            similarities[trial, s] = Sim
            SUPs[trial, s] = SUP
            Preds[trial, s] = Pred

            println("  $name: time=$(round(elapsed, digits=2))s, Sim=$(round(Sim, digits=3)), SUP=$SUP, Pred=$(round(Pred, digits=4))")
            flush(stdout)
        end
    end

    # Summary statistics
    println("\n" * "="^80)
    println("SUMMARY STATISTICS")
    println("="^80)

    println("\nAverage Times (seconds):")
    for (s, name) in enumerate(strategies)
        avg_time = sum(times[:, s]) / T
        std_time = sqrt(sum((times[:, s] .- avg_time) .^ 2) / T)
        println("  $name: $(round(avg_time, digits=2)) ± $(round(std_time, digits=2))")
    end

    println("\nAverage Similarities:")
    for (s, name) in enumerate(strategies)
        avg_sim = sum(similarities[:, s]) / T
        std_sim = sqrt(sum((similarities[:, s] .- avg_sim) .^ 2) / T)
        println("  $name: $(round(avg_sim, digits=3)) ± $(round(std_sim, digits=3))")
    end

    println("\nAverage Support (SUP):")
    for (s, name) in enumerate(strategies)
        avg_sup = sum(SUPs[:, s]) / T
        println("  $name: $(round(avg_sup, digits=1))")
    end

    println("\nAverage Prediction Error:")
    for (s, name) in enumerate(strategies)
        avg_pred = sum(Preds[:, s]) / T
        println("  $name: $(round(avg_pred, digits=4))")
    end

    println("\n" * "="^80)
    println("DETAILED RESULTS (per trial)")
    println("="^80)

    println("\nTimes:")
    println("Trial | " * join([rpad(s, 18) for s in strategies], " | "))
    println("-"^90)
    for t = 1:T
        row = "$t     | " * join([rpad(round(times[t, s], digits=2), 18) for s = 1:4], " | ")
        println(row)
    end

    println("\nSimilarities:")
    println("Trial | " * join([rpad(s, 18) for s in strategies], " | "))
    println("-"^90)
    for t = 1:T
        row = "$t     | " * join([rpad(round(similarities[t, s], digits=3), 18) for s = 1:4], " | ")
        println(row)
    end

    return times, similarities, SUPs, Preds
end

times, similarities, SUPs, Preds = main()
