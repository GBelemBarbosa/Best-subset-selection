# Benchmark script to identify performance bottlenecks in new_test_terminal.jl

using Random
using Distributions
using LinearAlgebra
using BenchmarkTools

# ============================================================================
# Parameters (smaller for fast benchmarking)
# ============================================================================
const corr = "exp"
const ρ = 0.9
const p = 500  # Reduced for faster benchmarking
const SNR = 5
const k⃰_val = 20
const kₘₐₓ = 100  # Reduced iterations
const ϵ = 10^-7

# ============================================================================
# Copy core functions from new_test_terminal.jl
# ============================================================================

function variables(; corr="exp", ρ=0.9, n=250, p=500, SNR=5, k⃰=20)
    Σ = corr == "exp" ? [ρ^abs(i - j) for i = 1:p, j = 1:p] : [1 - (1 - ρ) * (i != j) for i = 1:p, j = 1:p]
    d = MvNormal(zeros(p), Σ)
    X = rand(d, n)'
    for i = 1:p
        X[:, i] /= norm(X[:, i])
    end
    β⃰ = [1.0 * (mod(i, Int(p / k⃰)) == 0) for i = 1:p]
    σ = sqrt(norm(X * β⃰)^2 / (n * SNR))
    y = X * β⃰ + randn(n) * σ
    yval = X * β⃰ + randn(n) * σ
    XTX = X'X
    return X, y, yval, XTX, p, β⃰, k⃰
end

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
    return r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, X, XTX
end

function VMSPG(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64), µ=10^-3)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, X, XTX = funcs
    sᵏ = xᵏ⁻¹ = xᵏ = x⁰
    rᵏ = r(xᵏ)
    ∇fxᵏ⁻¹ = ∇fxᵏ = ∇f(rᵏ)
    Fxᵏ⁻¹ = Fxᵏ = F(rᵏ, xᵏ)
    r!(rᵏ, x⁰ - ∇fxᵏ * 10^-5)
    yᵏ = ∇fxᵏ - ∇f(rᵏ)
    yᵏTsᵏ = dot(yᵏ, ∇fxᵏ)
    nsᵏ = γₖ¹ = dot(∇fxᵏ, ∇fxᵏ) * 10^-5 / yᵏTsᵏ
    γₖ² = yᵏTsᵏ * 10^-5 / dot(yᵏ, yᵏ)
    Uₖ₋₁ = Uₖ = min.(max.(10^-5 * ∇fxᵏ .* ∇fxᵏ ./ (∇fxᵏ .* yᵏ), γₖ²), γₖ¹)
    lastₘ = [Fxᵏ for i = 1:m]

    for k = 1:kₘₐₓ
        Fxₗ₍ₖ₎ = maximum(lastₘ)
        while true
            xᵏ = proxl0VM(xᵏ⁻¹ - Uₖ .* ∇fxᵏ, Uₖ)
            r!(rᵏ, xᵏ)
            Fxᵏ = F(rᵏ, xᵏ)
            sᵏ = xᵏ - xᵏ⁻¹
            if Fxᵏ + δ * dot(sᵏ, sᵏ ./ Uₖ) / 2 <= Fxₗ₍ₖ₎
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
        xᵏ⁻¹, Fxᵏ⁻¹ = xᵏ, Fxᵏ
        ∇fxᵏ⁻¹ = copy(∇fxᵏ)
        ∇f!(∇fxᵏ, rᵏ)
        nsᵏ = dot(sᵏ, sᵏ)
        yᵏ = ∇fxᵏ - ∇fxᵏ⁻¹
        nyᵏ = dot(yᵏ, yᵏ)
        yᵏTsᵏ = dot(yᵏ, sᵏ)
        γₖ¹ = yᵏTsᵏ > 0 ? nsᵏ / yᵏTsᵏ : sqrt(nsᵏ / nyᵏ)
        γₖ² = yᵏTsᵏ > 0 ? yᵏTsᵏ / nyᵏ : 1 / γₖ¹
        Uₖ₋₁, Uₖ = Uₖ, min.(max.((sᵏ .* sᵏ + µ .* Uₖ₋₁) ./ (sᵏ .* yᵏ .+ µ), γₖ²), γₖ¹)
    end
    return xᵏ, kₘₐₓ
end

function SPGH(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64))
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, X, XTX = funcs
    sᵏ = xᵏ⁻¹ = xᵏ = x⁰
    rᵏ = r(xᵏ)
    ∇fxᵏ⁻¹ = ∇fxᵏ = ∇f(rᵏ)
    Fxᵏ⁻¹ = Fxᵏ = F(rᵏ, xᵏ)
    r!(rᵏ, x⁰ - ∇fxᵏ * 10^-5)
    yᵏ = ∇fxᵏ - ∇f(rᵏ)
    yᵏTsᵏ = dot(yᵏ, ∇fxᵏ)
    nsᵏ = γₖ¹ = dot(∇fxᵏ, ∇fxᵏ) * 10^-5 / yᵏTsᵏ
    γₖ² = yᵏTsᵏ * 10^-5 / dot(yᵏ, yᵏ)
    γₖ = γₖ¹ < 2 * γₖ² ? γₖ² : γₖ¹ - γₖ² / 2
    lastₘ = [Fxᵏ for i = 1:m]

    for k = 1:kₘₐₓ
        Fxₗ₍ₖ₎ = maximum(lastₘ)
        while true
            xᵏ = proxl0(xᵏ⁻¹ - γₖ * ∇fxᵏ, γₖ)
            r!(rᵏ, xᵏ)
            Fxᵏ = F(rᵏ, xᵏ)
            sᵏ = xᵏ - xᵏ⁻¹
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
        xᵏ⁻¹, Fxᵏ⁻¹ = xᵏ, Fxᵏ
        ∇fxᵏ⁻¹ = copy(∇fxᵏ)
        ∇f!(∇fxᵏ, rᵏ)
        yᵏ = ∇fxᵏ - ∇fxᵏ⁻¹
        nyᵏ = dot(yᵏ, yᵏ)
        yᵏTsᵏ = dot(yᵏ, sᵏ)
        γₖ¹ = yᵏTsᵏ > 0 ? nsᵏ / yᵏTsᵏ : sqrt(nsᵏ / nyᵏ)
        γₖ² = yᵏTsᵏ > 0 ? yᵏTsᵏ / nyᵏ : 1 / γₖ¹
        γₖ = γₖ¹ < 2 * γₖ² ? γₖ² : γₖ¹ - γₖ² / 2
    end
    return xᵏ, kₘₐₓ
end

function SPG(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64), γₖ=0.0)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, X, XTX = funcs
    sᵏ = xᵏ⁻¹ = xᵏ = x⁰
    rᵏ = r(xᵏ)
    ∇fxᵏ⁻¹ = ∇fxᵏ = ∇f(rᵏ)
    Fxᵏ⁻¹ = Fxᵏ = F(rᵏ, xᵏ)
    if iszero(γₖ)
        r!(rᵏ, x⁰ - ∇fxᵏ * 10^-5)
        γₖ = dot(∇fxᵏ, ∇fxᵏ) * 10^-5 / dot(∇fxᵏ - ∇f(rᵏ), ∇fxᵏ)
    end
    nsᵏ = γₖ
    lastₘ = [Fxᵏ for i = 1:m]

    for k = 1:kₘₐₓ
        Fxₗ₍ₖ₎ = maximum(lastₘ)
        while true
            xᵏ = proxl0(xᵏ⁻¹ - γₖ * ∇fxᵏ, γₖ)
            r!(rᵏ, xᵏ)
            Fxᵏ = F(rᵏ, xᵏ)
            sᵏ = xᵏ - xᵏ⁻¹
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
        xᵏ⁻¹, Fxᵏ⁻¹ = xᵏ, Fxᵏ
        ∇fxᵏ⁻¹ = copy(∇fxᵏ)
        ∇f!(∇fxᵏ, rᵏ)
        yᵏ = ∇fxᵏ - ∇fxᵏ⁻¹
        γₖ = nsᵏ / dot(∇fxᵏ - ∇fxᵏ⁻¹, sᵏ)
        if γₖ > γₘₐₓ || γₖ < γₘᵢₙ
            γₖ = sqrt(nsᵏ / dot(yᵏ, yᵏ))
        end
    end
    return xᵏ, kₘₐₓ
end

function CDSS(x⁰, funcs; sortperc=1 / 4)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, X, XTX = funcs
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

function PSI1(xˡ, funcs)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, X, XTX = funcs
    r⃰ = -X'r(xˡ)

    for i = findall(!iszero, xˡ)
        jₘₐₓ = 0
        v⃰ₘₐₓ = 0.0
        for j = findall(iszero, xˡ)
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
# Setup test data
# ============================================================================
println("Setting up test data...")
Random.seed!(42)
vars = variables(corr=corr, ρ=ρ, n=250, p=p, SNR=SNR, k⃰=k⃰_val)
X, y, yval, XTX, p_val, β⃰, k⃰ = vars
λ₀ = 0.5 * maximum(abs.(X' * y))^2 / 2
fn = funcs(X, y, yval, XTX, p_val, β⃰, k⃰, λ₀)
x⁰ = zeros(Float64, p_val)

# Warmup
println("Warming up functions...")
VMSPG(copy(x⁰), fn)
SPGH(copy(x⁰), fn)
SPG(copy(x⁰), fn)
CDSS(copy(x⁰), fn)

# ============================================================================
# Benchmarks
# ============================================================================
println("\n" * "="^70)
println("BENCHMARKING INDIVIDUAL ALGORITHMS")
println("="^70)

println("\n--- VMSPG ---")
@btime VMSPG(x, $fn) setup = (x = copy($x⁰)) evals = 1

println("\n--- SPGH ---")
@btime SPGH(x, $fn) setup = (x = copy($x⁰)) evals = 1

println("\n--- SPG ---")
@btime SPG(x, $fn) setup = (x = copy($x⁰)) evals = 1

println("\n--- CDSS ---")
@btime CDSS(x, $fn) setup = (x = copy($x⁰)) evals = 1

# Benchmark PSI1 with a non-zero input
x_nonzero = copy(x⁰)
x_nonzero[1:20] .= randn(20)
println("\n--- PSI1 (with 20 non-zero elements) ---")
@btime PSI1(x, $fn) setup = (x = copy($x_nonzero)) evals = 1

# ============================================================================
# Allocation profiling per function
# ============================================================================
println("\n" * "="^70)
println("DETAILED ALLOCATION ANALYSIS")
println("="^70)

println("\n--- VMSPG allocation breakdown ---")
@time result_vmspg = VMSPG(copy(x⁰), fn)
println("Iterations: ", result_vmspg[2])

println("\n--- SPG allocation breakdown ---")
@time result_spg = SPG(copy(x⁰), fn)
println("Iterations: ", result_spg[2])

println("\n--- CDSS allocation breakdown ---")
@time result_cdss = CDSS(copy(x⁰), fn)
println("Iterations: ", result_cdss[2])

# ============================================================================
# Per-operation benchmarks (inner loop operations)
# ============================================================================
println("\n" * "="^70)
println("INNER LOOP OPERATION BENCHMARKS")
println("="^70)

# Setup for inner loop tests
r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, Xf, XTXf = fn
rᵏ = r(x⁰)
∇fxᵏ = ∇f(rᵏ)
Uₖ = ones(p_val) * 0.01

println("\n--- r(β) = X * β - y (residual, allocating) ---")
@btime r($x⁰)

println("\n--- r!(r, β) (residual, in-place) ---")
r_buf = similar(y)
@btime r!($r_buf, $x⁰)

println("\n--- ∇f(r) = X'r (gradient, allocating) ---")
@btime ∇f($rᵏ)

println("\n--- ∇f!(∇f, r) (gradient, in-place) ---")
grad_buf = similar(x⁰)
@btime ∇f!($grad_buf, $rᵏ)

println("\n--- proxl0VM (proximal operator with variable metric) ---")
input_prox = randn(p_val)
@btime proxl0VM($input_prox, $Uₖ)

println("\n--- sᵏ = xᵏ - xᵏ⁻¹ (difference, allocating) ---")
xᵏ = randn(p_val)
xᵏ⁻¹ = randn(p_val)
@btime $xᵏ - $xᵏ⁻¹

println("\n--- In-place alternative: sᵏ .= xᵏ .- xᵏ⁻¹ ---")
sᵏ = similar(x⁰)
@btime $sᵏ .= $xᵏ .- $xᵏ⁻¹

println("\n--- yᵏ = ∇fxᵏ - ∇fxᵏ⁻¹ (allocating) ---")
∇fxᵏ⁻¹ = randn(p_val)
@btime $∇fxᵏ - $∇fxᵏ⁻¹

println("\n--- copy(∇fxᵏ) ---")
@btime copy($∇fxᵏ)

println("\n--- copyto! alternative ---")
dest = similar(∇fxᵏ)
@btime copyto!($dest, $∇fxᵏ)

println("\n--- Uₖ update: min.(max.(...), γₖ¹) (allocating) ---")
sᵏ_test = randn(p_val)
yᵏ_test = randn(p_val)
µ = 10^-3
γₖ¹ = 0.1
γₖ² = 0.01
Uₖ₋₁ = ones(p_val) * 0.01
@btime min.(max.(($sᵏ_test .* $sᵏ_test + $µ .* $Uₖ₋₁) ./ ($sᵏ_test .* $yᵏ_test .+ $µ), $γₖ²), $γₖ¹)

println("\n--- popfirst!/push! on Vector ---")
lastₘ = collect(1.0:15.0)
@btime begin
    popfirst!($lastₘ)
    push!($lastₘ, 16.0)
end setup = (lastₘ = collect(1.0:15.0))

println("\n--- findall(!iszero, x) ---")
sparse_x = zeros(p_val)
sparse_x[1:20] .= randn(20)
@btime findall(!iszero, $sparse_x)

println("\n--- findall(iszero, x) ---")
@btime findall(iszero, $sparse_x)

println("\n" * "="^70)
println("BENCHMARK COMPLETE")
println("="^70)
