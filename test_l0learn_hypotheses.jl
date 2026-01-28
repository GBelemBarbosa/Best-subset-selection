# Test hypotheses about L0Learn vs CDSS differences
# Hypothesis 1: L0Learn uses different λ scaling/normalization
# Hypothesis 2: L0Learn's CD converges to different local minima

using Random
using Distributions
using LinearAlgebra
using RCall

Random.seed!(42)

# Generate test data
n, p, k = 200, 100, 10
ρ = 0.9
Σ = [ρ^abs(i - j) for i = 1:p, j = 1:p]
d = MvNormal(zeros(p), Σ)
X = rand(d, n)'
for i = 1:p
    X[:, i] /= norm(view(X, :, i))
end
β_true = [1.0 * (mod(i, Int(p / k)) == 0) for i = 1:p]
σ = sqrt(norm(X * β_true)^2 / (n * 5))  # SNR = 5
y = X * β_true + randn(n) * σ

println("="^60)
println("TEST DATA")
println("="^60)
println("n=$n, p=$p, k=$k (true support size)")
println("True support indices: ", findall(!iszero, β_true))

# ============================================================================
# HYPOTHESIS 1: λ scaling differences
# ============================================================================
println("\n" * "="^60)
println("HYPOTHESIS 1: λ SCALING COMPARISON")
println("="^60)

# CDSS λ max formula: λ_max = 1.01 * max|X'y|² / 2
∇f = -X'y
cdss_λ_max = 1.01 * maximum(abs.(∇f))^2 / 2
println("\nCDSS λ_max formula: 1.01 * max|X'y|² / 2 = $(round(cdss_λ_max, sigdigits=5))")

# L0Learn's λ max
RCall.reval(RCall.rparse("suppressMessages(library(L0Learn))"))
RCall.globalEnv[:X_test] = collect(X)
RCall.globalEnv[:y_test] = collect(y)

r_code = raw"""
fit <- L0Learn.fit(X_test, y_test, penalty="L0", maxSuppSize=50)
l0learn_lambdas <- fit$lambda[[1]]
"""
RCall.reval(RCall.rparse(r_code))
l0learn_lambdas = RCall.rcopy(RCall.reval(RCall.rparse("l0learn_lambdas")))

println("L0Learn λ path: ", round.(l0learn_lambdas[1:min(10, length(l0learn_lambdas))], sigdigits=4))
println("L0Learn λ_max = $(round(l0learn_lambdas[1], sigdigits=5))")
println("L0Learn λ_min = $(round(l0learn_lambdas[end], sigdigits=5))")

ratio = cdss_λ_max / l0learn_lambdas[1]
println("\nRATIO (CDSS λ_max / L0Learn λ_max) = $(round(ratio, sigdigits=4))")
if ratio > 1.5 || ratio < 0.67
    println("⚠️  HYPOTHESIS 1 SUPPORTED: Significant λ scaling difference!")
else
    println("✓  λ values are in similar range")
end

# ============================================================================
# HYPOTHESIS 2: Different convergence at same λ
# ============================================================================
println("\n" * "="^60)
println("HYPOTHESIS 2: CONVERGENCE COMPARISON AT SAME λ")
println("="^60)

# Pick a λ from L0Learn's path that gives reasonable support
test_λ = l0learn_lambdas[div(length(l0learn_lambdas), 2)]
println("\nTest λ = $(round(test_λ, sigdigits=5))")

# Get L0Learn solution at this λ
RCall.globalEnv[:test_lambda] = test_λ
r_code2 = raw"""
beta_l0 <- as.numeric(coef(fit, lambda=test_lambda, gamma=0)[-1])
"""
RCall.reval(RCall.rparse(r_code2))
β_l0learn = RCall.rcopy(RCall.reval(RCall.rparse("beta_l0")))

# Now run CDSS at same λ
kₘₐₓ = 1000
ϵ = 1e-7
HT = sqrt(2 * test_λ)
proxl0(x) = (abs(x) >= HT) * x
XTX = X'X

# Simple CDSS implementation
function simple_CDSS(X, y, λ; maxiter=1000)
    n, p = size(X)
    HT = sqrt(2λ)
    β = zeros(p)
    r = -y  # residual = Xβ - y, starting at β=0
    
    for iter in 1:maxiter
        β_old = copy(β)
        for i in 1:p
            xi_col = view(X, :, i)
            xi = (dot(r, xi_col) + β[i])
            xi_new = (abs(xi) >= HT) ? xi : 0.0
            if xi_new != β[i]
                r .+= (β[i] - xi_new) .* xi_col
                β[i] = xi_new
            end
        end
        if norm(β - β_old) / max(norm(β), 1e-10) < 1e-8
            return β, iter
        end
    end
    return β, maxiter
end

β_cdss, cdss_iters = simple_CDSS(X, y, test_λ)

# Compare
supp_l0learn = findall(x -> abs(x) > 1e-10, β_l0learn)
supp_cdss = findall(!iszero, β_cdss)
supp_true = findall(!iszero, β_true)

println("\n--- Results at λ = $(round(test_λ, sigdigits=4)) ---")
println("L0Learn support size: $(length(supp_l0learn))")
println("L0Learn support: ", supp_l0learn)
println("\nCDSS support size: $(length(supp_cdss))")
println("CDSS support: ", supp_cdss)
println("\nTrue support: ", supp_true)

# Overlap
overlap = length(intersect(supp_l0learn, supp_cdss))
println("\n--- Comparison ---")
println("Support overlap (L0Learn ∩ CDSS): $overlap / $(max(length(supp_l0learn), length(supp_cdss)))")

if overlap == length(supp_l0learn) == length(supp_cdss)
    println("✓  Same support found")
    
    # Check coefficient values
    coef_diff = norm(β_l0learn - β_cdss) / max(norm(β_cdss), 1e-10)
    println("Coefficient difference: $(round(coef_diff * 100, digits=2))%")
    if coef_diff > 0.01
        println("⚠️  Different coefficient values despite same support")
    end
else
    println("⚠️  HYPOTHESIS 2 SUPPORTED: Different supports at same λ!")
    println("    L0Learn finds different local minimum than CDSS")
end

# Also test at multiple λ values
println("\n" * "="^60)
println("SUPPORT COMPARISON ACROSS λ PATH")
println("="^60)

println("\n λ value      | L0Learn supp | CDSS supp | True overlap L0 | True overlap CDSS")
println("-"^80)

for (i, λ) in enumerate(l0learn_lambdas[1:min(15, length(l0learn_lambdas))])
    RCall.globalEnv[:curr_lambda] = λ
    RCall.reval(RCall.rparse("beta_curr <- as.numeric(coef(fit, lambda=curr_lambda, gamma=0)[-1])"))
    β_l0 = RCall.rcopy(RCall.reval(RCall.rparse("beta_curr")))
    
    β_cd, _ = simple_CDSS(X, y, λ)
    
    supp_l0 = findall(x -> abs(x) > 1e-10, β_l0)
    supp_cd = findall(!iszero, β_cd)
    
    true_in_l0 = length(intersect(supp_l0, supp_true))
    true_in_cd = length(intersect(supp_cd, supp_true))
    
    println(" $(round(λ, sigdigits=4)) | $(lpad(length(supp_l0), 5)) | $(lpad(length(supp_cd), 5)) | $(lpad(true_in_l0, 5))/$(k) | $(lpad(true_in_cd, 5))/$(k)")
end

println("\n" * "="^60)
println("CONCLUSION")
println("="^60)
