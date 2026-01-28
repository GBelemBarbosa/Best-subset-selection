# Fix the λ scaling between CDSS and L0Learn
# CDSS λ_max = 3.53, L0Learn λ_max = 0.14 → ratio ~26x
# This suggests L0Learn uses different normalization (likely divides by n)

using Random, Distributions, LinearAlgebra, RCall

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
σ = sqrt(norm(X * β_true)^2 / (n * 5))
y = X * β_true + randn(n) * σ

println("Test data: n=$n, p=$p, k=$k")
println("True support: ", findall(!iszero, β_true))

# Get L0Learn's λ path
RCall.reval(RCall.rparse("suppressMessages(library(L0Learn))"))
RCall.globalEnv[:X_test] = collect(X)
RCall.globalEnv[:y_test] = collect(y)
RCall.globalEnv[:n_obs] = n

RCall.reval(RCall.rparse("fit <- L0Learn.fit(X_test, y_test, penalty='L0', maxSuppSize=50)"))
RCall.reval(RCall.rparse("l0learn_lambdas <- fit\$lambda[[1]]"))
l0_lambdas = RCall.rcopy(RCall.reval(RCall.rparse("l0learn_lambdas")))

# CDSS formulas
∇f = -X'y
cdss_λ_max = 1.01 * maximum(abs.(∇f))^2 / 2

println("\n" * "="^60)
println("TESTING SCALING HYPOTHESES")
println("="^60)

# Hypothesis A: L0Learn divides loss by n (common in sklearn-style)
# So L0Learn's λ should be CDSS_λ / n
scaling_A = n
println("\nHypothesis A: L0Learn divides by n")
println("  CDSS λ_max / n = $(cdss_λ_max / n)")
println("  L0Learn λ_max  = $(l0_lambdas[1])")
println("  Ratio = $(cdss_λ_max / n / l0_lambdas[1])")

# Hypothesis B: L0Learn uses |∇f|/n as threshold (not squared)
# λ_max = max|X'y|/n  (not squared)
l0learn_style_λ = maximum(abs.(∇f)) / n
println("\nHypothesis B: L0Learn uses gradient/n (not squared)")
println("  max|X'y|/n = $(l0learn_style_λ)")
println("  L0Learn λ_max = $(l0_lambdas[1])")
println("  Ratio = $(l0learn_style_λ / l0_lambdas[1])")

# Hypothesis C: L0Learn uses max|X'y|²/(2n) 
# (squared gradient but divided by n)
l0learn_style_C = maximum(abs.(∇f))^2 / (2 * n)
println("\nHypothesis C: L0Learn uses |X'y|²/(2n)")
println("  max|X'y|²/(2n) = $(l0learn_style_C)")
println("  L0Learn λ_max = $(l0_lambdas[1])")
println("  Ratio = $(l0learn_style_C / l0_lambdas[1])")

# Hypothesis D: Maybe columns aren't normalized the same way
# Check L0Learn's automatic normalization behavior
RCall.reval(RCall.rparse("""
# Check if L0Learn normalizes internally
X_colnorms <- apply(X_test, 2, function(x) sqrt(sum(x^2)))
mean_norm <- mean(X_colnorms)
"""))
mean_norm = RCall.rcopy(RCall.reval(RCall.rparse("mean_norm")))
println("\nHypothesis D: Column normalization check")
println("  Mean column norm (should be 1): $(mean_norm)")

# Let's try the empirical approach: find the scaling that makes supports match
println("\n" * "="^60)
println("EMPIRICAL SCALING SEARCH")
println("="^60)

function simple_CDSS(X, y, λ; maxiter=1000)
    n, p = size(X)
    HT = sqrt(2λ)
    β = zeros(p)
    r = -y
    for iter in 1:maxiter
        β_old = copy(β)
        for i in 1:p
            xi = (dot(r, view(X,:,i)) + β[i])
            xi_new = (abs(xi) >= HT) ? xi : 0.0
            if xi_new != β[i]
                r .+= (β[i] - xi_new) .* view(X,:,i)
                β[i] = xi_new
            end
        end
        if norm(β - β_old) / max(norm(β), 1e-10) < 1e-8
            break
        end
    end
    return β
end

# For each L0Learn λ, find the CDSS λ that gives same support size
println("\nFinding empirical λ mapping...")
println("L0Learn λ | L0Learn supp | CDSS scaled λ | CDSS supp | Scale factor")
println("-"^70)

scale_factors = Float64[]
for l0_λ in l0_lambdas[1:min(10, length(l0_lambdas))]
    # Get L0Learn support at this λ
    RCall.globalEnv[:curr_lambda] = l0_λ
    RCall.reval(RCall.rparse("beta_l0 <- as.numeric(coef(fit, lambda=curr_lambda, gamma=0)[-1])"))
    β_l0 = RCall.rcopy(RCall.reval(RCall.rparse("beta_l0")))
    supp_l0 = count(x -> abs(x) > 1e-10, β_l0)
    
    if supp_l0 == 0
        continue
    end
    
    # Binary search for CDSS λ that gives same support size
    lo, hi = l0_λ, cdss_λ_max * 2
    for _ in 1:30
        mid = (lo + hi) / 2
        β_cdss = simple_CDSS(X, y, mid)
        supp_cdss = count(!iszero, β_cdss)
        if supp_cdss > supp_l0
            lo = mid  # Need larger λ (more sparsity)
        else
            hi = mid  # Need smaller λ (less sparsity)
        end
    end
    
    β_cdss = simple_CDSS(X, y, hi)
    supp_cdss = count(!iszero, β_cdss)
    scale = hi / l0_λ
    push!(scale_factors, scale)
    
    println("$(round(l0_λ, sigdigits=4)) | $(lpad(supp_l0, 5)) | $(round(hi, sigdigits=4)) | $(lpad(supp_cdss, 5)) | $(round(scale, digits=1))")
end

if !isempty(scale_factors)
    avg_scale = sum(scale_factors) / length(scale_factors)
    println("\n" * "="^60)
    println("RESULT: Average scale factor = $(round(avg_scale, digits=1))")
    println("="^60)
    println("\nTo use L0Learn with CDSS framework:")
    println("  cdss_λ = l0learn_λ * $(round(avg_scale, digits=1))")
    println("  l0learn_λ = cdss_λ / $(round(avg_scale, digits=1))")
end
