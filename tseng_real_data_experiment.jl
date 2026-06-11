# ============================================================================
# Real Data Experiment
# Reuses solver infrastructure from snr_experiment.jl
#
# Replicates Section 5.6 of "Fast Best Subset Selection" (Hazimeh & Mazumder, 2020)
# 
# Datasets from the paper:
#   1. House Prices: Boston Housing + pairwise interactions + 1000 probe permutations
#      → p=104,000, n=200/100/206 (train/val/test)
#   2. Amazon Reviews: Grocery & Gourmet Food TF-IDF
#      → p=17,580, n=2500/500/1868
#   3. US Census: 2016 Planning Database, mail-return rate
#      → p=55,537, n=5000/5000/5000
#
# For our comparison, we start with the Boston Housing dataset (simplest to load)
# and add probe features, matching the paper's protocol.
#
# Usage: julia real_data_experiment.jl [dataset] [T] [n_probes] [--dry-run]
#   dataset: "boston", "boston_probes", or "all" (default: "boston")
#   T: number of trials (default: 5)
#   n_probes: number of probe permutations per column (default: 0; paper uses 1000)
#   --dry-run: run 1 trial on boston (no probes)
#
# Example: julia real_data_experiment.jl boston 5
#          julia real_data_experiment.jl boston_probes 5 100
# ============================================================================

# ============================================================================
# Command-line Parsing
# ============================================================================

dry_run = "--dry-run" in ARGS
filter!(a -> a != "--dry-run", ARGS)

dataset_name = length(ARGS) >= 1 ? ARGS[1] : "boston"
T_real = dry_run ? 1 : (length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5)
n_probes = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0

# Include the full solver infrastructure
include("snr_experiment.jl")

using Statistics

@info "Real Data Experiment" dataset=dataset_name trials=T_real n_probes dry_run

# ============================================================================
# Data Loading via RCall
# ============================================================================

"""
    load_boston_housing() -> (X::Matrix{Float64}, y::Vector{Float64})

Load the Boston Housing dataset. Returns 506 samples × 13 features.
"""
function load_boston_housing()
    # MASS::Boston or mlbench::BostonHousing
    RCall.reval(raw"""
    if (requireNamespace("mlbench", quietly=TRUE)) {
        library(mlbench)
        data(BostonHousing)
        X_r <- as.matrix(sapply(BostonHousing[, 1:13], as.numeric))
        y_r <- BostonHousing$medv
    } else if (requireNamespace("MASS", quietly=TRUE)) {
        library(MASS)
        data(Boston)
        X_r <- as.matrix(sapply(Boston[, 1:13], as.numeric))
        y_r <- Boston$medv
    } else {
        stop("Neither mlbench nor MASS package available for Boston Housing")
    }
    """)
    X = Float64.(RCall.rcopy(RCall.reval("X_r")))
    y = Float64.(RCall.rcopy(RCall.reval("y_r")))
    return X, y
end

"""
    add_pairwise_interactions(X) -> X_extended

Add pairwise interaction features (x_i * x_j for all i <= j), 
matching the paper: "We added pairwise interactions to get 104 features."
For 13 original features: 13 + C(13,2) + 13 = 13 + 78 + 13 = 104
(original + cross-interactions + squared terms)
"""
function add_pairwise_interactions(X::Matrix{Float64})
    n, p = size(X)
    cols = AbstractMatrix{Float64}[X]
    for i in 1:p
        for j in i:p
            push!(cols, reshape(X[:, i] .* X[:, j], n, 1))
        end
    end
    return reduce(hcat, cols)
end

"""
    add_probe_features(X, n_probes) -> X_augmented

Add random probe (noise) features by appending `n_probes` random 
permutations of every column. Paper uses 1000 permutations.
"""
function add_probe_features(X::Matrix{Float64}, n_probes::Int)
    n, p = size(X)
    if n_probes <= 0
        return X
    end
    X_aug = copy(X)
    for _ in 1:n_probes
        probe = similar(X)
        for j in 1:p
            probe[:, j] = X[randperm(n), j]
        end
        X_aug = hcat(X_aug, probe)
    end
    @info "  Added $n_probes probe permutations: p=$(size(X,2)) → p=$(size(X_aug,2))"
    return X_aug
end

"""
    preprocess_real_data(X, y) -> (X_proc, y_proc)

Center columns and normalize to unit L2 norm (matching snr_experiment.jl convention).
"""
function preprocess_real_data(X::Matrix{Float64}, y::Vector{Float64})
    n, p_dim = size(X)
    X_proc = copy(X)
    
    # Center columns
    col_means = mean(X_proc, dims=1)
    X_proc .-= col_means
    
    # Center y
    y_proc = y .- mean(y)
    
    # Normalize columns to unit L2 norm (consistent with snr_experiment.jl line 100)
    for j in 1:p_dim
        nrm = norm(view(X_proc, :, j))
        if nrm > 0
            X_proc[:, j] ./= nrm
        end
    end
    
    return X_proc, y_proc
end

"""
    split_data(X, y; n_train, n_val)

Fixed-size train/val/test split (paper uses specific sizes per dataset).
"""
function split_data(X, y; n_train::Int, n_val::Int)
    n = size(X, 1)
    perm = randperm(n)
    
    idx_train = perm[1:n_train]
    idx_val = perm[n_train+1:n_train+n_val]
    idx_test = perm[n_train+n_val+1:end]
    
    return X[idx_train, :], y[idx_train], 
           X[idx_val, :], y[idx_val],
           X[idx_test, :], y[idx_test]
end

"""
    make_vars_real(X_train, y_train, y_val)

Construct the vars tuple compatible with snr_experiment.jl's solver infrastructure.
Since there's no ground truth β†, we use β† = zeros(p) and k† = 0.
"""
function make_vars_real(X_train, y_train, X_val, y_val)
    p_dim = size(X_train, 2)
    XTX = X_train' * X_train
    β_dummy = zeros(p_dim)  # No ground truth
    k_dummy = 0
    return X_train, y_train, X_val, y_val, XTX, p_dim, β_dummy, k_dummy
end

# ============================================================================
# Main Experiment Loop
# ============================================================================

function run_real_data_experiment(dataset::String, T_trials::Int, n_probe::Int)
    @info "Loading dataset: $dataset"
    
    if dataset in ("boston", "boston_probes")
        X_raw, y_raw = load_boston_housing()
        @info "  Raw Boston Housing loaded" n=size(X_raw, 1) p=size(X_raw, 2)
        
        # Add pairwise interactions (13 → 104 features, as in paper)
        X_raw = add_pairwise_interactions(X_raw)
        @info "  After interactions" p=size(X_raw, 2)
        
        # Add probe features if requested
        if dataset == "boston_probes" || n_probe > 0
            actual_probes = n_probe > 0 ? n_probe : 10  # default to 10 for quick testing (paper uses 1000)
            X_raw = add_probe_features(X_raw, actual_probes)
        end
        
        # Paper split: n_train=200, n_val=100, n_test=206
        split_train = 200
        split_val = 100
        # test = remaining
        
    else
        error("Unknown dataset: $dataset. Use 'boston', 'boston_probes'.")
    end
    
    X_full, y_full = preprocess_real_data(X_raw, y_raw)
    n_full, p_full = size(X_full)
    
    @info "Preprocessed data" n=n_full p=p_full
    
    algo_names = [
        "NSPG (L0L2 Dual CV)",
        "NSPG+PGCCD (L0L2 Dual CV)",
        "PGCCD (L0L2 Dual CV)",
        "NSPG (L0L1 Dual CV)",
        "NSPG+PGCCD (L0L1 Dual CV)",
        "PGCCD (L0L1 Dual CV)",
        "NSPG (L0)",
        "NSPG+PGCCD (L0)",
        "PGCCD (L0)",
        "NSPG (L0 + Opt L1)",
        "NSPG+PGCCD (L0 + Opt L1)",
        "PGCCD (L0 + Opt L1)",
        "NSPG (L0 + Opt L2)",
        "NSPG+PGCCD (L0 + Opt L2)",
        "PGCCD (L0 + Opt L2)",
        "L0Learn Native (L0)",
        "L0Learn Native (L0 + Opt L1)",
        "L0Learn Native (L0 + Opt L2)"
    ]
    n_algos = length(algo_names)
    
    # Metrics storage
    TestMSE = zeros(n_algos)
    Cardinality = zeros(n_algos)
    TimeTotal = zeros(n_algos)
    ValMSE = zeros(n_algos)
    
    for trial in 1:T_trials
        @info "Trial $trial/$T_trials for dataset=$dataset"
        
        # Split data (paper uses fixed sizes)
        X_tr, y_tr, X_va, y_va, X_te, y_te = split_data(X_full, y_full;
                                                          n_train=split_train, n_val=split_val)
        
        p_local = size(X_tr, 2)
        
        vars = make_vars_real(X_tr, y_tr, X_va, y_va)
        
        # Precompute optimal ridge & lasso penalties for fair comparisons (time excluded from solvers)
        opt_ridge_val = find_optimal_ridge(X_tr, y_tr, X_va, y_va)
        opt_lasso_val = find_optimal_lasso(X_tr, y_tr, X_va, y_va)
        @info "Optimal penalties" opt_ridge_val opt_lasso_val
        
        # Test MSE helpers
        test_mse(β) = norm(X_te * β - y_te)^2 / length(y_te)
        val_mse(β) = norm(X_va * β - y_va)^2 / length(y_va)
        
        # Helper to execute an algorithm and update metrics
        function run_algo!(idx, name, solver_fn)
            trial_start = time()
            β_res = try
                solver_fn()
            catch e
                @warn "Algorithm $idx ($name) failed: $e"
                zeros(p_local)
            end
            dt = time() - trial_start
            TestMSE[idx] += test_mse(β_res)
            Cardinality[idx] += norm(β_res, 0)
            TimeTotal[idx] += dt
            ValMSE[idx] += val_mse(β_res)
            @info "  [$idx] $name" k=Int(norm(β_res,0)) test_mse=round(test_mse(β_res), digits=4) time=round(dt, digits=1)
        end

        # ============================================================
        # Set 1: Dual Grid Search (6 algorithms)
        # ============================================================

        # 1. NSPG (L0L2 Dual CV)
        run_algo!(1, "NSPG (L0L2 Dual CV)", () -> begin
            β, _ = outer_l2_smart_cross_validation(
                (x, f; kw...) -> SolverPSI1(NSPG, x, f; kw...), 
                vars; NSPG=true, use_refinement=true, inner_cv=smart_adaptive_cross_validation)
            β
        end)

        # 2. NSPG+PGCCD (L0L2 Dual CV)
        run_algo!(2, "NSPG+PGCCD (L0L2 Dual CV)", () -> begin
            β, _ = outer_l2_smart_cross_validation(
                (x, f; kw...) -> SolverPSI1(NSPG_PGCCD, x, f; kw...), 
                vars; NSPG=true, use_refinement=true, inner_cv=smart_adaptive_cross_validation)
            β
        end)

        # 3. PGCCD (L0L2 Dual CV)
        run_algo!(3, "PGCCD (L0L2 Dual CV)", () -> begin
            β, _ = outer_l2_smart_cross_validation(
                (x, f; kw...) -> SolverPSI1(PGCCD, x, f; kw...), 
                vars; NSPG=false, use_refinement=true, inner_cv=cross_validation)
            β
        end)

        # 4. NSPG (L0L1 Dual CV)
        run_algo!(4, "NSPG (L0L1 Dual CV)", () -> begin
            β, _ = outer_l1_smart_cross_validation(
                (x, f; kw...) -> SolverPSI1(NSPG, x, f; kw...), 
                vars; NSPG=true, use_refinement=true, inner_cv=smart_adaptive_cross_validation)
            β
        end)

        # 5. NSPG+PGCCD (L0L1 Dual CV)
        run_algo!(5, "NSPG+PGCCD (L0L1 Dual CV)", () -> begin
            β, _ = outer_l1_smart_cross_validation(
                (x, f; kw...) -> SolverPSI1(NSPG_PGCCD, x, f; kw...), 
                vars; NSPG=true, use_refinement=true, inner_cv=smart_adaptive_cross_validation)
            β
        end)

        # 6. PGCCD (L0L1 Dual CV)
        run_algo!(6, "PGCCD (L0L1 Dual CV)", () -> begin
            β, _ = outer_l1_smart_cross_validation(
                (x, f; kw...) -> SolverPSI1(PGCCD, x, f; kw...), 
                vars; NSPG=false, use_refinement=true, inner_cv=cross_validation)
            β
        end)

        # ============================================================
        # Set 2: 1D Grid Search over L0 (12 algorithms)
        # ============================================================

        # -- L0 only (L2=0, L1=0) --

        # 7. NSPG (L0)
        run_algo!(7, "NSPG (L0)", () -> begin
            res = smart_adaptive_cross_validation(
                (x, f; kw...) -> SolverPSI1(NSPG, x, f; kw...), 
                vars; NSPG=true, use_refinement=true)
            res[1]
        end)

        # 8. NSPG+PGCCD (L0)
        run_algo!(8, "NSPG+PGCCD (L0)", () -> begin
            res = smart_adaptive_cross_validation(
                (x, f; kw...) -> SolverPSI1(NSPG_PGCCD, x, f; kw...), 
                vars; NSPG=true, use_refinement=true)
            res[1]
        end)

        # 9. PGCCD (L0)
        run_algo!(9, "PGCCD (L0)", () -> begin
            res = cross_validation(
                (x, f; kw...) -> SolverPSI1(PGCCD, x, f; kw...), 
                vars; NSPG=false, use_refinement=true)
            res[1]
        end)

        # -- L0 + fixed Lasso L1 penalty --

        # 10. NSPG (L0 + Opt L1)
        run_algo!(10, "NSPG (L0 + Opt L1)", () -> begin
            res = smart_adaptive_cross_validation(
                (x, f; kw...) -> SolverPSI1(NSPG, x, f; kw...), 
                vars; NSPG=true, use_refinement=true, fixed_l1=opt_lasso_val)
            res[1]
        end)

        # 11. NSPG+PGCCD (L0 + Opt L1)
        run_algo!(11, "NSPG+PGCCD (L0 + Opt L1)", () -> begin
            res = smart_adaptive_cross_validation(
                (x, f; kw...) -> SolverPSI1(NSPG_PGCCD, x, f; kw...), 
                vars; NSPG=true, use_refinement=true, fixed_l1=opt_lasso_val)
            res[1]
        end)

        # 12. PGCCD (L0 + Opt L1)
        run_algo!(12, "PGCCD (L0 + Opt L1)", () -> begin
            res = cross_validation(
                (x, f; kw...) -> SolverPSI1(PGCCD, x, f; kw...), 
                vars; NSPG=false, use_refinement=true, fixed_l1=opt_lasso_val)
            res[1]
        end)

        # -- L0 + fixed Ridge L2 penalty --

        # 13. NSPG (L0 + Opt L2)
        run_algo!(13, "NSPG (L0 + Opt L2)", () -> begin
            res = smart_adaptive_cross_validation(
                (x, f; kw...) -> SolverPSI1(NSPG, x, f; kw...), 
                vars; NSPG=true, use_refinement=true, fixed_l2=opt_ridge_val)
            res[1]
        end)

        # 14. NSPG+PGCCD (L0 + Opt L2)
        run_algo!(14, "NSPG+PGCCD (L0 + Opt L2)", () -> begin
            res = smart_adaptive_cross_validation(
                (x, f; kw...) -> SolverPSI1(NSPG_PGCCD, x, f; kw...), 
                vars; NSPG=true, use_refinement=true, fixed_l2=opt_ridge_val)
            res[1]
        end)

        # 15. PGCCD (L0 + Opt L2)
        run_algo!(15, "PGCCD (L0 + Opt L2)", () -> begin
            res = cross_validation(
                (x, f; kw...) -> SolverPSI1(PGCCD, x, f; kw...), 
                vars; NSPG=false, use_refinement=true, fixed_l2=opt_ridge_val)
            res[1]
        end)

        # -- Native R L0Learn counterparts --
        if L0LEARN_AVAILABLE
            initialize_l0learn_R(vars)

            # 16. L0Learn Native (L0)
            run_algo!(16, "L0Learn Native (L0)", () -> begin
                pure_l0learn_solver_R(vars; maxSuppSize=p_local, l2_val=0.0, l1_val=0.0)
            end)

            # 17. L0Learn Native (L0 + Opt L1)
            run_algo!(17, "L0Learn Native (L0 + Opt L1)", () -> begin
                pure_l0learn_solver_R(vars; maxSuppSize=p_local, l2_val=0.0, l1_val=opt_lasso_val)
            end)

            # 18. L0Learn Native (L0 + Opt L2)
            run_algo!(18, "L0Learn Native (L0 + Opt L2)", () -> begin
                pure_l0learn_solver_R(vars; maxSuppSize=p_local, l2_val=opt_ridge_val, l1_val=0.0)
            end)
        end
        
        @info "Trial $trial/$T_trials completed"
    end
    
    # ============================================================
    # Normalize by T and print results
    # ============================================================
    TestMSE ./= T_trials
    Cardinality ./= T_trials
    TimeTotal ./= T_trials
    ValMSE ./= T_trials
    
    println("\n" * "="^80)
    println("RESULTS — Dataset: $dataset (n=$(n_full), p=$(p_full)), T=$T_trials trials")
    println("="^80)
    println()
    
    println("SET 1: Dual Grid Search")
    println("-"^70)
    @printf("%-30s %12s %12s %10s\n", "Algorithm", "TestMSE", "‖β‖₀", "Time(s)")
    println("-"^70)
    for i in 1:6
        @printf("%-30s %12.4f %12.1f %10.2f\n", 
                algo_names[i], TestMSE[i], Cardinality[i], TimeTotal[i])
    end
    println("-"^70)
    println()
    
    println("SET 2: 1D Grid Search over L0")
    println("-"^70)
    @printf("%-30s %12s %12s %10s\n", "Algorithm", "TestMSE", "‖β‖₀", "Time(s)")
    println("-"^70)
    for i in 7:18
        if !L0LEARN_AVAILABLE && i >= 16
            continue
        end
        @printf("%-30s %12.4f %12.1f %10.2f\n", 
                algo_names[i], TestMSE[i], Cardinality[i], TimeTotal[i])
    end
    println("="^80)
    
    # Save results to file
    results_file = "realdata_$(dataset)_p$(p_full)_T$(T_trials)_v3.txt"
    open(results_file, "w") do io
        println(io, "Dataset: $dataset, n=$n_full, p=$p_full, T=$T_trials")
        println(io, "")
        println(io, "SET 1: Dual Grid Search")
        println(io, "-"^70)
        @printf(io, "%-30s %12s %12s %10s\n", "Algorithm", "TestMSE", "‖β‖₀", "Time(s)")
        println(io, "-"^70)
        for i in 1:6
            @printf(io, "%-30s %12.4f %12.1f %10.2f\n", 
                    algo_names[i], TestMSE[i], Cardinality[i], TimeTotal[i])
        end
        println(io, "-"^70)
        println(io, "")
        
        println(io, "SET 2: 1D Grid Search over L0")
        println(io, "-"^70)
        @printf(io, "%-30s %12s %12s %10s\n", "Algorithm", "TestMSE", "‖β‖₀", "Time(s)")
        println(io, "-"^70)
        for i in 7:18
            if !L0LEARN_AVAILABLE && i >= 16; continue; end
            @printf(io, "%-30s %12.4f %12.1f %10.2f\n",
                    algo_names[i], TestMSE[i], Cardinality[i], TimeTotal[i])
        end
        println(io, "="^80)
    end
    @info "Results saved to $results_file"
    
    return TestMSE, Cardinality, TimeTotal
end

# ============================================================================
# Entry Point
# ============================================================================

function main_real()
    datasets = if dataset_name == "all"
        ["boston", "boston_probes"]
    else
        [dataset_name]
    end
    
    for ds in datasets
        @info "="^60
        @info "Starting experiment for dataset: $ds"
        @info "="^60
        try
            run_real_data_experiment(ds, T_real, n_probes)
        catch e
            @error "Experiment failed for dataset $ds" exception=(e, catch_backtrace())
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_real()
end
