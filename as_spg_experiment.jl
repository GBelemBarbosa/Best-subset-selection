# ============================================================================
# Active-Set NSPG Experiment
# Tests the scalability hypothesis on boston_probes
# ============================================================================

using Printf
using Random
using LinearAlgebra
using Statistics
using RCall

include("snr_experiment.jl")

# ----------------------------------------------------------------------------
# Dataset Loaders (from real_data_experiment.jl)
# ----------------------------------------------------------------------------

function load_boston_housing()
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
    return X_aug
end

function preprocess_real_data(X::Matrix{Float64}, y::Vector{Float64})
    n, p_dim = size(X)
    X_proc = copy(X)
    
    col_means = mean(X_proc, dims=1)
    X_proc .-= col_means
    
    y_proc = y .- mean(y)
    
    for j in 1:p_dim
        nrm = norm(view(X_proc, :, j))
        if nrm > 0
            X_proc[:, j] ./= nrm
        end
    end
    
    return X_proc, y_proc
end

function split_data(X, y; n_train, n_val)
    n = size(X, 1)
    idx = randperm(n)
    
    train_idx = idx[1:n_train]
    val_idx = idx[n_train+1:n_train+n_val]
    test_idx = idx[n_train+n_val+1:end]
    
    return X[train_idx, :], y[train_idx], X[val_idx, :], y[val_idx], X[test_idx, :], y[test_idx]
end

function make_vars_real(X_train, y_train, X_val, y_val)
    XTX = X_train'X_train
    p = size(X_train, 2)
    return X_train, y_train, X_val, y_val, XTX, p, zeros(p), 0
end

# ----------------------------------------------------------------------------
# Main Experiment Loop
# ----------------------------------------------------------------------------

function run_asspg_experiment()
    T_trials = 5
    n_probe = 100 # boston_probes configuration
    
    @info "Loading Boston Housing and generating probes..."
    X_raw, y_raw = load_boston_housing()
    X_raw = add_pairwise_interactions(X_raw)
    X_raw = add_probe_features(X_raw, n_probe)
    
    p_full = size(X_raw, 2)
    @info "Dataset ready: p = $p_full"
    
    # Store results: TestMSE, L0-norm, Time
    # Algos: SPG_Inf, AS_2, AS_4, AS_6, AS_10, PGCCD_6
    n_algos = 6
    res_mse = zeros(n_algos, T_trials)
    res_l0  = zeros(n_algos, T_trials)
    res_t   = zeros(n_algos, T_trials)
    
    for t in 1:T_trials
        @info "Trial $t/$T_trials"
        X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = 
            split_data(X_raw, y_raw; n_train=200, n_val=100)
            
        X_tr, y_tr = preprocess_real_data(X_train_raw, y_train_raw)
        X_v, y_v   = preprocess_real_data(X_val_raw, y_val_raw)
        X_te, y_te = preprocess_real_data(X_test_raw, y_test_raw)
        
        vars = make_vars_real(X_tr, y_tr, X_v, y_v)
        
        function run_eval(idx, name, solver_call)
            t0 = time()
            res = inverse_cross_validation(solver_call, vars; NSPG=true, use_refinement=false)
            β = res[1]
            elapsed = time() - t0
            
            mse = mean((y_te - X_te * β).^2)
            l0 = count(!iszero, β)
            
            res_mse[idx, t] = mse
            res_l0[idx, t] = l0
            res_t[idx, t] = elapsed
            
            @info "  [$idx] $name: MSE=$(round(mse, digits=4)), L0=$l0, time=$(round(elapsed, digits=2))s"
        end

        # 1. Standard NSPG (ActiveSetNum = Inf effectively)
        run_eval(1, "NSPG (Baseline)", 
            (x, f; kw...) -> SolverPSI1((x, f; k...) -> NSPG(x, f; k...), x, f; kw...))
            
        # 2. ActiveSetSPG (Num = 2)
        run_eval(2, "AS-NSPG (2)", 
            (x, f; kw...) -> SolverPSI1((x, f; k...) -> ActiveSetSPG(x, f; ActiveSetNum=2, k...), x, f; kw...))

        # 3. ActiveSetSPG (Num = 4)
        run_eval(3, "AS-NSPG (4)", 
            (x, f; kw...) -> SolverPSI1((x, f; k...) -> ActiveSetSPG(x, f; ActiveSetNum=4, k...), x, f; kw...))

        # 4. ActiveSetSPG (Num = 6)
        run_eval(4, "AS-NSPG (6)", 
            (x, f; kw...) -> SolverPSI1((x, f; k...) -> ActiveSetSPG(x, f; ActiveSetNum=6, k...), x, f; kw...))

        # 5. ActiveSetSPG (Num = 10)
        run_eval(5, "AS-NSPG (10)", 
            (x, f; kw...) -> SolverPSI1((x, f; k...) -> ActiveSetSPG(x, f; ActiveSetNum=10, k...), x, f; kw...))

        # 6. PGCCD (Coordinate Descent, ActiveSetNum = 6)
        run_eval(6, "PGCCD (PGCCD 6)", 
            (x, f; kw...) -> SolverPSI1((x, f; k...) -> PGCCD(x, f; ActiveSetNum=6, k...), x, f; kw...))
            
    end
    
    avg_mse = mean(res_mse, dims=2)[:]
    avg_l0  = mean(res_l0, dims=2)[:]
    avg_t   = mean(res_t, dims=2)[:]
    
    names = [
        "NSPG (Baseline)",
        "AS-NSPG (2)",
        "AS-NSPG (4)",
        "AS-NSPG (6)",
        "AS-NSPG (10)",
        "PGCCD (PGCCD 6)"
    ]
    
    results_file = "as_spg_results.txt"
    open(results_file, "w") do f
        println(f, "="^80)
        println(f, "ACTIVE SET NSPG SCALING EXPERIMENT")
        println(f, "="^80)
        println(f, @sprintf("%-25s %-15s %-10s %-10s", "Algorithm", "TestMSE", "‖β‖₀", "Time(s)"))
        println(f, "-"^80)
        for i in 1:n_algos
            println(f, @sprintf("%-25s %-15.4f %-10.1f %-10.2f", names[i], avg_mse[i], avg_l0[i], avg_t[i]))
        end
        println(f, "="^80)
    end
    
    println(read(results_file, String))
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_asspg_experiment()
end
