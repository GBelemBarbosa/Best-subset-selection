### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ f6f0822d-4c0e-4244-a31c-aba72d4288df
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
# ╠═╡ pluto_cell_id = "f6f0822d-4c0e-4244-a31c-aba72d4288df"
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using Random
	using Distributions
	using LinearAlgebra
	using BenchmarkTools, Profile, TimerOutputs
	using Plots, StatsPlots, Plots.PlotMeasures
	using LaTeXStrings
	using PlutoUI
end
  ╠═╡ =#

# ╔═╡ e5bb3d96-3982-4fa6-8418-9b9223105d8b
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "e5bb3d96-3982-4fa6-8418-9b9223105d8b"
begin
    corr = "exp"
	ρ = 0.9
	p = 1000
	SNR = 5
	k⃰ = 20
end;

# ╔═╡ 816a0d16-1fed-4eaf-808c-4dc6c67fbf72
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "816a0d16-1fed-4eaf-808c-4dc6c67fbf72"
function variables(; corr = "exp", ρ = 0.9, n = 250, p = 1000, SNR = 5, k⃰ = 20)
	Σ = corr=="exp" ? [ρ^abs(i-j) for i=1:p, j=1:p] : [1-(1-ρ)*(i!=j) for i=1:p, j=1:p]
	d = MvNormal(zeros(p), Σ)
	X = rand(d, n)'
	for i=1:p; X[:, i] /= norm(X[:, i]); end
	β⃰ = [1.0*(mod(i, Int(p/k⃰))==0) for i=1:p]

	σ = sqrt(norm(X*β⃰)^2/(n*SNR))
	
	y = X*β⃰+randn(n)*σ

	XTX = X'X

	return X, y, XTX, p, β⃰, k⃰
end;

# ╔═╡ e5e9fb82-9944-44dd-b78c-6698b9d886d7
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "e5e9fb82-9944-44dd-b78c-6698b9d886d7"
function funcs(X, y, XTX, p, β⃰, k⃰, λ₀)
	HT = sqrt(2*λ₀)
	r!(r, β) = (mul!(r, X, β); r .-= y)
	r(β) = X*β-y
	f(r) = norm(r)^2/2
	h(β) = λ₀*norm(β, 0)
	F(r, β) = f(r)+h(β)
	∇f!(∇f, r) = mul!(∇f, X', r)
	∇f(r) = X'r
	proxl0(x) = (abs(x)>=HT)*x
	proxl0(x, τ) = (abs.(x).>=sqrt(2λ₀*τ)).*x
	proxl0VM(x, Uₖ) = (abs.(x).>=sqrt.(2λ₀.*Uₖ)).*x

	return r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, X, XTX
end;

# ╔═╡ dc30376c-3641-4c94-94d0-205d5940b52b
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "dc30376c-3641-4c94-94d0-205d5940b52b"
begin
    kₘₐₓ = 1000
    ϵ = 10^-7
end

# ╔═╡ d7fbd2ce-8c0c-45d8-8757-b6a150a2b953
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "d7fbd2ce-8c0c-45d8-8757-b6a150a2b953"
function VMSPG(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64), µ=10^-3)
	r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, X, XTX = funcs
	
	sᵏ = xᵏ⁻¹ = xᵏ = x⁰
	rᵏ = r(xᵏ)
	∇fxᵏ⁻¹ = ∇fxᵏ = ∇f(rᵏ)
	Fxᵏ⁻¹ = Fxᵏ = F(rᵏ, xᵏ)
	r!(rᵏ, x⁰-∇fxᵏ*10^-5)
	yᵏ = ∇fxᵏ-∇f(rᵏ)
	yᵏTsᵏ = dot(yᵏ, ∇fxᵏ)
	nsᵏ = γₖ¹ = dot(∇fxᵏ, ∇fxᵏ)*10^-5/yᵏTsᵏ
	γₖ² = yᵏTsᵏ*10^-5/dot(yᵏ, yᵏ)
	Uₖ₋₁ = Uₖ = min.(max.(10^-5*∇fxᵏ.*∇fxᵏ./(∇fxᵏ.*yᵏ), γₖ²), γₖ¹)
	lastₘ = [Fxᵏ for i = 1:m]

	for k=1:kₘₐₓ
		Fxₗ₍ₖ₎ = maximum(lastₘ)
		
		while true
			xᵏ = proxl0VM(xᵏ⁻¹-Uₖ.*∇fxᵏ, Uₖ)
			r!(rᵏ, xᵏ)
			Fxᵏ = F(rᵏ, xᵏ)
            sᵏ = xᵏ-xᵏ⁻¹

			if Fxᵏ+δ*dot(sᵏ, sᵏ./Uₖ)/2 <= Fxₗ₍ₖ₎
                break
            end

            BLAS.scal!(τ, Uₖ)

			if any(isnan, Uₖ) || any(x -> x<γₘᵢₙ, Uₖ)
                break
            end
		end	
		
		if abs(Fxᵏ⁻¹-Fxᵏ)/Fxᵏ<=ϵ
			return xᵏ, k
		end

		popfirst!(lastₘ)
        push!(lastₘ, Fxᵏ)
		xᵏ⁻¹, Fxᵏ⁻¹ = xᵏ, Fxᵏ
		∇fxᵏ⁻¹ = copy(∇fxᵏ)
		∇f!(∇fxᵏ, rᵏ)
		nsᵏ = dot(sᵏ, sᵏ)
		yᵏ = ∇fxᵏ-∇fxᵏ⁻¹
		nyᵏ = dot(yᵏ, yᵏ)
		yᵏTsᵏ = dot(yᵏ, sᵏ)
		γₖ¹ = yᵏTsᵏ>0 ? nsᵏ/yᵏTsᵏ : sqrt(nsᵏ/nyᵏ)
		γₖ² = yᵏTsᵏ>0 ? yᵏTsᵏ/nyᵏ : 1/γₖ¹
		Uₖ₋₁, Uₖ = Uₖ, min.(max.((sᵏ.*sᵏ+µ.*Uₖ₋₁)./(sᵏ.*yᵏ.+µ), γₖ²), γₖ¹)
	end

	return xᵏ, kₘₐₓ
end;

# ╔═╡ 2ba39b49-487d-424d-b898-d4c8a76cb4f0
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "2ba39b49-487d-424d-b898-d4c8a76cb4f0"
function SPGH(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64))
	r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, X, XTX = funcs

	sᵏ = xᵏ⁻¹ = xᵏ = x⁰
	rᵏ = r(xᵏ)
	∇fxᵏ⁻¹ = ∇fxᵏ = ∇f(rᵏ)
	Fxᵏ⁻¹ = Fxᵏ = F(rᵏ, xᵏ)
	r!(rᵏ, x⁰-∇fxᵏ*10^-5)
	yᵏ = ∇fxᵏ-∇f(rᵏ)
	yᵏTsᵏ = dot(yᵏ, ∇fxᵏ)
	nsᵏ = γₖ¹ = dot(∇fxᵏ, ∇fxᵏ)*10^-5/yᵏTsᵏ
	γₖ² = yᵏTsᵏ*10^-5/dot(yᵏ, yᵏ)
	γₖ = γₖ¹<2*γₖ² ? γₖ² : γₖ¹-γₖ²/2
	lastₘ = [Fxᵏ for i = 1:m]

	for k=1:kₘₐₓ
		Fxₗ₍ₖ₎ = maximum(lastₘ)
		
		while true
			xᵏ = proxl0(xᵏ⁻¹-γₖ*∇fxᵏ, γₖ)
			r!(rᵏ, xᵏ)
			Fxᵏ = F(rᵏ, xᵏ)
            sᵏ = xᵏ-xᵏ⁻¹
			nsᵏ = dot(sᵏ, sᵏ)

			if Fxᵏ+δ*nsᵏ/(2*γₖ) <= Fxₗ₍ₖ₎
                break
            end

            γₖ *= τ

			if isnan(γₖ) || γₖ < γₘᵢₙ
                break
            end
		end	
		
		if abs(Fxᵏ⁻¹-Fxᵏ)/Fxᵏ<=ϵ
			return xᵏ, k
		end

		popfirst!(lastₘ)
        push!(lastₘ, Fxᵏ)
		xᵏ⁻¹, Fxᵏ⁻¹ = xᵏ, Fxᵏ
		∇fxᵏ⁻¹ = copy(∇fxᵏ)
		∇f!(∇fxᵏ, rᵏ)
		yᵏ = ∇fxᵏ-∇fxᵏ⁻¹
		nyᵏ = dot(yᵏ, yᵏ)
		yᵏTsᵏ = dot(yᵏ, sᵏ)
		γₖ¹ = yᵏTsᵏ>0 ? nsᵏ/yᵏTsᵏ : sqrt(nsᵏ/nyᵏ)
		γₖ² = yᵏTsᵏ>0 ? yᵏTsᵏ/nyᵏ : 1/γₖ¹
        γₖ = γₖ¹<2*γₖ² ? γₖ² : γₖ¹-γₖ²/2
	end

	return xᵏ, kₘₐₓ
end;

# ╔═╡ 196b60fd-6d2e-4083-867f-844b3cdd187d
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "196b60fd-6d2e-4083-867f-844b3cdd187d"
function SPG(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64))
	r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, X, XTX = funcs

	sᵏ = xᵏ⁻¹ = xᵏ = x⁰
	rᵏ = r(xᵏ)
	∇fxᵏ⁻¹ = ∇fxᵏ = ∇f(rᵏ)
	Fxᵏ⁻¹ = Fxᵏ = F(rᵏ, xᵏ)
	r!(rᵏ, x⁰-∇fxᵏ*10^-5)
	nsᵏ = γₖ = dot(∇fxᵏ, ∇fxᵏ)*10^-5/dot(∇fxᵏ-∇f(rᵏ), ∇fxᵏ)
	lastₘ = [Fxᵏ for i = 1:m]

	for k=1:kₘₐₓ
		Fxₗ₍ₖ₎ = maximum(lastₘ)
		
		while true
			xᵏ = proxl0(xᵏ⁻¹-γₖ*∇fxᵏ, γₖ)
			r!(rᵏ, xᵏ)
			Fxᵏ = F(rᵏ, xᵏ)
            sᵏ = xᵏ-xᵏ⁻¹
			nsᵏ = dot(sᵏ, sᵏ)

			if Fxᵏ+δ*nsᵏ/(2*γₖ) <= Fxₗ₍ₖ₎
                break
            end

            γₖ *= τ

			if isnan(γₖ) || γₖ < γₘᵢₙ
                break
            end
		end	
		
		if abs(Fxᵏ⁻¹-Fxᵏ)/Fxᵏ<=ϵ
			return xᵏ, k
		end

		popfirst!(lastₘ)
        push!(lastₘ, Fxᵏ)
		xᵏ⁻¹, Fxᵏ⁻¹ = xᵏ, Fxᵏ
		∇fxᵏ⁻¹ = copy(∇fxᵏ)
		∇f!(∇fxᵏ, rᵏ)
		
		yᵏ = ∇fxᵏ-∇fxᵏ⁻¹
		γₖ = nsᵏ/dot(∇fxᵏ-∇fxᵏ⁻¹, sᵏ)
        if γₖ > γₘₐₓ || γₖ < γₘᵢₙ
            γₖ = sqrt(nsᵏ/dot(yᵏ, yᵏ))
        end
	end

	return xᵏ, kₘₐₓ
end;

# ╔═╡ 3c06123a-f43e-4df5-8f28-838ee68b9712
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "3c06123a-f43e-4df5-8f28-838ee68b9712"
function CDSS(x⁰, funcs; sortperc=1/4)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, X, XTX = funcs

    xᵏ = copy(x⁰)
    
    rᵏ = -r(xᵏ)      

    Fxᵏ⁻¹ = F(rᵏ, xᵏ)
    
    ksort = round(Int64, length(x⁰)*sortperc)
    greedy = partialsortperm(abs.(∇f(rᵏ)), 1:ksort, rev=true)
    greedy = vcat(greedy, setdiff(1:p, greedy))

    for k=1:kₘₐₓ
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
end;

# ╔═╡ 737b576a-dc16-4c1e-ab15-bd8f193f77f4
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "737b576a-dc16-4c1e-ab15-bd8f193f77f4"
function SPGpCDSS(x⁰, funcs)
	x, k = SPG(x⁰, funcs)
	x, k2 = CDSS(x, funcs)
	
	return x, k+k2
end;

# ╔═╡ 740c7450-a0c6-4f9e-878a-6d383e184671
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "740c7450-a0c6-4f9e-878a-6d383e184671"
function PSI1(xˡ, funcs)
	r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, X, XTX = funcs

	r⃰ = -X'r(xˡ) 
	
	for i=findall(!iszero, xˡ)
		jₘₐₓ = 0
		v⃰ₘₐₓ = 0.0
		
		for j=findall(iszero, xˡ)
			v⃰ = proxl0(r⃰[j]+XTX[i, j]*xˡ[i])

			if abs(v⃰)>abs(v⃰ₘₐₓ)
				v⃰ₘₐₓ = v⃰
				jₘₐₓ = j
			end
		end
		
		if abs(v⃰ₘₐₓ)>abs(xˡ[i])
			xˡ[i] = 0.0
			xˡ[jₘₐₓ] = v⃰ₘₐₓ
			
			return xˡ, false
		end
	end

	return xˡ, true
end;

# ╔═╡ 96eab8b9-4f88-4575-99e6-3d032d76dddd
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "96eab8b9-4f88-4575-99e6-3d032d76dddd"
function SolverPSI1(solver, x⁰, funcs)
	β = x⁰
	kᵢ = kₒ = 0
	isPSI1 = false
	
	while !isPSI1 && kₒ<kₘₐₓ
		kₒ += 1
		β, k = solver(β, funcs)
		kᵢ += k
		β, isPSI1 = PSI1(β, funcs)
	end

	return β, kᵢ, kₒ
end;

# ╔═╡ 0467f015-bd57-4379-b192-367a5287d31c
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "0467f015-bd57-4379-b192-367a5287d31c"
function cross_validation(solver, vars; λ_grid = [10^i for i in range(1, stop=-3, length=20)], k_folds=5)
    X, y, XTX, p, β⃰, k⃰ = vars
    suppsim(β) = count(i -> !iszero(β⃰[i]) && !iszero(β[i]), 1:p)/max(k⃰, norm(β, 0))
	pred(β) = norm(X*β.-y)^2/norm(y)^2
    n_samples = size(X, 1)
    indices = shuffle(1:n_samples)
    fold_size = div(n_samples, k_folds)
    
    mse_results = zeros(length(λ_grid))
    β_fold = zeros(p)

    for (i, λ) in enumerate(λ_grid)
        fold_mses = Float64[]
        
        for k in 1:k_folds
            # Split data
            test_idx = indices[(k-1)*fold_size + 1 : min(k*fold_size, n_samples)]
            train_idx = setdiff(indices, test_idx)
            
            X_train, y_train = X[train_idx, :], y[train_idx]
            X_test, y_test = X[test_idx, :], y[test_idx]
            
            # Run solver
            β_fold, kᵢ, kₒ = solver(β_fold, funcs(X_train, y_train, X_train'X_train, p, β⃰, k⃰, λ))
            
            # Calculate Error
            push!(fold_mses, norm(y_test .- X_test * β_fold))
        end
        
        mse_results[i] = mean(fold_mses)
    end

    best_idx = argmin(mse_results)
    best_λ = λ_grid[best_idx]
    β, kᵢ, kₒ = solver(zeros(p), funcs(X, y, XTX, p, β⃰, k⃰, best_λ))
    
    return β, best_λ, norm(β, 0), pred(β), suppsim(β), norm(β-β⃰, Inf)
end;

# ╔═╡ 0d883a10-c092-4913-8e60-59a164f8069b
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "0d883a10-c092-4913-8e60-59a164f8069b"
begin
    x⁰ = zeros(Float64, p)
    ns=1000:100:1000;
    T=5
end;

# ╔═╡ 81ec0ae9-0d55-4549-80b7-fa5039560527
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "81ec0ae9-0d55-4549-80b7-fa5039560527"
begin
	Predhist = zeros(length(ns), 3)
	SUPhist = zeros(length(ns), 3)
	Infhist = zeros(length(ns), 3)
	Simhist = zeros(length(ns), 3)
end;

# ╔═╡ e4e94a69-5f99-43cb-85f8-ed021d6ae802
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "e4e94a69-5f99-43cb-85f8-ed021d6ae802"
for (t, n)=enumerate(ns)	
	for i=1:T
		vars = variables(corr = corr, ρ = ρ, n = n, p = p, SNR = SNR, k⃰ = k⃰)

		β, best_λ, SUP, Pred, Sim, Infv = cross_validation((x, f)->SolverPSI1(CDSS, x, f), vars)
		SUPhist[t, 1] += SUP
		Predhist[t, 1] += Pred
		Simhist[t, 1] += Sim
		Infhist[t, 1] += Infv

		β, best_λ, SUP, Pred, Sim, Infv = cross_validation((x, f)->SolverPSI1(SPG, x, f), vars)
		SUPhist[t, 2] += SUP
		Predhist[t, 2] += Pred
		Simhist[t, 2] += Sim
		Infhist[t, 2] += Infv

		β, best_λ, SUP, Pred, Sim, Infv = cross_validation((x, f)->SolverPSI1(SPGpCDSS, x, f), vars)
		SUPhist[t, 3] += SUP
		Predhist[t, 3] += Pred
		Simhist[t, 3] += Sim
		Infhist[t, 3] += Infv

		println(n, ": ", t)
	end

	SUPhist ./= T
	Predhist ./= T
	Simhist ./= T
	Infhist ./= T
end

# ╔═╡ 29fdb407-e14e-4f09-833d-59741b7cd580
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "29fdb407-e14e-4f09-833d-59741b7cd580"
Simhist

# ╔═╡ 26d22c97-3636-4821-a6cf-f9958f09a3b1
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "26d22c97-3636-4821-a6cf-f9958f09a3b1"
names =  ["Greedy CD" "NSPG" "NSPG+CD"];

# ╔═╡ 219e1960-8477-48c5-942a-e05d8827369f
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "219e1960-8477-48c5-942a-e05d8827369f"
plotname = "$corr-$ρ-$p-$SNR-$k⃰";

# ╔═╡ 92605860-0a44-428a-92f0-658e5d50ea00
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "92605860-0a44-428a-92f0-658e5d50ea00"
pPred = plot(ns, Predhist, labels=names, xlabel=L"n", ylabel=L"\frac{\Vert Ax-b\Vert^2}{\Vert b\Vert^2}", left_margin=5mm, dpi=600)

# ╔═╡ e0d6b923-0bea-4afe-b00e-6f4cc1a8cbbe
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "e0d6b923-0bea-4afe-b00e-6f4cc1a8cbbe"
DownloadButton(pPred, "pnPred-$plotname.png")

# ╔═╡ f3eab76f-2284-4eea-b1ab-d87daa392b8d
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "f3eab76f-2284-4eea-b1ab-d87daa392b8d"
pSim = plot(ns, Simhist, labels=names, xlabel=L"n", ylabel=L"\frac{|Supp(x)\cap Supp(x^\dagger)|}{\max\{|Supp(x)|,k^\dagger\}}", left_margin=5mm, dpi=600)

# ╔═╡ 4a495ee7-f90f-49a3-bcce-08176e1caef9
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "4a495ee7-f90f-49a3-bcce-08176e1caef9"
DownloadButton(pSim, "pnSUP-$plotname.png")

# ╔═╡ d68d2ff1-fd2c-44cd-8106-db28eaed6bef
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "d68d2ff1-fd2c-44cd-8106-db28eaed6bef"
pInf = plot(ns, Infhist, labels=names, xlabel=L"n", ylabel=L"\|x-x^\dagger\|_\infty", dpi=600)

# ╔═╡ 1641f1b6-cbc7-4b34-a1dd-930ff74e85f3
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "1641f1b6-cbc7-4b34-a1dd-930ff74e85f3"
DownloadButton(pInf, "pnInf-$plotname.png")

# ╔═╡ 3a13d7d2-a95b-423e-a580-31c7b2263305
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "3a13d7d2-a95b-423e-a580-31c7b2263305"
pSUP = plot(ns, SUPhist, labels=names, xlabel=L"n", ylabel=L"\|x\|_0", dpi=600)

# ╔═╡ ea769181-b723-41ce-81bb-933187859ee6
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "ea769181-b723-41ce-81bb-933187859ee6"
DownloadButton(pSUP, "pnSim-$plotname.png")

# ╔═╡ Cell order:
# ╠═f6f0822d-4c0e-4244-a31c-aba72d4288df
# ╠═e5bb3d96-3982-4fa6-8418-9b9223105d8b
# ╠═816a0d16-1fed-4eaf-808c-4dc6c67fbf72
# ╠═e5e9fb82-9944-44dd-b78c-6698b9d886d7
# ╠═dc30376c-3641-4c94-94d0-205d5940b52b
# ╠═d7fbd2ce-8c0c-45d8-8757-b6a150a2b953
# ╠═2ba39b49-487d-424d-b898-d4c8a76cb4f0
# ╠═196b60fd-6d2e-4083-867f-844b3cdd187d
# ╠═3c06123a-f43e-4df5-8f28-838ee68b9712
# ╠═737b576a-dc16-4c1e-ab15-bd8f193f77f4
# ╠═740c7450-a0c6-4f9e-878a-6d383e184671
# ╠═96eab8b9-4f88-4575-99e6-3d032d76dddd
# ╠═0467f015-bd57-4379-b192-367a5287d31c
# ╠═81ec0ae9-0d55-4549-80b7-fa5039560527
# ╠═0d883a10-c092-4913-8e60-59a164f8069b
# ╠═e4e94a69-5f99-43cb-85f8-ed021d6ae802
# ╠═29fdb407-e14e-4f09-833d-59741b7cd580
# ╠═26d22c97-3636-4821-a6cf-f9958f09a3b1
# ╠═219e1960-8477-48c5-942a-e05d8827369f
# ╠═92605860-0a44-428a-92f0-658e5d50ea00
# ╠═e0d6b923-0bea-4afe-b00e-6f4cc1a8cbbe
# ╠═f3eab76f-2284-4eea-b1ab-d87daa392b8d
# ╠═4a495ee7-f90f-49a3-bcce-08176e1caef9
# ╠═d68d2ff1-fd2c-44cd-8106-db28eaed6bef
# ╠═1641f1b6-cbc7-4b34-a1dd-930ff74e85f3
# ╠═3a13d7d2-a95b-423e-a580-31c7b2263305
# ╠═ea769181-b723-41ce-81bb-933187859ee6
