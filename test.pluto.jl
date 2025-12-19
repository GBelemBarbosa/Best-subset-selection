### A Pluto.jl notebook ###
# v0.20.10

using Markdown
using InteractiveUtils

# ╔═╡ 614ae2e5-173e-49cf-b259-8e7716aa8f75
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "614ae2e5-173e-49cf-b259-8e7716aa8f75"
begin
	using Random
	using Distributions
	using LinearAlgebra
	using BenchmarkTools, Profile, TimerOutputs
	using Plots, StatsPlots, Plots.PlotMeasures
	using LaTeXStrings
	using PlutoUI
end

# ╔═╡ e8d86478-ca40-4546-beea-97d1374032e0
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "e8d86478-ca40-4546-beea-97d1374032e0"
begin
	corr = "exp"
	ρ = 0.9
	n = 250
	p = 1000
	SNR = 300
	k⃰ = 50
	
	Σ = corr=="exp" ? [ρ^abs(i-j) for i=1:p, j=1:p] : [1-(1-ρ)*(i!=j) for i=1:p, j=1:p]
	d = MvNormal(zeros(p), Σ)
	X = rand(d, n)'
	for i=1:p; X[:, i] /= norm(X[:, i]); end
	β⃰ = [1.0*(mod(i, Int(p/k⃰))==0) for i=1:p]

	σ = sqrt(norm(X*β⃰)^2/(n*SNR))
	
	y = X*β⃰+randn(n)*σ

	XTX = X'X
end;

# ╔═╡ 82cf2bcc-b00b-42c4-9039-5c4edace3dc1
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "82cf2bcc-b00b-42c4-9039-5c4edace3dc1"
norm(X*β⃰)^2/norm(y-X*β⃰)^2, var(X*β⃰)/var(y-X*β⃰)

# ╔═╡ 7124ee0a-799a-4dbe-b3c6-40ec6d7c4094
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "7124ee0a-799a-4dbe-b3c6-40ec6d7c4094"
function funcs()
	λ₀ = 0.5*norm(X'y, Inf)^2/(2*eigmax(X*X'))
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

	return r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, λ₀
end;

# ╔═╡ 313e3507-60af-4090-9d2a-2d49c0a66415
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "313e3507-60af-4090-9d2a-2d49c0a66415"
r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, λ₀ = funcs();

# ╔═╡ 4ad60b75-e3d1-4892-9a16-90f972b7bc6e
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "4ad60b75-e3d1-4892-9a16-90f972b7bc6e"
begin
    kₘₐₓ = 1000
    ϵ = 10^-7
end

# ╔═╡ fc6a6b8f-158f-4bcc-90fc-625fab76896b
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "fc6a6b8f-158f-4bcc-90fc-625fab76896b"
function VMSPG(x⁰; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64), µ=10^-3)
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

	return xᵏ
end;

# ╔═╡ acfb1cad-0acf-4af0-b800-0308314d0883
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "acfb1cad-0acf-4af0-b800-0308314d0883"
function SPGH(x⁰; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64))
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

	return xᵏ
end;

# ╔═╡ 9b27d9ec-aac7-4553-bd87-229fe21dd8d6
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "9b27d9ec-aac7-4553-bd87-229fe21dd8d6"
function SPG(x⁰; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64))
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

	return xᵏ
end;

# ╔═╡ abe2dd28-8db4-4e4c-b30f-ede14c269628
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "abe2dd28-8db4-4e4c-b30f-ede14c269628"
begin
    function CDSS(x⁰; p=p, X=X, ksort=round(Int64, p/4))
        xᵏ = copy(x⁰)
        
        rᵏ = -r(xᵏ) 
    
        Fxᵏ⁻¹ = F(rᵏ, xᵏ)
        
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
    
        return xᵏ
    end
end;

# ╔═╡ bdc1ee58-b1a7-4294-86a4-c0491f6ce0a9
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "bdc1ee58-b1a7-4294-86a4-c0491f6ce0a9"
begin
    if !(@isdefined tmr)
        tmr = TimerOutput()
    end

    function CDSStimeit(x⁰)
    
        ksort=round(Int64, p/4)
        @timeit tmr "total" begin
            xᵏ = copy(x⁰)
            rᵏ = -r(xᵏ)
            Fxᵏ⁻¹ = F(rᵏ, xᵏ)
    
            @timeit tmr "greedy" begin
                    greedy = partialsortperm(abs.(∇f(rᵏ)), 1:ksort, rev=true)
                    greedy = vcat(greedy, setdiff(1:p, greedy))
            end
    
            for k=1:kₘₐₓ
                @inbounds for i in greedy
                    @timeit tmr "prox" xi = proxl0(dot(rᵏ, view(X, :, i)) + xᵏ[i])
                    if xi != xᵏ[i]
                        @timeit tmr "blas" BLAS.axpy!(xᵏ[i] - xi, view(X, :, i), rᵏ)
                        xᵏ[i] = xi
                    end
                end
    
                @timeit tmr "Feval" Fxᵏ = F(rᵏ, xᵏ)
                if (Fxᵏ⁻¹ - Fxᵏ) / Fxᵏ <= ϵ
                    return xᵏ, k
                end
                Fxᵏ⁻¹ = Fxᵏ
            end
    
            return xᵏ
        end
    end
end;

# ╔═╡ 8f02cd37-7d93-4f1b-9642-514aad3260a5
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "8f02cd37-7d93-4f1b-9642-514aad3260a5"
βCD, kCD = @btime CDSS(zeros(p))

# ╔═╡ 96b63c3f-9def-4507-a9df-e7ebf9ad97c7
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "96b63c3f-9def-4507-a9df-e7ebf9ad97c7"
CDSStimeit(zeros(p))

# ╔═╡ 816ee8eb-cbd9-4be9-bb68-120ee2fba6b2
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "816ee8eb-cbd9-4be9-bb68-120ee2fba6b2"
show(tmr)

# ╔═╡ 641e6c7f-3ba6-47e6-ba7a-dc8d1ffb0221
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "641e6c7f-3ba6-47e6-ba7a-dc8d1ffb0221"
begin
    Profile.clear()
    @profile CDSS(zeros(p))
    Profile.print()
end

# ╔═╡ 2512d879-ac75-4b3b-ab22-1fb3bd843570
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "2512d879-ac75-4b3b-ab22-1fb3bd843570"
analyze(βCD)

# ╔═╡ 2bf50f74-3100-4ae3-9b22-ca6e0e194723
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "2bf50f74-3100-4ae3-9b22-ca6e0e194723"
βSPG, kSPG = @btime SPG(zeros(p))

# ╔═╡ 6fc6f520-d68d-40f9-9c0f-d30414a7e1cd
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "6fc6f520-d68d-40f9-9c0f-d30414a7e1cd"
analyze(βSPG)

# ╔═╡ 71fced6f-ac26-433c-972a-981a37cb34a5
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "71fced6f-ac26-433c-972a-981a37cb34a5"
βSPGH, kSPGH = @btime SPGH(zeros(p))

# ╔═╡ 8fa2a124-8083-42b1-93c8-2694e9e1ea10
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "8fa2a124-8083-42b1-93c8-2694e9e1ea10"
analyze(βSPGH)

# ╔═╡ c3193363-8be5-4e52-b8f2-25a9948784a5
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "c3193363-8be5-4e52-b8f2-25a9948784a5"
βVMSPG, kVMSPG = @btime VMSPG(zeros(p))

# ╔═╡ 271da144-5cc0-4dfe-ac8a-419e17b9ca98
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "271da144-5cc0-4dfe-ac8a-419e17b9ca98"
T=1;

# ╔═╡ 0c03927b-157e-4a04-8298-c42f779882ef
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "0c03927b-157e-4a04-8298-c42f779882ef"
begin
	Fhist = zeros(T, 4)
	SUPhist = zeros(T, 4)
	Khist = zeros(T, 4)
	Kouthist = zeros(T, 4)
	CWhist = Matrix{Tuple{Int64, Bool}}(undef, T, 3)
end;

# ╔═╡ 52c96374-c293-498b-9583-bbfcc23ef69e
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "52c96374-c293-498b-9583-bbfcc23ef69e"
#VSCODE-MARKDOWN
md"""for t=1:T
	x⁰ = zeros(Float64, p)
	supp = randperm(p)[1:k⃰]
	x⁰[supp] = rand(Uniform(0,1), k⃰)

	β, k = CDSS(x⁰)
	SUPhist[t, 1], Fhist[t, 1], Khist[t, 1] = suppsim(β), F(r(β), β), k
	
	β, k = SPG(x⁰)
	SUPhist[t, 2], Fhist[t, 2], Khist[t, 2] = suppsim(β), F(r(β), β), k
	CWhist[t, 1] = isCW(β)
	
	β, k = SPGH(x⁰)
	SUPhist[t, 3], Fhist[t, 3], Khist[t, 3] = suppsim(β), F(r(β), β), k
	CWhist[t, 2] = isCW(β)
	
	β, k = VMSPG(x⁰)
	SUPhist[t, 4], Fhist[t, 4], Khist[t, 4] = suppsim(β), F(r(β), β), k
	CWhist[t, 3] = isCW(β)
end"""

# ╔═╡ 31cc28a8-c998-4648-9f84-e258c8ca107b
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "31cc28a8-c998-4648-9f84-e258c8ca107b"
function SPGpCDSS(x⁰)
	x, k = SPG(x⁰)
	x, k2 = CDSS(x)
	
	return x, k+k2
end;

# ╔═╡ eee04e89-4e9a-40f6-9e28-2d4def2f6474
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "eee04e89-4e9a-40f6-9e28-2d4def2f6474"
for t=1:T
	x⁰ = zeros(Float64, p)
	supp = randperm(p)[1:k⃰]
	x⁰[supp] = rand(Uniform(0,1), k⃰)

	β, kᵢ, kₒ  = SolverPSI1(CDSS, x⁰)
	SUPhist[t, 1], Fhist[t, 1], Khist[t, 1], Kouthist[t, 1] = suppsim(β), F(r(β), β), kᵢ, kₒ
	
	β, kᵢ, kₒ = SolverPSI1(SPG, x⁰)
	SUPhist[t, 2], Fhist[t, 2], Khist[t, 2], Kouthist[t, 2] = suppsim(β), F(r(β), β), kᵢ, kₒ
	CWhist[t, 1] = isCW(β)
	
	β, kᵢ, kₒ = SolverPSI1(SPGpCDSS, x⁰)
	SUPhist[t, 3], Fhist[t, 3], Khist[t, 3], Kouthist[t, 3] = suppsim(β), F(r(β), β), kᵢ, kₒ
	CWhist[t, 2] = isCW(β)

	β, k = SPG(x⁰)
	SUPhist[t, 4], Fhist[t, 4], Khist[t, 4] = suppsim(β), F(r(β), β), k
	CWhist[t, 3] = isCW(β)
end

# ╔═╡ 360dca67-11d3-496a-9a10-8c0213a2f7ae
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "360dca67-11d3-496a-9a10-8c0213a2f7ae"
#VSCODE-MARKDOWN
md"""for t=1:T
	x⁰ = zeros(Float64, p)
	supp = randperm(p)[1:k⃰]
	x⁰[supp] = rand(Uniform(0,1), k⃰)

	β, k = VMSPG(x⁰, μ=10^-5)
	SUPhist[t, 1], Fhist[t, 1], Khist[t, 1] = suppsim(β), F(r(β), β), k
	β, k = VMSPG(x⁰, μ=10^-4)
	SUPhist[t, 2], Fhist[t, 2], Khist[t, 2] = suppsim(β), F(r(β), β), k
	β, k = VMSPG(x⁰, μ=10^-3)
	SUPhist[t, 3], Fhist[t, 3], Khist[t, 3] = suppsim(β), F(r(β), β), k
	β, k = VMSPG(x⁰, μ=10^-2)
	SUPhist[t, 4], Fhist[t, 4], Khist[t, 4] = suppsim(β), F(r(β), β), k
end;"""

# ╔═╡ 363fe380-53ac-40f4-9c6a-1605f9660925
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "363fe380-53ac-40f4-9c6a-1605f9660925"
begin
	m=2
	bars=[count(j->CWhist[j, m][1]==i && CWhist[j, m][2]==l, 1:T) for i=1:maximum(CWhist[k, m][1] for k=1:T), l=[true, false]]

	pCW = groupedbar(bars, groups=repeat(["CW support", "non-CW support"], inner = size(bars, 1)), xlabel="Iterations of greedy CD", ylabel="Ocurrances")
end

# ╔═╡ ba64459a-a538-42c8-8cfd-bd8496678f61
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "ba64459a-a538-42c8-8cfd-bd8496678f61"
count(j->CWhist[j, 3][1]==1 && CWhist[j, 3][2]==true, 1:T)

# ╔═╡ c4f485e8-92b4-477d-b203-e113216a0a42
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "c4f485e8-92b4-477d-b203-e113216a0a42"
names =  ["Greedy CD" "NSPG" "NSPG+CD" "NSPG (no CDPSI)"]; #["Greedy CD" "NSPG" "NSPGH" "VMNSPG"]

# ╔═╡ 3ede3346-bbdb-4315-8f8a-46cb11fb3e72
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "3ede3346-bbdb-4315-8f8a-46cb11fb3e72"
plotname = "$corr-$ρ-$n-$p-$SNR-$k⃰";

# ╔═╡ 612f0450-4aa3-45d9-8c6f-bc93b95a8353
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "612f0450-4aa3-45d9-8c6f-bc93b95a8353"
DownloadButton(pCW, "pCW-$plotname.png")

# ╔═╡ 195b6a97-1af8-48cb-97b9-c6003a293040
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "195b6a97-1af8-48cb-97b9-c6003a293040"
pF = boxplot(names, Fhist, legend=false, ylabel=L"F(x)", dpi=600)

# ╔═╡ da43a978-f220-4014-b1e7-06ade9ccf56d
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "da43a978-f220-4014-b1e7-06ade9ccf56d"
DownloadButton(pF, "pF-$plotname.png")

# ╔═╡ 28cdc111-ca92-4f4d-b5d5-50c0d1d790e8
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "28cdc111-ca92-4f4d-b5d5-50c0d1d790e8"
pSUP = boxplot(names, SUPhist, legend=false, ylabel=L"\frac{|Supp(x)\cap Supp(x^\dagger)|}{\max\{|Supp(x)|,k^\dagger\}}", left_margin=5mm, dpi=600)

# ╔═╡ eb921868-a303-4285-bdde-d65834c25c76
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "eb921868-a303-4285-bdde-d65834c25c76"
DownloadButton(pSUP, "pSUP-$plotname.png")

# ╔═╡ 9b0e6a51-02e1-4243-818a-31d07dac5242
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "9b0e6a51-02e1-4243-818a-31d07dac5242"
pK = boxplot(names, Khist, legend=false, ylabel="Iterations", dpi=600)

# ╔═╡ 82597748-160f-4dd2-9df4-5b51c869adc9
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "82597748-160f-4dd2-9df4-5b51c869adc9"
mean.(Fhist[:,i] for i=1:4)

# ╔═╡ 394b9915-b88a-4782-9edb-5a27308991a8
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "394b9915-b88a-4782-9edb-5a27308991a8"
DownloadButton(pK, "pK-$plotname.png")

# ╔═╡ 23e2f945-8b51-434c-9199-4a9ebd491b1f
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "23e2f945-8b51-434c-9199-4a9ebd491b1f"
pKout = boxplot(names, Kouthist, legend=false, ylabel="Outer iterations", dpi=600)

# ╔═╡ 11b83403-fd56-49d3-8d9e-fd5e816d86a9
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "11b83403-fd56-49d3-8d9e-fd5e816d86a9"
DownloadButton(pKout, "pKout-$plotname.png")

# ╔═╡ 278559fb-a712-406a-b4da-2e01a113319b
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "278559fb-a712-406a-b4da-2e01a113319b"
mean.(Khist[:, i] for i=1:4)

# ╔═╡ 4e0f2ca1-eff7-455e-a4c5-346820749e0e
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "4e0f2ca1-eff7-455e-a4c5-346820749e0e"
function SolverPSI1(solver, x⁰)
	β = x⁰
	kᵢ = kₒ = 0
	isPSI1 = false
	
	while !isPSI1 && kₒ<kₘₐₓ
		kₒ += 1
		β, k = solver(β)
		kᵢ += k
		β, isPSI1 = PSI1(β)
	end

	return β, kᵢ, kₒ
end;

# ╔═╡ 0c3b0c8f-edfb-4e7f-aa66-983ccf6c8dc3
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "0c3b0c8f-edfb-4e7f-aa66-983ccf6c8dc3"
function PSI1(xˡ)
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

# ╔═╡ 86da3fad-1417-413a-9ccc-8ed231e6f9d9
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "86da3fad-1417-413a-9ccc-8ed231e6f9d9"
suppsim(β) = count(i -> !iszero(β⃰[i]) && !iszero(β[i]), 1:p)/max(k⃰, norm(β, 0));

# ╔═╡ d089557d-5e1a-47a4-b6c7-bdfb3ef4573e
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "d089557d-5e1a-47a4-b6c7-bdfb3ef4573e"
analyze(β) = @show F(r(β), β) f(r(β)) norm(β, 0) suppsim(β) nothing;

# ╔═╡ afc7374a-065f-4e92-9717-96b1659b142b
# ╠═╡ show_logs = false
# ╠═╡ pluto_cell_id = "afc7374a-065f-4e92-9717-96b1659b142b"
function isCW(β)
	CW, k = CDSS(β)
	Fβ = F(r(β), β)
	return k, !any(i -> iszero(β[i]*CW[i]) && (!iszero(β[i]) || !iszero(CW[i])), 1:p)
end;

# ╔═╡ Cell order:
# ╠═614ae2e5-173e-49cf-b259-8e7716aa8f75
# ╠═e8d86478-ca40-4546-beea-97d1374032e0
# ╠═82cf2bcc-b00b-42c4-9039-5c4edace3dc1
# ╠═7124ee0a-799a-4dbe-b3c6-40ec6d7c4094
# ╠═313e3507-60af-4090-9d2a-2d49c0a66415
# ╠═4ad60b75-e3d1-4892-9a16-90f972b7bc6e
# ╠═fc6a6b8f-158f-4bcc-90fc-625fab76896b
# ╠═acfb1cad-0acf-4af0-b800-0308314d0883
# ╠═9b27d9ec-aac7-4553-bd87-229fe21dd8d6
# ╠═abe2dd28-8db4-4e4c-b30f-ede14c269628
# ╠═bdc1ee58-b1a7-4294-86a4-c0491f6ce0a9
# ╠═8f02cd37-7d93-4f1b-9642-514aad3260a5
# ╠═96b63c3f-9def-4507-a9df-e7ebf9ad97c7
# ╠═816ee8eb-cbd9-4be9-bb68-120ee2fba6b2
# ╠═641e6c7f-3ba6-47e6-ba7a-dc8d1ffb0221
# ╠═2512d879-ac75-4b3b-ab22-1fb3bd843570
# ╠═2bf50f74-3100-4ae3-9b22-ca6e0e194723
# ╠═6fc6f520-d68d-40f9-9c0f-d30414a7e1cd
# ╠═71fced6f-ac26-433c-972a-981a37cb34a5
# ╠═8fa2a124-8083-42b1-93c8-2694e9e1ea10
# ╠═c3193363-8be5-4e52-b8f2-25a9948784a5
# ╠═271da144-5cc0-4dfe-ac8a-419e17b9ca98
# ╠═0c03927b-157e-4a04-8298-c42f779882ef
# ╠═52c96374-c293-498b-9583-bbfcc23ef69e
# ╠═31cc28a8-c998-4648-9f84-e258c8ca107b
# ╠═eee04e89-4e9a-40f6-9e28-2d4def2f6474
# ╠═360dca67-11d3-496a-9a10-8c0213a2f7ae
# ╠═363fe380-53ac-40f4-9c6a-1605f9660925
# ╠═ba64459a-a538-42c8-8cfd-bd8496678f61
# ╠═c4f485e8-92b4-477d-b203-e113216a0a42
# ╠═3ede3346-bbdb-4315-8f8a-46cb11fb3e72
# ╠═612f0450-4aa3-45d9-8c6f-bc93b95a8353
# ╠═195b6a97-1af8-48cb-97b9-c6003a293040
# ╠═da43a978-f220-4014-b1e7-06ade9ccf56d
# ╠═28cdc111-ca92-4f4d-b5d5-50c0d1d790e8
# ╠═eb921868-a303-4285-bdde-d65834c25c76
# ╠═9b0e6a51-02e1-4243-818a-31d07dac5242
# ╠═82597748-160f-4dd2-9df4-5b51c869adc9
# ╠═394b9915-b88a-4782-9edb-5a27308991a8
# ╠═23e2f945-8b51-434c-9199-4a9ebd491b1f
# ╠═11b83403-fd56-49d3-8d9e-fd5e816d86a9
# ╠═278559fb-a712-406a-b4da-2e01a113319b
# ╠═4e0f2ca1-eff7-455e-a4c5-346820749e0e
# ╠═0c3b0c8f-edfb-4e7f-aa66-983ccf6c8dc3
# ╠═86da3fad-1417-413a-9ccc-8ed231e6f9d9
# ╠═d089557d-5e1a-47a4-b6c7-bdfb3ef4573e
# ╠═afc7374a-065f-4e92-9717-96b1659b142b
