
function organize()
    src = "tmp_download_rerun"
    dest_base = "plots/CV comp"
    
    !isdir(dest_base) && mkpath(dest_base)
    
    files = readdir(src)
    for f in files
        if endswith(f, ".png")
            # Comp_Inf-L0LearnPSI1_Comparison_const-0.5-2000-10.0-100.png
            m = match(r"Comp_[^-]+-([^_]+)_Comparison_(.+)\.png", f)
            if m !== nothing
                algo = m.captures[1]
                params = m.captures[2]
                
                algo_dir = joinpath(dest_base, algo)
                param_dir = joinpath(algo_dir, params)
                mkpath(param_dir)
                
                # Directly move/overwrite plots
                mv(joinpath(src, f), joinpath(param_dir, f); force=true)
            end
        elseif endswith(f, ".log")
            # cv_spgpcdss_const_0.9_p1000.log
            # Map suffix/middle parts to algo names
            algo = ""
            if occursin("l0learn", f)
                algo = "L0LearnPSI1"
            elseif occursin("spgpcdss", f)
                algo = "SPGpCDSS"
            elseif occursin("spg", f)
                algo = "SPG"
            elseif occursin("cdss", f)
                algo = "CDSS"
            end
            
            if algo != ""
                log_dir = joinpath(dest_base, algo, "logs")
                mkpath(log_dir)
                mv(joinpath(src, f), joinpath(log_dir, f); force=true)
            end
        end
    end
    println("Organization complete.")
end

organize()
