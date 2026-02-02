
using Printf
using Dates

function extract_metrics(log_path)
    if !isfile(log_path) return nothing end
    lines = readlines(log_path)
    # strat_data[strategy_name] = [row1_data, row2_data, ...]
    strat_data = Dict{String, Any}()
    current_n = 0
    
    for row in lines
        m_n = match(r"--- n=(\d+) Results", row)
        if m_n !== nothing
            current_n = parse(Int, m_n.captures[1])
            continue
        end
        
        if current_n > 0 && occursin("|", row) && !occursin("Strategy", row) && !occursin("---", row)
            parts = split(row, "|")
            length(parts) < 7 && continue
            strat = strip(parts[1])
            pred = parse(Float64, strip(parts[3]))
            sim = parse(Float64, strip(parts[4]))
            time_str = strip(parts[5])
            exec_time = try parse(Float64, time_str) catch; 9999.0 end
            
            # Refinement stats
            metrics = Dict(:sim => sim, :pred => pred, :time => exec_time, :time_str => time_str)
            if length(parts) == 8
                # B / T / W format
                metrics[:zwins] = parse(Int, strip(parts[6]))
                metrics[:zt] = parse(Int, strip(parts[7]))
                metrics[:zl] = parse(Int, strip(parts[8]))
            else
                # Old Wins/Improv format
                metrics[:zwins] = parse(Int, strip(parts[6]))
                metrics[:zimprov] = strip(parts[7])
            end
            
            if !haskey(strat_data, strat) strat_data[strat] = [] end
            push!(strat_data[strat], metrics)
        end
    end
    
    if isempty(strat_data) return nothing end
    
    # Aggregate by strategy
    aggr_results = []
    for (strat, history) in strat_data
        n_points = length(history)
        avg_sim = sum(h[:sim] for h in history) / n_points
        avg_pred = sum(h[:pred] for h in history) / n_points
        total_time = sum(h[:time] for h in history)
        avg_time_str = @sprintf("%.2f", total_time / n_points)
        
        zwins = sum(h[:zwins] for h in history)
        zt = haskey(history[1], :zt) ? sum(h[:zt] for h in history) : nothing
        zl = haskey(history[1], :zl) ? sum(h[:zl] for h in history) : nothing
        
        # In old logs, zimprov was a separate count. In new logs, B (Better Recovery) is stored in zwins.
        zimprov_val = haskey(history[1], :zimprov) ? sum(try parse(Int, h[:zimprov]) catch; 0 end for h in history) : nothing
        
        push!(aggr_results, (
            strat=strat, 
            sim=avg_sim, 
            pred=avg_pred, 
            time=total_time, # Use total for comparisons
            avg_time=total_time/n_points,
            time_str=avg_time_str, 
            zwins=zwins, 
            zt=zt, 
            zl=zl, 
            zimprov=zimprov_val
        ))
    end
    
    return aggr_results
end

function build_dashboard()
    base_dir = "plots/CV comp"
    html_path = joinpath(base_dir, "dashboard.html")
    md_path = joinpath(base_dir, "cv_performance_analysis.md")
    
    algorithms = ["SPG", "SPGpCDSS", "L0LearnPSI1"]
    
    html_rows = ""
    h2h_rows = ""
    plot_sections = ""
    md_rows = ""
    h2h_md_rows = ""
    
    for algo in algorithms
        algo_path = joinpath(base_dir, algo)
        !isdir(algo_path) && continue
        
        params_folders = filter(d -> isdir(joinpath(algo_path, d)) && d != "logs", readdir(algo_path))
        
        for params in params_folders
            log_dir = joinpath(algo_path, "logs")
            log_files = isdir(log_dir) ? readdir(log_dir) : []
            p_parts = split(params, "-")
            target_corr = p_parts[1]
            target_p = p_parts[3]
            
            matched_log = nothing
            for lf in log_files
                if occursin(target_corr, lf) && occursin("p"*target_p, lf)
                    short_algo = lowercase(algo)
                    if occursin("l0learn", short_algo) short_algo = "l0learn" end
                    if occursin("spgpcdss", short_algo) short_algo = "spgpcdss" end
                    if occursin("spg", short_algo) && !occursin("spgpcdss", short_algo) short_algo = "spg" end
                    
                    if occursin(short_algo, lowercase(lf))
                        matched_log = joinpath(log_dir, lf)
                        break
                    end
                end
            end
            
            config_metrics = []
            if matched_log !== nothing
                config_metrics = extract_metrics(matched_log)
            end
            
            if config_metrics !== nothing && !isempty(config_metrics)
                # 1. Accuracy Winner (Max average Sim)
                acc_winner = sort(config_metrics, by=x -> (-x.sim, x.pred))[1]
                
                # 2. Efficiency Winner (95% Sim threshold, min Total Time)
                max_sim = acc_winner.sim
                threshold = 0.95 * max_sim
                efficient_candidates = filter(x -> x.sim >= threshold, config_metrics)
                eff_winner = sort(efficient_candidates, by=x -> x.time)[1]
                
                # Gain Ratio (Ratio of Sums)
                gain_ratio = acc_winner.time / max(eff_winner.time, 1e-6)
                gain_str = gain_ratio > 1.05 ? @sprintf("%.1fx Faster", gain_ratio) : "Same"
                
                # Sim Gain (Tradeoff: AccuracyWinner.Sim - EfficiencyWinner.Sim)
                sim_gain = acc_winner.sim - eff_winner.sim
                sim_gain_str = sim_gain > 1e-4 ? @sprintf("+%.3f", sim_gain) : "0"

                # Sim Improv (B in new logs, zimprov in old logs)
                sim_improv = acc_winner.zt !== nothing ? Int(acc_winner.zwins) : Int(acc_winner.zimprov)

                # Refinement Breakdown Display (Aggregated Sums)
                ref_breakdown = try
                    if acc_winner.zt !== nothing
                        "$(Int(acc_winner.zwins))B / $(Int(acc_winner.zt))T / $(Int(acc_winner.zl))W"
                    else
                        "$(Int(acc_winner.zwins)) Wins"
                    end
                catch; "N/A" end

                # H2H Comparison: Adaptive vs Inverse (for SPG)
                if algo in ["SPG", "SPGpCDSS"]
                    adaptive = filter(x -> x.strat == "Smart Adaptive", config_metrics)
                    inverse = filter(x -> x.strat == "Inverse CV", config_metrics)
                    
                    if !isempty(adaptive) && !isempty(inverse)
                        ad = adaptive[1]
                        inv = inverse[1]
                        
                        sim_diff = ad.sim - inv.sim
                        sim_diff_str = abs(sim_diff) < 1e-4 ? "Tied" : (sim_diff > 0 ? @sprintf("+%.3f (Ad)", sim_diff) : @sprintf("+%.3f (Inv)", -sim_diff))
                        
                        # Speedup: how much faster is Inverse than Adaptive? (Usually Inverse is faster)
                        speedup = inv.time / max(ad.time, 1e-6)
                        speedup_str = speedup > 1.05 ? @sprintf("%.1fx (Ad Faster)", speedup) : (speedup < 0.95 ? @sprintf("%.1fx (Inv Faster)", 1/speedup) : "Tied")
                        
                        h2h_rows *= """
                        <tr>
                            <td>$algo</td>
                            <td><span class="param-tag">$params</span></td>
                            <td class="stat-cell $(sim_diff > 1e-4 ? "success" : (sim_diff < -1e-4 ? "warning" : ""))">$sim_diff_str</td>
                            <td>$speedup_str</td>
                            <td>$(@sprintf("%.2f", ad.avg_time))s vs $(@sprintf("%.2f", inv.avg_time))s</td>
                        </tr>
                        """
                        h2h_md_rows *= "| $algo | $params | $sim_diff_str | $speedup_str | $(@sprintf("%.2f", ad.avg_time))s vs $(@sprintf("%.2f", inv.avg_time))s |\n"
                    end
                end

                html_rows *= """
                <tr>
                    <td>$algo</td>
                    <td><span class="param-tag">$params</span></td>
                    <td class="winner-cell"><span class="winner-label acc">$(acc_winner.strat)</span></td>
                    <td class="winner-cell"><span class="winner-label eff">$(eff_winner.strat)</span></td>
                    <td class="stat-cell $(acc_winner.sim > 0.9 ? "success" : "")">$(@sprintf("%.3f", acc_winner.sim))</td>
                    <td class="stat-cell $(sim_gain > 1e-4 ? "warning" : "")">$sim_gain_str</td>
                    <td><span class="gain-badge">$(gain_str)</span></td>
                    <td>$(eff_winner.time_str)s/n</td>
                    <td class="stat-cell $(sim_improv > 0 ? "success" : "")">$sim_improv</td>
                    <td class="ref-breakdown">$ref_breakdown</td>
                </tr>
                """
                
                md_rows *= "| **$algo** | $params | $(acc_winner.strat) | $(eff_winner.strat) | $(@sprintf("%.3f", acc_winner.sim)) | $sim_gain_str | $(gain_str) | $(eff_winner.time_str)s/n | $sim_improv | $ref_breakdown |\n"
                
                # Plot Gallery Section
                param_path = joinpath(algo, params)
                plot_section_id = replace(params, "."=>"")
                plot_sections *= """
                <div class="card plot-group" id="$plot_section_id">
                    <div class="card-header">
                        <h3>$algo &mdash; $params</h3>
                    </div>
                    <div class="plot-grid">
                        <div class="plot-box">
                            <img src="$param_path/Comp_Sim-$(algo)_Comparison_$(params).png" alt="Similarity">
                            <p>Support Similarity (Recovery Rate)</p>
                        </div>
                        <div class="plot-box">
                            <img src="$param_path/Comp_Refinement-$(algo)_Comparison_$(params).png" alt="Refinement">
                            <p>Refinement Statistics (Z-Wins/Improvs)</p>
                        </div>
                    </div>
                </div>
                """
            else
                md_rows *= "| **$algo** | $params | *Missing* | *Missing* | - | - | - | - |\n"
            end
        end
    end
    
    # Save MD
    md_header = """# CV Strategy Performance Analysis (Dual-Winner)

## Best CV Performance Summary

| Algorithm | Parameters | Pure Accuracy | Balanced/Efficient | Sim | Gain | Time (s/n) | Sim Improv | Refinement Impact |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""
    open(md_path, "w") do f 
        println(f, md_header * md_rows)
        println(f, "\n## Refinement Impact Analysis")
        println(f, "\nThe **Refinement Duel** compares starting from the previous \$\\lambda\$ result (**Path-Start**) vs. starting from zero (**Zero-Start**).")
        println(f, "\n### Observations:")
        println(f, "1. **SPG & SPGpCDSS**: Show high Z-Wins in the `exp` correlation cases. Zero-start is frequently essential to escape local minima.")
        println(f, "2. **L0Learn**: Extremely stable; refinement rarely avoids path-stalling as the R implementation is robust.")
        
        println(f, "\n## CV Strategy Duel: Smart Adaptive vs Inverse")
        println(f, "\n| Algorithm | Parameters | Best Sim Δ | Speedup | Avg Time (Ad vs Inv) |")
        println(f, "| :--- | :--- | :---: | :---: | :---: |")
        println(f, h2h_md_rows)
    end
    
    # Final HTML
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CV Performance Dashboard - Global Analysis</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #0b0f19;
                --text: #e2e8f0;
                --accent: #38bdf8;
                --accent-soft: rgba(56, 189, 248, 0.1);
                --card-bg: rgba(30, 41, 59, 0.5);
                --card-border: rgba(255, 255, 255, 0.1);
                --success: #10b981;
                --winner-acc: #f59e0b;
                --winner-eff: #818cf8;
            }
            * { box-sizing: border-box; }
            body { 
                margin: 0; font-family: 'Outfit', sans-serif; 
                background: var(--bg); color: var(--text);
                background-image: radial-gradient(circle at top right, #1e293b, transparent), 
                                 radial-gradient(circle at bottom left, #0f172a, transparent);
                background-attachment: fixed;
            }
            .container { max-width: 1400px; margin: 0 auto; padding: 40px 20px; }
            header { text-align: center; margin-bottom: 60px; }
            h1 { font-size: 3rem; font-weight: 700; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
            .subtitle { color: #94a3b8; font-size: 1.2rem; font-weight: 300; }
            
            .card { 
                background: var(--card-bg); backdrop-filter: blur(16px);
                border: 1px solid var(--card-border); border-radius: 20px;
                box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
                margin-bottom: 40px; overflow: hidden;
            }
            .card-header { padding: 20px 30px; border-bottom: 1px solid var(--card-border); background: rgba(0,0,0,0.1); }
            
            table { width: 100%; border-collapse: collapse; }
            th { text-align: left; padding: 15px; background: rgba(0,0,0,0.2); font-weight: 600; color: var(--accent); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
            td { padding: 18px 15px; border-bottom: 1px solid var(--card-border); font-size: 0.9rem; }
            tr:hover { background: rgba(255,255,255,0.02); }
            
            .param-tag { background: rgba(56, 189, 248, 0.05); border: 1px solid rgba(56, 189, 248, 0.3); color: var(--accent); padding: 4px 8px; border-radius: 6px; font-family: monospace; font-size: 0.75rem; }
            
            .winner-label { padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
            .winner-label.acc { background: rgba(245, 158, 11, 0.15); color: var(--winner-acc); border: 1px solid rgba(245, 158, 11, 0.3); }
            .winner-label.eff { background: rgba(129, 140, 248, 0.15); color: var(--winner-eff); border: 1px solid rgba(129, 140, 248, 0.3); }
            
            .stat-cell.success { color: var(--success); font-weight: 700; }
            .stat-cell.warning { color: var(--winner-acc); font-weight: 600; }
            
            .gain-badge { background: rgba(16, 185, 129, 0.1); color: var(--success); padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.8rem; border: 1px solid rgba(16, 185, 129, 0.3); }
            .ref-breakdown { font-family: monospace; font-size: 0.8rem; color: #94a3b8; }
            
            .plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 30px; padding: 30px; }
            .plot-box { background: rgba(0,0,0,0.2); border-radius: 12px; padding: 15px; text-align: center; }
            .plot-box img { width: 100%; border-radius: 8px; transition: 0.3s; }
            .plot-box img:hover { transform: translateY(-5px); }
            .plot-box p { margin-top: 15px; color: #94a3b8; font-size: 0.85rem; }
            
            .legend { display: flex; flex-wrap: wrap; gap: 30px; margin-top: 20px; font-size: 0.85rem; color: #94a3b8; }
            .legend-item { display: flex; align-items: center; gap: 8px; }
            .dot { width: 12px; height: 12px; border-radius: 50%; }
            
            footer { text-align: center; padding: 40px; color: #64748b; font-size: 0.8rem; border-top: 1px solid var(--card-border); }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Experiment Dashboard</h1>
                <p class="subtitle">Aggregate Performance Analysis: Averaged across all Sample Sizes (\$n\$)</p>
            </header>
 
            <div class="card">
                <div class="card-header">
                    <h2 style="margin:0; font-size: 1.2rem;">Global Comparative Matrix</h2>
                    <div class="legend">
                        <div class="legend-item"><span class="dot" style="background: var(--winner-acc);"></span> <b>Pure Accuracy</b>: Highest mean Sim across all \$n\$</div>
                        <div class="legend-item"><span class="dot" style="background: var(--winner-eff);"></span> <b>Efficient</b>: Fastest total Time within 5% of Max Sim</div>
                        <div class="legend-item"><span class="dot" style="background: var(--success);"></span> <b>Gain</b>: Efficiency speedup (Ratio of total experiment sums)</div>
                        <div class="legend-item"><b>Sim Gain</b>: Accuracy tradeoff (AccWinner.Sim - EffWinner.Sim)</div>
                        <div class="legend-item"><b>Refinement Impact</b>: [Better] / [Tied] / [Worse] similarity (Total experiment counts)</div>
                    </div>
                </div>
                <div style="padding:0;">
                    <table>
                        <thead>
                            <tr>
                                <th>Algorithm</th>
                                <th>Configuration</th>
                                <th>Accuracy Winner</th>
                                <th>Efficiency Winner</th>
                                <th>Best Sim</th>
                                <th>Sim Gain</th>
                                <th>Gain</th>
                                <th>Eff. Time</th>
                                <th>Sim Improv</th>
                                <th>Refinement Impact</th>
                            </tr>
                        </thead>
                        <tbody>
                            $html_rows
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2 style="margin:0; font-size: 1.2rem;">CV Strategy Duel: Smart Adaptive vs Inverse</h2>
                    <p style="color: #94a3b8; font-size: 0.85rem; margin: 10px 0 0 0;">Focused comparison for SPG-based algorithms. <b>Ad</b> = Smart Adaptive, <b>Inv</b> = Inverse CV.</p>
                </div>
                <div style="padding:0;">
                    <table>
                        <thead>
                            <tr>
                                <th>Algorithm</th>
                                <th>Configuration</th>
                                <th>Support Sim Δ</th>
                                <th>Complexity Speedup</th>
                                <th>Avg. Time/n</th>
                            </tr>
                        </thead>
                        <tbody>
                            $h2h_rows
                        </tbody>
                    </table>
                </div>
            </div>

            <h2 style="margin: 60px 0 20px 0; font-size: 1.4rem; color: var(--accent);">Visual Evidence Gallery</h2>
            $plot_sections
            
            <footer>
                Dashboard generated at $(Dates.format(now(), "yyyy-mm-dd HH:MM")) &bull; Best Selection Projects
            </footer>
        </div>
    </body>
    </html>
    """
    
    open(html_path, "w") do f println(f, html_template) end
    println("Dual-Winner Dashboard successfully built: $html_path")
end

build_dashboard()
