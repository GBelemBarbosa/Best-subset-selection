
using Printf
using Dates

function extract_metrics(log_path)
    if !isfile(log_path) return nothing end
    lines = readlines(log_path)
    results = Dict()
    current_n = 0
    
    for i in 1:length(lines)
        m_n = match(r"--- n=(\d+) Results", lines[i])
        if m_n !== nothing
            current_n = parse(Int, m_n.captures[1])
            results[current_n] = []
        end
        
        if current_n > 0 && occursin("|", lines[i]) && !occursin("Strategy", lines[i]) && !occursin("---", lines[i])
            push!(results[current_n], lines[i])
        end
    end
    
    if isempty(results) return nothing end
    max_n = maximum(keys(results))
    return results[max_n]
end

function build_dashboard()
    base_dir = "plots/CV comp"
    html_path = joinpath(base_dir, "dashboard.html")
    md_path = joinpath(base_dir, "cv_performance_analysis.md")
    
    algorithms = ["SPG", "SPGpCDSS", "L0LearnPSI1"]
    
    html_rows = ""
    plot_sections = ""
    md_rows = ""
    
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
                metrics = extract_metrics(matched_log)
                if metrics !== nothing
                    for row in metrics
                        parts = split(row, "|")
                        length(parts) < 7 && continue
                        strat = strip(parts[1])
                        pred = parse(Float64, strip(parts[3]))
                        sim = parse(Float64, strip(parts[4]))
                        time_str = strip(parts[5])
                        exec_time = try parse(Float64, time_str) catch; 9999.0 end
                        zwins = strip(parts[6])
                        zimprov = strip(parts[7])
                        push!(config_metrics, (strat=strat, pred=pred, sim=sim, time=exec_time, time_str=time_str, zwins=zwins, zimprov=zimprov))
                    end
                end
            end
            
            if !isempty(config_metrics)
                # 1. Accuracy Winner
                acc_winner = sort(config_metrics, by=x -> (-x.sim, x.pred))[1]
                
                # 2. Efficiency Winner (95% Sim threshold, min Time)
                max_sim = acc_winner.sim
                threshold = 0.95 * max_sim
                efficient_candidates = filter(x -> x.sim >= threshold, config_metrics)
                eff_winner = sort(efficient_candidates, by=x -> x.time)[1]
                
                html_rows *= """
                <tr>
                    <td>$algo</td>
                    <td><span class="param-tag">$params</span></td>
                    <td class="winner-cell"><span class="winner-label acc">$(acc_winner.strat)</span></td>
                    <td class="winner-cell"><span class="winner-label eff">$(eff_winner.strat)</span></td>
                    <td class="stat-cell $(acc_winner.sim > 0.9 ? "success" : "")">$(@sprintf("%.3f", acc_winner.sim))</td>
                    <td>$(eff_winner.time_str)s</td>
                    <td>$(acc_winner.zwins)</td>
                    <td>$(acc_winner.zimprov)</td>
                </tr>
                """
                
                md_rows *= "| **$algo** | $params | $(acc_winner.strat) | $(eff_winner.strat) | $(@sprintf("%.3f", acc_winner.sim)) | $(eff_winner.time_str)s | $(acc_winner.zwins) | $(acc_winner.zimprov) |\n"
                
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

| Algorithm | Parameters | Pure Accuracy | Balanced/Efficient | Sim | Time (s) | Z-Wins | Z-Improv |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
"""
    open(md_path, "w") do f 
        println(f, md_header * md_rows)
        println(f, "\n## Refinement Impact Analysis")
        println(f, "\nThe **Refinement Duel** compares starting from the previous \$\\lambda\$ result (**Path-Start**) vs. starting from zero (**Zero-Start**).")
        println(f, "\n### Observations:")
        println(f, "1. **SPG & SPGpCDSS**: Show high Z-Wins in the `exp` correlation cases. Zero-start is frequently essential to escape local minima.")
        println(f, "2. **L0Learn**: Extremely stable; refinement rarely avoids path-stalling as the R implementation is robust.")
    end
    
    # Final HTML
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CV Performance Dashboard - Dual Analysis</title>
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
            
            .plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 30px; padding: 30px; }
            .plot-box { background: rgba(0,0,0,0.2); border-radius: 12px; padding: 15px; text-align: center; }
            .plot-box img { width: 100%; border-radius: 8px; transition: 0.3s; }
            .plot-box img:hover { transform: translateY(-5px); }
            .plot-box p { margin-top: 15px; color: #94a3b8; font-size: 0.85rem; }
            
            .legend { display: flex; gap: 30px; margin-top: 20px; font-size: 0.85rem; color: #94a3b8; }
            .legend-item { display: flex; align-items: center; gap: 8px; }
            .dot { width: 12px; height: 12px; border-radius: 50%; }
            
            footer { text-align: center; padding: 40px; color: #64748b; font-size: 0.8rem; border-top: 1px solid var(--card-border); }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Experiment Dashboard</h1>
                <p class="subtitle">Dual-Winner Analysis: Pure Accuracy vs. Efficiency (95% Sim Threshold)</p>
            </header>

            <div class="card">
                <div class="card-header">
                    <h2 style="margin:0; font-size: 1.2rem;">Comparative Performance Matrix</h2>
                    <div class="legend">
                        <div class="legend-item"><span class="dot" style="background: var(--winner-acc);"></span> Pure Accuracy (Max Sim)</div>
                        <div class="legend-item"><span class="dot" style="background: var(--winner-eff);"></span> Efficient (Fastest within 5% of Max Sim)</div>
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
                                <th>Eff. Time</th>
                                <th>Z-Wins</th>
                                <th>Improvs</th>
                            </tr>
                        </thead>
                        <tbody>
                            $html_rows
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
