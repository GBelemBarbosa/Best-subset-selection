using Dates

base_dir_adapt = "/mnt/c/Users/gggt4/Documents/Pesquisa/Best selection/plots/newtestterminal"
base_dir_inv = "/mnt/c/Users/gggt4/Documents/Pesquisa/Best selection/plots/NewTestTerminal_inv"
html_path = joinpath(base_dir_adapt, "comparison_dashboard.html")

# Metrics to display
metrics = [
    ("pnSim", "Recovery Rate (Support Similarity)"),
    ("pnPred", "Prediction Error"),
    ("pnSimImprov", "Similarity Improvement"),
    ("pnTime", "Execution Time"),
    ("pnInf", "Infinity Norm Error (L-inf)"),
    ("pnRefinement", "Refinement Statistics")
]

function get_folders(dir)
    if !isdir(dir) return String[] end
    return filter(d -> isdir(joinpath(dir, d)) && d != "logs", readdir(dir))
end

function build_dashboard()
    # Get all parameter sets from Regular Adaptive folder (Master List)
    reg_adapt_dir = joinpath(base_dir_adapt, "Regular")
    
    # We assume other folders exist and match
    reg_folders = get_folders(reg_adapt_dir)
    param_sets = []
    
    for folder in reg_folders
        # Standard format: const-0.5-2000..._standard
        base_name = replace(folder, "_standard" => "")
        push!(param_sets, (base_name, folder))
    end
    
    # Sort for consistent order
    sort!(param_sets, by = x -> x[1])

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Adaptive vs Inverse Comparison Dashboard</title>
         <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
             :root { --bg: #0f172a; --text: #e2e8f0; --card-bg: #1e293b; --accent: #38bdf8; --row-bg: #15202e; --inv-accent: #f472b6; }
            body { margin: 0; font-family: 'Outfit', sans-serif; background: var(--bg); color: var(--text); padding: 20px; }
            h1 { text-align: center; color: var(--accent); margin-bottom: 60px; font-size: 2.5rem; }
            
            .metric-section { margin-bottom: 100px; border: 1px solid #334155; border-radius: 20px; padding: 20px; background: rgba(30, 41, 59, 0.3); }
            .metric-header { text-align: center; margin-bottom: 40px; margin-top: -40px; }
            .metric-header h2 { 
                font-size: 2rem; background: var(--accent); color: #0f172a; 
                display: inline-block; padding: 15px 40px; border-radius: 30px; 
                box-shadow: 0 4px 15px rgba(56, 189, 248, 0.4);
            }
            
            /* 5-column grid: Label | Reg(Ad) | Tri(Ad) | Reg(Inv) | Tri(Inv) */
            .header-row { 
                display: grid; grid-template-columns: 150px 1fr 1fr 1fr 1fr; gap: 20px; 
                margin-bottom: 20px; text-align: center; font-weight: 700; 
                color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; font-size: 0.9rem;
            }
            
            .comparison-row { 
                display: grid; grid-template-columns: 150px 1fr 1fr 1fr 1fr; gap: 20px; 
                align-items: center; margin-bottom: 30px; 
                background: var(--row-bg); padding: 20px; border-radius: 12px;
                border: 1px solid #334155;
            }
            
            .param-label { font-size: 0.85rem; font-family: monospace; color: var(--accent); word-break: break-word; font-weight: 600; padding: 10px; }
            
            .plot-box { text-align: center; position: relative; }
            .plot-box img { width: 100%; border-radius: 8px; transition: transform 0.2s; cursor: pointer; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #334155; }
            .plot-box img:hover { transform: scale(1.8); box-shadow: 0 20px 40px rgba(0,0,0,0.8); z-index: 100; position: fixed; top: 10%; left: 20%; width: 60%; height: auto; object-fit: contain; }
            
            .strat-label { position: absolute; top: 5px; right: 5px; font-size: 0.7rem; background: rgba(0,0,0,0.7); padding: 2px 6px; border-radius: 4px; pointer-events: none; }
            .strat-inv { color: var(--inv-accent); border: 1px solid var(--inv-accent); }
            .strat-ad { color: var(--accent); border: 1px solid var(--accent); }
            
            .col-header { padding-bottom: 10px; border-bottom: 2px solid #334155; }
            .col-header.inv { border-bottom-color: var(--inv-accent); color: var(--inv-accent); }
        </style>
    </head>
    <body>
        <h1>Adaptive vs Inverse Strategy Comparison</h1>
        
        <p style="text-align: center; margin-bottom: 50px; color: #94a3b8;">
            Grouped by Metric. <b style="color:var(--accent)">Blue headers = Adaptive</b> (Old). <b style="color:var(--inv-accent)">Pink headers = Inverse</b> (New).
        </p>
    """

    for (prefix, title) in metrics
        html_content *= """
        <div class="metric-section">
            <div class="metric-header">
                <h2>$title</h2>
            </div>
            
            <div class="header-row">
                <div>Parameter Set</div>
                <div class="col-header">Regular (Adapt)</div>
                <div class="col-header">Tridiagonal (Adapt)</div>
                <div class="col-header inv">Regular (Inverse)</div>
                <div class="col-header inv">Tridiagonal (Inverse)</div>
            </div>
        """
        
        for (base_name, folder_std) in param_sets
            folder_tri = base_name * "_tridiagonal"
            
            # Paths relative to html file location (base_dir_adapt)
            
            img_name_std = "$(prefix)-NewTerminal_$(base_name)_standard.png"
            img_name_tri = "$(prefix)-NewTerminal_$(base_name)_tridiagonal.png"
            
            img_name_std_inv = replace(img_name_std, ".png" => "_inv.png")
            img_name_tri_inv = replace(img_name_tri, ".png" => "_inv.png")
            
            # 1. Adapt Regular
            src_ad_reg = "Regular/$folder_std/$img_name_std"
            # 2. Adapt Tri
            src_ad_tri = "Tridiagonal/$folder_tri/$img_name_tri"
            # 3. Inv Regular (NewTestTerminal_inv capitalized)
            src_inv_reg = "../NewTestTerminal_inv/Regular/$folder_std/$img_name_std_inv"
            # 4. Inv Tri
            src_inv_tri = "../NewTestTerminal_inv/Tridiagonal/$folder_tri/$img_name_tri_inv"
            
            param_display = replace(base_name, "-" => " ")
            
            html_content *= """
            <div class="comparison-row">
                <div class="param-label">$param_display</div>
                
                <div class="plot-box">
                    <img src="$src_ad_reg" alt="Reg Adapt" onclick="window.open(this.src)" loading="lazy">
                    <span class="strat-label strat-ad">Adapt</span>
                </div>
                <div class="plot-box">
                    <img src="$src_ad_tri" alt="Tri Adapt" onclick="window.open(this.src)" loading="lazy">
                    <span class="strat-label strat-ad">Adapt</span>
                </div>
                <div class="plot-box">
                    <img src="$src_inv_reg" alt="Reg Inv" onclick="window.open(this.src)" loading="lazy">
                    <span class="strat-label strat-inv">Inv</span>
                </div>
                <div class="plot-box">
                    <img src="$src_inv_tri" alt="Tri Inv" onclick="window.open(this.src)" loading="lazy">
                    <span class="strat-label strat-inv">Inv</span>
                </div>
            </div>
            """
        end
        html_content *= "</div>" # End Metric Section
    end

    html_content *= """
        <footer>
            <p style="text-align: center; color: #64748b; margin-top: 60px;">Generated at $(Dates.format(now(), "yyyy-mm-dd HH:MM"))</p>
        </footer>
    </body>
    </html>
    """

    open(html_path, "w") do f
        println(f, html_content)
    end
    println("Dashboard generated: $html_path")
end

build_dashboard()
