# Check singular log file for ANSI codes
const LOG = "plots/NewTestTerminal_mixed/logs/robust_mixed_v2_const_0.5_p2000.log"

if !isfile(LOG)
    println("Log not found")
else
    lines = readlines(LOG)
    println("Checking line 2078 (SUP):")
    line = lines[2078]
    println("Range: $(2078)")
    println("Content: '$line'")
    println("Bytes: ", codeunits(line))
    
    println("\nChecking line 2077 (Algo):")
    line = lines[2077]
    println("Content: '$line'")
    println("Bytes: ", codeunits(line))
end
