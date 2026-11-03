param([string]$Log = "s8_pick_20261103.log")
$lines = Get-Content $Log
$order = @(); $data = @{}; $hash = @{}; $cur = ""
foreach ($l in $lines) {
    if ($l -match "^\s*---\s+(\S+)\s*---") { $cur = $Matches[1]; if (-not $order.Contains($cur)) { $order += $cur }; continue }
    if ($cur -eq "") { continue }
    if ($l -match "^\s*([A-Za-z]\d*_[0-9A-Za-z_\-]+)\s+occ=\d+.*?med\s+([\d.]+)\s+ms.*?fnv\s+([0-9a-f]+)\s*\|\s*cfnv\s+([0-9a-f]+)") {
        $row = $Matches[1]; $ms = [double]$Matches[2]
        if (-not $data.ContainsKey($row)) { $data[$row] = @{} }
        if (-not $data[$row].ContainsKey($cur)) { $data[$row][$cur] = @() }
        $data[$row][$cur] += $ms
        $k = "$row"
        if (-not $hash.ContainsKey($k)) { $hash[$k] = @{} }
        $hash[$k]["$($Matches[3])/$($Matches[4])"] = $true
    }
}
"builds (in order of appearance):"
$i = 0; foreach ($b in $order) { "  {0,2}  {1}" -f $i, $b; $i++ }
""
"=== med ms per row (mean over passes; n = passes) ==="
$hdr = "row".PadRight(22)
foreach ($b in $order) { $hdr += ("{0,10}" -f ($b -replace "fa_turbo_", "")) }
$hdr
foreach ($row in ($data.Keys | Sort-Object)) {
    $s = $row.PadRight(22)
    foreach ($b in $order) {
        if ($data[$row].ContainsKey($b)) {
            $v = $data[$row][$b]
            $s += ("{0,10}" -f ([math]::Round((($v | Measure-Object -Average).Average), 4)))
        } else { $s += ("{0,10}" -f "-") }
    }
    $s
}
""
"=== distinct fnv/cfnv per row across ALL builds (must be exactly 1) ==="
foreach ($row in ($hash.Keys | Sort-Object)) {
    $keys = @($hash[$row].Keys)
    "  {0,-22} {1} distinct: {2}" -f $row, $keys.Count, ($keys -join " ")
}
""
"=== per-pass spread (max-min within a row/build, sanity) ==="
foreach ($row in ($data.Keys | Sort-Object)) {
    foreach ($b in $order) {
        if ($data[$row].ContainsKey($b) -and $data[$row][$b].Count -gt 1) {
            $v = $data[$row][$b]
            $d = ($v | Measure-Object -Maximum).Maximum - ($v | Measure-Object -Minimum).Minimum
            if ($d -gt 0.02) { "  {0,-22} {1,-18} spread {2:N4}" -f $row, ($b -replace "fa_turbo_",""), $d }
        }
    }
}
