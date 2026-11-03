param([string[]]$Names = @("fa_turbo_vtabe","fa_turbo_vtab5_ce","fa_turbo_vtab3_ce"))
$bin = "build-x64-windows-msvc-release\bin"
foreach ($e in $Names) {
    "=== $($e -replace 'fa_turbo_','') ==="
    $r = & cuobjdump -res-usage "$bin\$e.exe" 2>$null
    foreach ($l in $r) {
        $t = "$l".Trim()
        if ($t.StartsWith("Function ")) {
            $n = $t.Substring(9).TrimEnd(':')
            if ($n -match "flash_attn_ext_vecILi(\d+)ELi\d+EL9ggml_type(\d+)ELS0_(\d+)ELb\d+ELb(\d+)ELb(\d+)") {
                "  D{0} K{1} V{2} mma{3} f16{4}" -f $Matches[1], $Matches[2], $Matches[3], $Matches[4], $Matches[5]
            }
        } elseif ($t.StartsWith("REG:")) {
            "      {0}" -f ($t -replace "CONSTANT.*", "")
        }
    }
}
