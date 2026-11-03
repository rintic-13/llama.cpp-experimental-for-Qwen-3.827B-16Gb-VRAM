param([string[]]$Names = @("fa_turbo_vtab","fa_turbo_vtab8","fa_turbo_vtabe","fa_turbo_vtab_ce","fa_turbo_vtab8_ce","fa_turbo_vtabe_ce","fa_turbo_vtab3_ce","fa_turbo_vtab5_ce"))
$bin = "build-x64-windows-msvc-release\bin"
$op = '/\*[0-9a-f]+\*/'
foreach ($e in $Names) {
    "=== $($e -replace 'fa_turbo_','') ==="
    $s = & cuobjdump -sass "$bin\$e.exe" 2>$null
    $cur = ""; $n = 0; $rows = @()
    foreach ($l in $s) {
        if ($l -match "^\s*Function : (\S+)") {
            if ($cur -match "flash_attn_ext_vec") {
                $rows += [pscustomobject]@{ n = $n; name = ($cur -replace "_Z18flash_attn_ext_vecILi(\d+)ELi(\d+)EL9ggml_type(\d+)ELS0_(\d+)ELb(\d)ELb(\d)ELb(\d)EE.*", "D$1 K$3 V$4 sc$5 mma$6 f16$7") }
            }
            $cur = $Matches[1]; $n = 0
        } elseif ($cur -ne "" -and $l -match "^\s*$op") { $n++ }
    }
    if ($cur -match "flash_attn_ext_vec") {
        $rows += [pscustomobject]@{ n = $n; name = ($cur -replace "_Z18flash_attn_ext_vecILi(\d+)ELi(\d+)EL9ggml_type(\d+)ELS0_(\d+)ELb(\d)ELb(\d)ELb(\d)EE.*", "D$1 K$3 V$4 sc$5 mma$6 f16$7") }
    }
    foreach ($r in $rows) { "  {0,6}  {1}" -f $r.n, $r.name }
}
