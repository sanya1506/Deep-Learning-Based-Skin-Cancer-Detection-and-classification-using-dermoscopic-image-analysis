param($in, $out)
$w = New-Object -ComObject Word.Application; $w.Visible = $false
$d = $w.Documents.Open($in, $false, $true)
$d.Repaginate()
$res = @()
$n = $d.Paragraphs.Count
for ($i = 1; $i -le $n; $i++) {
  $r = $d.Paragraphs.Item($i).Range
  if ($r.Information(12)) { continue }   # wdWithInTable
  $t = $r.Text.Trim()
  if ($t.Length -gt 0 -and $t.Length -lt 160) { $res += [pscustomobject]@{t = $t; p = $r.Information(1)} }
}
$res | ConvertTo-Json -Compress | Out-File -Encoding utf8 $out
"pages: " + $d.ComputeStatistics(2)
$d.Close($false); $w.Quit()
