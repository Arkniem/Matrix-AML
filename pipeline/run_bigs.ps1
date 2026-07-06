Set-Location 'C:\Users\krog5w\.gemini\antigravity\scratch\AML-multimodal\pipeline'
$env:AMLMM_NJOBS = '4'
Remove-Item '..\runs\single_modality\bigs_done.flag' -ErrorAction SilentlyContinue
foreach ($m in @('GRN','RNA','Cell-comm')) {
  $env:AMLMM_MODALITY = $m
  & python _single_modality.py *> "..\runs\single_modality\local_$m.log"
}
'BIGS_DONE' | Out-File -Encoding utf8 '..\runs\single_modality\bigs_done.flag'
