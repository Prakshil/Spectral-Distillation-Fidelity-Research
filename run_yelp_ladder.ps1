param([string]$Budgets, [int]$TimeoutSec = 7200)
$env:OMP_NUM_THREADS = 8
$log = "logs\yelp_ladder_resume.out"
foreach ($chunk in @(@(0.15,0.08,0.04))) {
  $b = ($chunk -join ",")
  "=== RESUME budgets=$b at $(Get-Date -Format HH:mm:ss) ===" | Out-File -FilePath $log -Append -Encoding utf8
  python spectral_distillation/experiments/run_sparsify_ladder.py --dataset yelpchi `
    --budgets $b --methods er,random,degree --skip-sd --resume --no-progress --out logs `
    2>&1 | Tee-Object -FilePath $log -Append | Select-String -Pattern "resuming|^\w+\s+r=|wrote"
}