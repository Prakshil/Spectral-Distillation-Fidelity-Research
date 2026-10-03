$env:OMP_NUM_THREADS = 8
foreach ($ds in @("tolokers","yelpchi")) {
  "=== START $ds at $(Get-Date -Format HH:mm:ss) ===" | Out-File -FilePath "logs/fe_seq.out" -Append -Encoding utf8
  python spectral_distillation/experiments/run_fixed_expert_protocol.py --dataset $ds --splits 10 --seeds 3 --out logs `
    2>&1 | Tee-Object -FilePath "logs/fe_seq.out" -Append | Out-Null
  "=== DONE $ds exit=$LASTEXITCODE at $(Get-Date -Format HH:mm:ss) ===" | Out-File -FilePath "logs/fe_seq.out" -Append -Encoding utf8
}
