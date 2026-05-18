$ErrorActionPreference = "Stop"

Write-Host "Running LawRAG unit tests..."
python -m unittest discover -s tests -p "test_lawrag_qa.py"

Write-Host "Compiling LawRAG package..."
python -m compileall -q lawrag_qa

Write-Host "Running sample retrieval evaluation..."
python -m lawrag_qa.evaluation --cases data/eval/labor_qa_sample.jsonl --top-k 5

Write-Host "Scanning for local API keys in LawRAG files..."
$secretPattern = ("agt" + "_codex_|" + "LAWRAG" + "_LLM_API_KEY=.*" + "agt_|" + "OPENAI" + "_API_KEY=.*" + "agt_")
$matches = rg -n $secretPattern .env.example docs lawrag_qa tests requirements-lawrag.txt .gitignore configs schemas data/eval
if ($LASTEXITCODE -eq 0) {
  $matches
  Write-Error "Potential secret found. Inspect the matches above."
}
if ($LASTEXITCODE -gt 1) {
  exit $LASTEXITCODE
}

Write-Host "LawRAG checks passed."
