# Start the demo app (app_v3.py) on this laptop only, then open http://localhost:8501 in a browser.
Set-Location $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" -m streamlit run app_v3.py --server.address localhost --server.port 8501
