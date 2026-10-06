@echo off
cd /d "D:\MY project\Industrial_MultiCamera_Tracking"
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000