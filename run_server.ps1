cd "D:\MY project\Industrial_MultiCamera_Tracking"
python -c "import uvicorn; uvicorn.run('src.api.main:app', host='0.0.0.0', port=8000, reload=False)"