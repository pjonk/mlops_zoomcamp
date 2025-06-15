FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install --upgrade pip

COPY requirements.txt .
COPY pipeline.py . 

RUN pip install -r requirements.txt

CMD ["python", "pipeline.py", "--month", "05", "--year", "2023", "--model_path", './model2.bin', "--vectorizer_path", './dv.bin']
