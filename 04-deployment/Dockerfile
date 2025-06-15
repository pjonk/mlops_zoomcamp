FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install --upgrade pip

COPY requirements.txt .
COPY pipeline.py . 

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "pipeline.py"]
