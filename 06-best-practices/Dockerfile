FROM python:3.10.9-slim

RUN pip install -U pip 

COPY [ "requirements.txt", "./" ]

RUN pip install -r requirements.txt

COPY [ "batch.py", "batch.py" ]
COPY [ "model.bin", "model.bin" ]

ENTRYPOINT [ "python", "batch.py" ]