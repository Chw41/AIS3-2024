FROM python:3.10-slim

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY static /app/static
COPY app.py /app/app.py

WORKDIR /app

CMD ["python", "app.py"]