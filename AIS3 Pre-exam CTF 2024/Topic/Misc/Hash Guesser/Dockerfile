FROM python:3.12-slim

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY static /app/static
COPY templates /app/templates
COPY util.py /app/util.py
COPY app.py /app/app.py

WORKDIR /app

CMD ["python", "app.py"]