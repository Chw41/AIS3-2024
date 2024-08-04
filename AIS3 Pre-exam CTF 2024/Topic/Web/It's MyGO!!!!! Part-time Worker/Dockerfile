FROM python:3.12-slim

COPY src /app

RUN apt update -y && apt install unzip

WORKDIR /app

RUN mkdir /app/image

RUN chmod 777 /app/image

RUN pip install -r requirements.txt

USER nobody

CMD ["gunicorn","-b 0.0.0.0:51414","app:app"]
