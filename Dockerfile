FROM python:3.10

COPY web /app
COPY scripts /app
COPY requirements.txt /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["bash", "scripts/server.sh"]
