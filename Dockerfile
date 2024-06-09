FROM python:3.10

COPY requirements.txt /app/

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 8080

COPY web /app/web
COPY scripts /app/scripts

CMD ["bash", "scripts/server.sh"]
