FROM python:3.8.12-slim-buster

EXPOSE $PORT
WORKDIR /app

RUN apt-get update && apt install -y \ 
    libgl1-mesa-glx \
    libglib2.0-0 && apt-get clean

RUN pip install --no-cache-dir \
    streamlit \ 
    pydantic \
    requests

ENV BASE_URL http://localhost:8000

COPY . /app

CMD ["sh", "-c", "streamlit run --server.address=0.0.0.0 --server.port $PORT app.py"]