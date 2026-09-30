FROM python:3.14-slim  
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 
RUN mkdir -p /app/INDEX_STATE /app/VECTOR_EMBEDDING 
COPY src/ ./src/ 
EXPOSE 8003 

CMD ["uvicorn","api:app","--app-dir","src","--host","0.0.0.0","--port","8003"]