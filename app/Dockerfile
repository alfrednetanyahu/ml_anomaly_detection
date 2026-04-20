FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

# Use gunicorn for production; workers kept low for t3.small
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "30",  \
     "--access-logfile", "-", "--error-logfile", "-", "app:app"]
