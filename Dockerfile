FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "decision_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]