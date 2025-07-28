FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn pydantic numpy torch scikit-learn
EXPOSE 8000
CMD ["uvicorn", "dl_api:app", "--host", "0.0.0.0", "--port", "8000"]
