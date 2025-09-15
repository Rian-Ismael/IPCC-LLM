FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
#  Força o índice CPU do PyTorch (evita baixar CUDA)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
