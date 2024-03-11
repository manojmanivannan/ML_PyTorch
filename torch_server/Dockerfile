FROM pytorch/pytorch:latest
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get install -y libgomp1
# Set working directory
WORKDIR /app

COPY ./torch_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY ./model/model_onnx.onnx .
COPY ./model/model_data.json .
COPY ./torch_server/main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]