FROM pytorch/pytorch:latest
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

COPY ./torch_server/requirements.txt .
# Copy and install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY ./model/model_checkpoint.pth .
COPY ./prediction/nn_models.py .
COPY ./torch_server/main.py .

EXPOSE 8000
# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]