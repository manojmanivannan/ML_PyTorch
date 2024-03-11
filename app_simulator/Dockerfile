FROM python:3.9-slim


# Set working directory
WORKDIR /app

RUN pip install --no-cache-dir pandas scipy streamlit
RUN pip install --no-cache-dir matplotlib imageio
COPY ./gen/sample_generator.py .
COPY ./app_simulator/main.py .
COPY ./model/model_data.json .
# Copy and install dependencies


# Command to run the application
CMD ["streamlit", "run", "main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
