FROM pytorch/torchserve:latest-cpu


# Set working directory
COPY ./model/model_onnx.onnx /home/model-server/
COPY ./model/model_data.json /home/model-server/
COPY ./torch_server/c_handler.py /home/model-server/
COPY ./torch_server/config.properties /home/model-server/
WORKDIR /home/model-server


RUN torch-model-archiver \
--force \
--model-name onnx \
--version 1.0 \
--serialized-file model_onnx.onnx \
--handler c_handler.py \
--extra-files model_data.json \
--export-path ./model-store



# COPY ./model/model_onnx.onnx .
# COPY ./model/model_data.json .
# COPY ./torch_server/main.py .

CMD ["torchserve", \
 "--start", \
 "--ncs", \
 "--ts-config=config.properties", \
 "--model-store=model-store", \
 "--models=onnx.mar"]