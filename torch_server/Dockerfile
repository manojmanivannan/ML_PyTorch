FROM pytorch/torchserve:latest-gpu


COPY ./model/model_onnx.onnx /home/model-server/
COPY ./model/model_data.json /home/model-server/
COPY ./torch_server/c_handler.py /home/model-server/
COPY ./torch_server/config.properties /home/model-server/
WORKDIR /home/model-server

RUN /home/venv/bin/python -m pip install onnxruntime

RUN torch-model-archiver \
--force \
--model-name onnx \
--version 1.0 \
--serialized-file model_onnx.onnx \
--handler c_handler.py \
--extra-files model_data.json \
--export-path ./model-store


CMD ["torchserve", \
 "--start", \
 "--ncs", \
 "--ts-config=config.properties", \
 "--model-store=model-store", \
 "--models=onnx.mar"]

EXPOSE 8080
EXPOSE 8081