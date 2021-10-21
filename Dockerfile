FROM python:3.7-slim-buster
WORKDIR /
RUN pip install fastapi transformers torch pandas numpy joblib pydantic uvicorn
# RUN pip install -r requirements.txt
# ADD model /model #
ADD down.py /
RUN ["python3", "down.py"]
ADD main.py /
EXPOSE 5000
CMD ["python3", "main.py"]
