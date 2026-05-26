FROM python:3.9-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# Remove bitsandbytes from requirements if it exists as it causes issues on CPU-only containers
RUN sed -i '/bitsandbytes/d' /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code/

# Expose port for HuggingFace/Render
EXPOSE 7860

# CMD for HuggingFace spaces (they default to 7860)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
