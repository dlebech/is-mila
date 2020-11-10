FROM python:3.8

WORKDIR /usr/src/app

# Install requirements
COPY requirements_docker.txt .
COPY requirements_base.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Make sure mobilenet weights are pre-loaded
RUN python -c 'from tensorflow.keras.applications.mobilenet import MobileNet; MobileNet(weights="imagenet")'

COPY . .

EXPOSE 8000

CMD [ "python", "-m", "mila.serve.app" ]