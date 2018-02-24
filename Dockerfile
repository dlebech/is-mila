FROM python:3.6

WORKDIR /usr/src/app

# Install requirements
COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make sure mobilenet weights are pre-loaded
RUN python -c 'from keras.applications.mobilenet import MobileNet; MobileNet(weights="imagenet")'

COPY . .

EXPOSE 8000

CMD [ "python", "-m", "mila.serve.app" ]