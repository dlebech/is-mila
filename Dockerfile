FROM python:3.6

WORKDIR /usr/src/app

COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD [ "python", "-m", "mila.serve.app" ]