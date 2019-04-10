FROM ubuntu:latest

RUN apt-get update -y && \
    apt-get install -y python3-pip


WORKDIR /app

COPY . /app

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 3000


ENTRYPOINT [ "python3" ]

CMD [ "flaskproject.py" ]
