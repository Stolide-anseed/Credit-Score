FROM ubuntu:latest
LABEL authors="stoli"

ENTRYPOINT ["top", "-b"]