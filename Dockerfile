# Dockerfile for jupyter notebook with all necessary dependensies for spot-on-cli
# to build:
#   docker build --rm -f "Dockerfile" -t spot-on-cli:lab-scipy . 
# To run:
#   docker run --shm-size 4g -it -v c:\Users\andre:/home/jovian/andrey -p 8888:8888/tcp spot-on-cli:lab-scipy

FROM jupyter/scipy-notebook
WORKDIR /home/jovian
EXPOSE 8888
COPY . .
RUN pip install -r requirements.txt && \
    pip install -e .
CMD [ "jupyter", "lab" ]
