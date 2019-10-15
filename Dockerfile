FROM jupyter/scipy-notebook
WORKDIR /home/jovian
EXPOSE 8888
COPY . work/spoton
RUN pip install -r work/spoton/requirements.txt
CMD [ "jupyter", "lab" ]
