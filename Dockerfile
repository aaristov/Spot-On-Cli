FROM conda/miniconda3
WORKDIR /usr/local/spoton
RUN conda create -y -n spoton python=3.7 matplotlib scipy
RUN /bin/bash -c "source activate spoton"
RUN conda install -c conda-forge jupyter notebook 
COPY . .
RUN pip install -r requirements.txt
RUN pip install .
CMD [ "/bin/bash"]
# "source", "spoton", "&", "jupyter", "notebook" ]
