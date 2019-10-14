FROM jupyterhub/jupyterhub
WORKDIR /usr/local/spoton
RUN conda create -y -n spoton python=3.7 matplotlib scipy
RUN /bin/bash -c "source activate spoton"
# RUN conda install -c conda-forge jupyterhub 
# COPY . .
# RUN pip install -r requirements.txt 
# RUN pip install .
RUN useradd andrey -p "$(openssl passwd -1 pqzasdfmtr)"
CMD [ "/bin/bash", "-c", "source activate spoton & jupyterhub" ]
