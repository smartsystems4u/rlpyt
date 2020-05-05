FROM pytorch/pytorch:latest

RUN apt-get update \
     && apt-get install -y \
        libgl1-mesa-glx \
        libx11-xcb1 \
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

RUN /opt/conda/bin/conda install --yes \
    astropy \
    matplotlib \
    pandas \
    scikit-learn \
    scikit-image

WORKDIR /workspace
RUN chmod -R a+w .
RUN pip install torch tensorboard gym psutil pyprind
COPY . .
RUN pip install -e .
#CMD ["python", "rlpyt_testing/rplyt_deep_sea_treasure_dqn.py"]
RUN chmod +x bring_up.sh
#RUN start.sh