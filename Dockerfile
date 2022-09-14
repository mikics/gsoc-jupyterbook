FROM dolfinx/lab:v0.5.1
RUN pip3 install notebook ipywidgets==7.7.2 nbconvert==6.5.3 jupyter-book --upgrade

# create user with a home directory
ARG NB_USER
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

# Copy home directory for usage in binder
WORKDIR ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}

USER ${NB_USER}

ENTRYPOINT []
