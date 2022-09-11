FROM dolfinx/lab:v0.5.1
RUN PYVISTA_JUPYTER_BACKEND=static PYVISTA_OFF_SCREEN=false

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
