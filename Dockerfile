#### MIPOPT
# Kill 2 birds in one stone by making a `mipopt` image with both gurobi and cplex
FROM gurobi/optimizer:11.0.0 as mipopt

# CPLEX installer 22.1.1 and installer properties name
ARG INSTALLER=bin/cplex*.bin
ARG PROPERTIES=bin/cplex*.properties

# install location
ARG CPLEX_DIR=/opt/cplex/

# Install Java runtime. This is required by the installer
RUN apt-get update && apt-get install -y default-jre

# Copy over installer
COPY ${INSTALLER} /tmp/${INSTALLER}
COPY ${PROPERTIES} /tmp/${PROPERTIES}
RUN chmod u+x /tmp/${INSTALLER}

# install CPLEX to the system, while installing it will ask few questions
RUN /tmp/${INSTALLER} -f /tmp/${PROPERTIES}

# Remove installer, temporary files, and the JRE we installed
RUN rm -f /tmp/${INSTALLER} /tmp/${PROPERTIES}
RUN apt-get remove -y --purge default-jre && apt-get -y --purge autoremove

# Remove everything except bin, license and swidtag
RUN rm -rf \
    ${CPLEX_DIR}/concert \
    ${CPLEX_DIR}/cpoptimizer \
    ${CPLEX_DIR}/doc \
    ${CPLEX_DIR}/opl \
    ${CPLEX_DIR}/python \
    ${CPLEX_DIR}/Uninstall \
    ${CPLEX_DIR}/cplex/examples \
    ${CPLEX_DIR}/cplex/include \
    ${CPLEX_DIR}/cplex/python \
    ${CPLEX_DIR}/cplex/lib \
    ${CPLEX_DIR}/cplex/matlab \
    ${CPLEX_DIR}/cplex/readmeUNIX.html

#### JUMIP
# Now move the optimization binaries into a clean julia image
FROM julia:1.10.0

# copy over opt folder
COPY --from=mipopt /opt/ /opt/

# add in environement variables for Julia to happy
ENV CPLEX_STUDIO_BINARIES=/opt/cplex/cplex/bin/x86-64_linux/
ENV GUROBI_HOME=/opt/gurobi/linux64/

# add in Revise & OhMyRepl with startup file
RUN julia -e 'using Pkg; Pkg.add("Revise")'
RUN julia -e 'using Pkg; Pkg.add("OhMyREPL")'

COPY bin/startup.jl /root/.julia/config/startup.jl

##### Activate julia environment
# We ONLY copy over the environment setups
# We use volumes in the compose file to have a nice development experience
ARG PROJ_DIR
COPY Manifest.toml ${PROJ_DIR}/Manifest.toml
COPY Project.toml ${PROJ_DIR}/Project.toml

ENV JULIA_PROJECT=${PROJ_DIR}
RUN julia -e 'using Pkg; Pkg.instantiate()'
RUN julia -e 'using Pkg; Pkg.status()'
RUN julia -e 'println(Base.active_project())'
