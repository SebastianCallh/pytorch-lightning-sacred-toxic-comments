FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

ARG PKG_DIR
ARG PRIVATE_DEPS
ARG SACRED_USER
ARG SACRED_PW

RUN apt-get update && apt-get install -y git

WORKDIR /workspace

# Install packages from private repositories
COPY ${PKG_DIR}/ /pkg/
RUN if [ "${PRIVATE_DEPS}" != "none" ]; then \
	for pkg in /pkg/*/* ; \
	do pip install $pkg ; \
	done; \
	fi

# Fix permissions
RUN chmod 0777 /workspace
RUN mkdir /.local && chmod 0777 /.local
RUN mkdir /.jupyter && chmod 0777 /.jupyter
RUN mkdir /.cache && chmod 0777 /.cache

# Install python packages
ADD src ./src
ADD requirements.txt .
RUN pip install -r requirements.txt
