FROM registry.insight-centre.org/sit/mps/docker-images/base-services-tf-gpu:latest

## install only the service requirements
ADD ./Pipfile /service/Pipfile
ADD ./requirements.txt /service/requirements.txt
ADD ./requirements-arm64.txt /service/requirements-arm64.txt
ADD ./multi-arch-pip-install.sh /service/multi-arch-pip-install.sh
ADD ./setup.py /service/setup.py
RUN mkdir -p /service/object_detection_service/ && \
    touch /service/object_detection_service/__init__.py
WORKDIR /service

# Using requirements only, since pipenv is getting messy in the docker context... oh god, how I wanted to change to Poetry right now...
# RUN rm -f Pipfile.lock && pipenv lock -vvv && pipenv --rm && \
#     pipenv install --system  && \
#     rm -rf /tmp/pip* /root/.cache
# RUN pip install -r requirements.txt -i https://pypi.org/simple --extra-index-url https://${SIT_PYPI_USER}:${SIT_PYPI_PASS}@sit-pypi.herokuapp.com/simple && \
#     rm -rf /tmp/pip* /root/.cache
RUN ./multi-arch-pip-install.sh


## add all the rest of the code and install the actual package
## this should keep the cached layer above if no change to the pipfile or setup.py was done.
ADD . /service
RUN pip install -e . && \
    pip install -e ./tf_od_models && \
    rm -rf /tmp/pip* /root/.cache
