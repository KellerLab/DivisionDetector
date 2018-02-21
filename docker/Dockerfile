FROM funkey/gunpowder:v0.3-pre6
LABEL maintainer jfunke@iri.upc.edu

# install python wrapper for KLB reader
ENV PYKLB_ROOT=/src/pyklb
ENV PYKLB_REPOSITORY=https://github.com/bhoeckendorf/pyklb
ENV PYKLB_REVISION=870874e26129411382323467d62038456bcb0385

RUN pip install wheel
WORKDIR ${PYKLB_ROOT}
RUN git clone ${PYKLB_REPOSITORY} . && \
    git checkout ${PYKLB_REVISION}
RUN python setup.py bdist_wheel
RUN pip install dist/pyklb-0.0.3.dev0-cp27-cp27mu-linux_x86_64.whl
ENV PYTHONPATH ${PYKLB_ROOT}:$PYTHONPATH
