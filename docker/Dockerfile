FROM funkey/gunpowder:v0.3-pre7
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
ENV PYTHONPATH=${PYKLB_ROOT}/build/lib.linux-x86_64-2.7:${PYTHONPATH}
RUN cp build/lib/libklb.so /lib
