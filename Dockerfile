FROM balenalib/fincm3-python

# working directory
WORKDIR /usr/src/app

# update and upgrade
RUN apt update -qy && apt upgrade -qy

# install dependencies
RUN apt install git \ 
     build-essential \
     unzip \
     wget \
     python3-dev \
     python3-pip \
     python3-setuptools \
     swig \
     libatlas-base-dev \
     libi2c-dev \
     libavutil-dev \
     libavcodec-dev \
     libavformat-dev \
     libsdl2-dev

# install pillow and tensorflow lite runtime
RUN pip3 install pillow qrcode \
    https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl

# download and build MLX90640 thermal camera library
RUN git clone https://github.com/pimoroni/mlx90640-library.git
RUN cd mlx90640-library && make clean && make I2C_MODE=LINUX && make install
RUN cd mlx90640-library/python/library && make build && make install

RUN wget -q ftp://u45902898-ide:Ftd1%24erv@ftp.ftdichip.com/CES/Forum/BRT_AN_025_Beta/BRT_AN_025_Source_BETA.zip
RUN unzip BRT_AN_025_Source_BETA.zip -d BRT_AN_025 && \
    cp -r BRT_AN_025/Examples/RaspberryPi/lib . && \
    rm -rf BRT_AN_025 BRT_AN_025_Source_BETA.zip

COPY . .
RUN cp EVE_config.h lib/eve/include && make

# execute main.py on start
CMD ["python3", "main.py"]
