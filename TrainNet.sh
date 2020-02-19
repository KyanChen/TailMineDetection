export PATH=/root/buaa_target/python/bin:/root/buaa_target/python/bin/cuda/bin:$PATH
export LD_LIBRARY_PATH=/root/buaa_target/python/bin/gdal/lib:/root/buaa_target/python/bin/cuda/lib64:$LD_LIBRARY_PATH

python ./train_SSD.py