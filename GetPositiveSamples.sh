export PATH=/root/buaa_target/python/bin:/root/buaa_target/python/bin/cuda/bin:$PATH
export LD_LIBRARY_PATH=/root/buaa_target/python/bin/gdal/lib:/root/buaa_target/python/bin/cuda/lib64:$LD_LIBRARY_PATH

python ./tool/CutSuitableTarget.py /inspur/LanduseChange/sample/sample_wkk_BeiHang/8Bit ./dataset/positiveSamples