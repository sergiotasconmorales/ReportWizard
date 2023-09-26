To fix error from bitsandbytes while low_resources=True, go to environment/lib/pythonXXX/site-packages/bitsandbytes/ and run:
cp libbitsandbytes_cuda117.so libbitsandbytes_cpu.so