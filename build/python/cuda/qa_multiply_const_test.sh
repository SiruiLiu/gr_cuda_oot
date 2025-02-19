#!/usr/bin/sh
export VOLK_GENERIC=1
export GR_DONT_LOAD_PREFS=1
export srcdir=/home/dayao/Projects/gpu/gr-cuda/python/cuda
export GR_CONF_CONTROLPORT_ON=False
export PATH="/home/dayao/Projects/gpu/gr-cuda/build/python/cuda":"$PATH"
export LD_LIBRARY_PATH="":$LD_LIBRARY_PATH
export PYTHONPATH=/home/dayao/Projects/gpu/gr-cuda/build/test_modules:$PYTHONPATH
/usr/bin/python3 /home/dayao/Projects/gpu/gr-cuda/python/cuda/qa_multiply_const.py 
