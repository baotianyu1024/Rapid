#!/bin/bash
if [ "$LOCAL_RANK" = 4 ]; then
  exec gdb -batch -ex run -ex 'thread apply all bt 12' --args /tmp/rapid_v100_venv/bin/python /home/longjiu.bty/Rapid/code/pcie_nccl_smoke.py
fi
exec /tmp/rapid_v100_venv/bin/python /home/longjiu.bty/Rapid/code/pcie_nccl_smoke.py
