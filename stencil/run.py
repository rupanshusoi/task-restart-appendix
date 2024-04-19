#!/usr/bin/python3

import subprocess as sp
import os
import shlex
from math import log2

def _run(cmd):
  if os.getenv("SLURM_JOB_ID") is not None:
    print(cmd)
    sp.run(shlex.split(cmd), check=True)
  else:
    print("SLURM_JOB_ID is not set!")
    print(cmd)

def arg_helper(val):
  assert type(val) == bool
  if val: return "ON"
  else: return "OFF"

def make_build_name(restart, unroll, prob):
  return f"stencil_{arg_helper(restart)}_{unroll}_{prob}"

def make_out_name(num_nodes, repetition, restart, unroll, prob):
  return f"out_{num_nodes}_{repetition}_{arg_helper(restart)}_{unroll}_{prob}"

def _run_stencil(restart, unroll, prob, repetition, run_dir):
  os.chdir(owd)

  num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES"))
  assert num_nodes is not None

  name = make_out_name(num_nodes, repetition, restart, unroll, prob)
  build_name = make_build_name(restart, unroll, prob)

  power = round(log2(num_nodes))
  n = 2 ** power
  nx = 2 ** ((power + 1) // 2)
  ny = 2 ** (power // 2)

  srun_args = f"srun --unbuffered -n {n} -N {n} --ntasks-per-node 1 --cpu_bind none --output {run_dir}/{name}"
  stencil_loc = f"build/{build_name}.dir/{build_name}"
  stencil_args = f"-nx {nx * 15000 * 2} -ny {ny * 15000 * 2} -ntx {nx * 2} -nty {ny * 2} -tsteps 15000 -tprune 30"
  stencil_small_args = f"-nx {nx * 15000 * 2} -ny {ny * 15000 * 2} -ntx {nx * 2} -nty {ny * 2} -tsteps 940 -tprune 30"
  legion_args = "-hl:sched 1024 1 -ll:gpu 4 -ll:io 1 -ll:util 2 -ll:bgwork 4 -ll:fsize 36000 -ll:csize 144000 -ll:zsize 36000 -ll:rsize 0 -ll:gsize 0 -lg:eager_alloc_percentage 10 -dm:memoize 1 -dm:same_address_space 1 -level runtime=5"
  legion_prof_args = f"-lg:prof {num_nodes} -lg:prof_logfile {name}_prof_%.gz"

  old = os.environ["LD_LIBRARY_PATH"]
  try:
    os.environ["LD_LIBRARY_PATH"] += f":{owd}/build/{build_name}.dir/"
    os.environ["LD_LIBRARY_PATH"] += f":/global/homes/r/rsoi/legion/language/build/lib/"
    os.environ["GASNET_BACKTRACE"] = "1"
    os.environ["FI_CXI_RX_MATCH_MODE"] = "software"
    os.environ["FI_CXI_RDZV_THRESHOLD"]= "256"
    os.environ["FI_CXI_RDZV_GET_MIN"]= "256"
    os.environ["GASNET_OFI_RECEIVE_BUFF_SIZE"]= "recv"
    os.environ["GASNET_OFI_MAX_MEDIUM"]= "8192"
    _run(f"{srun_args} {stencil_loc} {stencil_args} {legion_args} {legion_prof_args}")
  except:
    print(f"failed: {name}")
  finally:
    os.environ["LD_LIBRARY_PATH"] = old

def build_stencil(restart, unroll, prob=0):
  os.chdir(owd)
  os.chdir("build")

  name = make_build_name(restart, unroll, prob)
  build_dir = name + ".dir"

  if os.path.isdir(build_dir):
    print("skipping build...")
    return

  os.mkdir(build_dir)
  os.chdir(owd)

  os.environ["USE_FOREIGN"] = "0"
  os.environ["OBJNAME"] = f"build/{build_dir}/{name}"
  os.environ["SAVEOBJ"] = "1"
  os.environ["STANDALONE"] = "1"
  os.environ["LEGION_RESTART"] = arg_helper(restart)
  os.environ["LEGION_RESTART_UNROLL"] = str(unroll)
  os.environ["LEGION_RESTART_PROB"] = str(prob)

  regent_loc = os.path.expanduser("~/legion/language/regent.py")

  stencil_loc = os.path.expanduser("~/restart/stencil/stencil_fast.rg")

  compile_args = "-fpredicate 0 -fflow 0 -fopenmp 0 -foverride-demand-cuda 1 -foverride-demand-index-launch 1"

  _run(f"{regent_loc} {stencil_loc} {compile_args}")
  
def build_and_run_stencil(run_dir, repetitions, restart, unroll, prob=0):
  build_stencil(restart, unroll, prob)

  os.chdir(owd)
  os.chdir(run_dir)

  for r in range(repetitions):
    _run_stencil(restart, unroll, prob, r, run_dir)

def main():
  global owd
  owd = os.getcwd()

  build_and_run_stencil("runs/run1", 1, False, 0)
  build_and_run_stencil("runs/run1", 1, True, 7500)

main()
exit()
