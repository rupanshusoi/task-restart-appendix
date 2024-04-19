#!/usr/bin/python3

import subprocess as sp
import os
import shlex

def _run(cmd):
  if os.getenv("SLURM_JOB_ID") is not None:
    print(cmd)
    sp.run(shlex.split(cmd), check=True)

def arg_helper(val):
  assert type(val) == bool
  if val: return "ON"
  else: return "OFF"

def make_build_name(restart, unroll, prob):
  return f"circuit_{arg_helper(restart)}_{unroll}_{prob}"

def make_out_name(num_nodes, repetition, restart, unroll, prob):
  return f"out_{num_nodes}_{repetition}_{arg_helper(restart)}_{unroll}_{prob}"

def _run_circuit(restart, unroll, prob, repetition, run_dir):
  os.chdir(owd)

  num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES"))
  assert num_nodes is not None

  name = make_out_name(num_nodes, repetition, restart, unroll, prob)
  build_name = make_build_name(restart, unroll, prob)

  srun_args = f"srun --unbuffered -n {num_nodes} -N {num_nodes} --ntasks-per-node 1 --cpu_bind none --output {run_dir}/{name}"
  circuit_loc = f"build/{build_name}.dir/{build_name}"
  circuit_args = f"-npp 5000 -wpp 20000 -l 11940 -p {num_nodes * 10} -pps 10 -prune 30"
  legion_args = "-hl:sched 1024 1 -ll:gpu 1 -ll:io 1 -ll:util 2 -ll:bgwork 4 -ll:csize 15000 -ll:fsize 15000 -ll:zsize 15000 -ll:rsize 0 -ll:gsize 0 -lg:eager_alloc_percentage 10 -lg:no_tracing -level runtime=5"
  legion_prof_args = f"-lg:prof {num_nodes} -lg:prof_logfile {name}_prof_%.gz"

  old = os.environ["LD_LIBRARY_PATH"]
  try:
    os.environ["LD_LIBRARY_PATH"] += f":{owd}/build/{build_name}.dir/"
    _run(f"{srun_args} {circuit_loc} {circuit_args} {legion_args}")
  except:
    print(f"failed: {name}")
  finally:
    os.environ["LD_LIBRARY_PATH"] = old

def build_circuit(restart, unroll, prob=0):
  os.chdir(owd)
  os.chdir("build")

  name = make_build_name(restart, unroll, prob)
  build_dir = name + ".dir"

  if os.path.isdir(build_dir):
    print("skipping build...")
    return

  os.mkdir(build_dir)
  os.chdir(owd)

  os.environ["OBJNAME"] = f"build/{build_dir}/{name}"
  os.environ["SAVEOBJ"] = "1"
  os.environ["STANDALONE"] = "1"
  os.environ["LEGION_RESTART"] = arg_helper(restart)
  os.environ["LEGION_RESTART_UNROLL"] = str(unroll)
  os.environ["LEGION_RESTART_PROB"] = str(prob)

  regent_loc = os.path.expanduser("~/legion/language/regent.py")
  circuit_loc = os.path.expanduser("~/restart/regent-circuit/circuit_sparse.rg")
  compile_args = "-fpredicate 0 -fflow 0 -fopenmp 0 -foverride-demand-index-launch 1"

  _run(f"{regent_loc} {circuit_loc} {compile_args}")
  
def build_and_run_circuit(run_dir, repetitions, restart, unroll, prob=0):
  build_circuit(restart, unroll, prob)

  os.chdir(owd)
  os.chdir(run_dir)

  for r in range(repetitions):
    _run_circuit(restart, unroll, prob, r, run_dir)

def main():
  global owd
  owd = os.getcwd()

  build_and_run_circuit("runs/run1", 1, False, 6000)
  build_and_run_circuit("runs/run1", 1, True, 6000)

main()
exit()
