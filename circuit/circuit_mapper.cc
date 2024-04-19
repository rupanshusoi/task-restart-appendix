/* Copyright 2023 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "circuit_mapper.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <map>
#include <vector>

#include "mappers/default_mapper.h"
#include "mappers/logging_wrapper.h"

using namespace Legion;
using namespace Legion::Mapping;

///
/// Mapper
///

// static LegionRuntime::Logger::Category log_circuit("circuit");

class ContigShardingFunctor: public ShardingFunctor
{
public:
  virtual ShardID shard(const DomainPoint &index_point,
                        const Domain &index_domain,
                        const size_t total_shards)
  {
    assert(index_domain.lo().point_data[0] == 0); 

    return (total_shards * index_point.point_data[0]) /
           (index_domain.hi().point_data[0] + 1); 
  }
};

class CircuitMapper : public DefaultMapper
{
public:
  CircuitMapper(MapperRuntime *rt, Machine machine, Processor local,
                const char *mapper_name,
                std::vector<Processor>* procs_list,
                std::vector<Memory>* sysmems_list,
                std::map<Memory, std::vector<Processor> >* sysmem_local_procs,
                std::map<Processor, Memory>* proc_sysmems,
                std::map<Processor, Memory>* proc_regmems);
  virtual Processor default_policy_select_initial_processor(
                                    MapperContext ctx, const Task &task);
  virtual void default_policy_select_target_processors(
                                    MapperContext ctx,
                                    const Task &task,
                                    std::vector<Processor> &target_procs);

  void select_task_options(MapperContext ctx,
                           const Task &task,
                           TaskOptions &options);

  void replicate_task(MapperContext ctx,
                      const Task &task,
                      const ReplicateTaskInput &input,
                      ReplicateTaskOutput &output);

  virtual LayoutConstraintID default_policy_select_layout_constraints(
                                    MapperContext ctx, Memory target_memory,
                                    const RegionRequirement &req,
                                    MappingKind mapping_kind,
                                    bool needs_field_constraint_check,
                                    bool &force_new_instances) override;

  void select_sharding_functor(const Mapping::MapperContext       ctx,
                               const Task&                        task,
                               const SelectShardingFunctorInput&  input,
                                     SelectShardingFunctorOutput& output);
private:
};

CircuitMapper::CircuitMapper(MapperRuntime *rt, Machine machine, Processor local,
                             const char *mapper_name,
                             std::vector<Processor>* _procs_list,
                             std::vector<Memory>* _sysmems_list,
                             std::map<Memory, std::vector<Processor> >* _sysmem_local_procs,
                             std::map<Processor, Memory>* _proc_sysmems,
                             std::map<Processor, Memory>* _proc_regmems)
  : DefaultMapper(rt, machine, local, mapper_name) //,
{
}

void CircuitMapper::select_sharding_functor(const Mapping::MapperContext ctx,
                               const Task&                        task,
                               const SelectShardingFunctorInput&  input,
                                     SelectShardingFunctorOutput& output) {
  output.chosen_functor = 1;
}


Processor CircuitMapper::default_policy_select_initial_processor(
                                    MapperContext ctx, const Task &task)
{
  return DefaultMapper::default_policy_select_initial_processor(ctx, task);
}

void CircuitMapper::replicate_task(MapperContext ctx, const Task &task, const ReplicateTaskInput &input, ReplicateTaskOutput &output)
{
      std::cout << "Replicate task: " << task.get_task_name() << std::endl;

      const Processor::Kind target_kind = task.target_proc.kind();
      // Get the variant that we are going to use to map this task
      const VariantInfo chosen = default_find_preferred_variant(task, ctx,
                        true/*needs tight bound*/, true/*cache*/, target_kind);

      assert(chosen.is_replicable);

      output.chosen_variant = chosen.variant;

      const std::vector<Processor> &remote_procs =
        remote_procs_by_kind(target_kind);

      // Only check for MPI interop case when dealing with CPUs
      assert(!((target_kind == Processor::LOC_PROC) &&
          runtime->is_MPI_interop_configured(ctx)));

      // Otherwise we can just assign shards based on address space
      // assert(total_nodes > 1);
      output.target_processors.resize(remote_procs.size());
      for (unsigned idx = 0; idx < remote_procs.size(); idx++)
	output.target_processors[idx] = remote_procs[idx];
}

LayoutConstraintID CircuitMapper::default_policy_select_layout_constraints(
                                    MapperContext ctx, Memory target_memory,
                                    const RegionRequirement &req,
                                    MappingKind mapping_kind,
                                    bool needs_field_constraint_check,
                                    bool &force_new_instances)
{
  LayoutConstraintID result = 
    DefaultMapper::default_policy_select_layout_constraints(
        ctx, target_memory, req, mapping_kind, 
        needs_field_constraint_check, force_new_instances);
  force_new_instances = true;
  return result;
}

void CircuitMapper::select_task_options(MapperContext ctx, const Task &task, TaskOptions &output)
{
  DefaultMapper::select_task_options(ctx, task, output);

  if (!strcmp(task.get_task_name(), "wrapper") || !strcmp(task.get_task_name(), "toplevel"))
    output.replicate = true;
}

void CircuitMapper::default_policy_select_target_processors(
                                    MapperContext ctx,
                                    const Task &task,
                                    std::vector<Processor> &target_procs)
{
  target_procs.push_back(task.target_proc);
}

static void create_mappers(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs)
{
  std::vector<Processor>* procs_list = new std::vector<Processor>();
  std::vector<Memory>* sysmems_list = new std::vector<Memory>();
  std::map<Memory, std::vector<Processor> >* sysmem_local_procs =
    new std::map<Memory, std::vector<Processor> >();
  std::map<Processor, Memory>* proc_sysmems = new std::map<Processor, Memory>();
  std::map<Processor, Memory>* proc_regmems = new std::map<Processor, Memory>();


  std::vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
  machine.get_proc_mem_affinity(proc_mem_affinities);

  for (unsigned idx = 0; idx < proc_mem_affinities.size(); ++idx) {
    Machine::ProcessorMemoryAffinity& affinity = proc_mem_affinities[idx];
    if (affinity.p.kind() == Processor::LOC_PROC) {
      if (affinity.m.kind() == Memory::SYSTEM_MEM) {
        (*proc_sysmems)[affinity.p] = affinity.m;
        if (proc_regmems->find(affinity.p) == proc_regmems->end())
          (*proc_regmems)[affinity.p] = affinity.m;
      }
      else if (affinity.m.kind() == Memory::REGDMA_MEM)
        (*proc_regmems)[affinity.p] = affinity.m;
    }
  }

  for (std::map<Processor, Memory>::iterator it = proc_sysmems->begin();
       it != proc_sysmems->end(); ++it) {
    procs_list->push_back(it->first);
    (*sysmem_local_procs)[it->second].push_back(it->first);
  }

  for (std::map<Memory, std::vector<Processor> >::iterator it =
        sysmem_local_procs->begin(); it != sysmem_local_procs->end(); ++it)
    sysmems_list->push_back(it->first);

  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    CircuitMapper* mapper = new CircuitMapper(runtime->get_mapper_runtime(),
                                              machine, *it, "circuit_mapper",
                                              procs_list,
                                              sysmems_list,
                                              sysmem_local_procs,
                                              proc_sysmems,
                                              proc_regmems);
    runtime->replace_default_mapper(new LoggingWrapper(mapper), *it);
  }
}

void register_mappers()
{
  Runtime::add_registration_callback(create_mappers);
  Runtime::preregister_sharding_functor(1, new ContigShardingFunctor());
}
