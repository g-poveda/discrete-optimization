#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from heapq import heappop, heappush
from typing import Any

import networkx as nx
import numpy as np

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.solvers import RcpspSolver
from discrete_optimization.rcpsp.utils import compute_graph_rcpsp

logger = logging.getLogger(__name__)


class CpmObject:
    def __init__(self, ESD, EFD, LSD, LFD):
        self._ESD = ESD
        self._EFD = EFD
        self._LSD = LSD
        self._LFD = LFD

    def set_earliest_start_date(self, ESD):
        self._ESD = ESD

    def set_earliest_finish_date(self, EFD):
        self._EFD = EFD

    def set_latest_start_date(self, LSD):
        self._LSD = LSD

    def set_latest_finish_date(self, LFD):
        self._LFD = LFD

    def __str__(self):
        return str({k: getattr(self, k) for k in self.__dict__.keys()})


class CpmRcpspSolver(RcpspSolver):
    problem: RcpspProblem

    def __init__(
        self,
        problem: RcpspProblem,
        params_objective_function: ParamsObjectiveFunction = None,
        **kwargs,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.graph = compute_graph_rcpsp(problem)
        self.graph_nx = self.graph.to_networkx()
        self.source = problem.source_task
        self.sink = problem.sink_task
        self.map_node: dict[Any, CpmObject] = {
            n: CpmObject(None, None, None, None) for n in self.graph_nx.nodes()
        }
        self.node_to_index = {
            node: i_node for i_node, node in enumerate(self.graph_nx.nodes())
        }
        self.index_to_node = {
            i_node: node for node, i_node in self.node_to_index.items()
        }
        successors = {
            n: nx.algorithms.descendants(self.graph_nx, n)
            for n in self.graph_nx.nodes()
        }
        self.immediate_successors = {
            n: set(nx.neighbors(self.graph_nx, n)) for n in self.graph_nx.nodes()
        }
        self.immediate_predecessors = {
            n: set(self.graph_nx.predecessors(n)) for n in self.graph_nx.nodes()
        }
        self.successors_map = {}
        self.predecessors_map = {}
        for k in successors:
            self.successors_map[k] = {"succs": successors[k], "nb": len(successors[k])}
        predecessors = {
            n: nx.algorithms.ancestors(self.graph_nx, n) for n in self.graph_nx.nodes()
        }
        for k in predecessors:
            self.predecessors_map[k] = {
                "succs": predecessors[k],
                "nb": len(predecessors[k]),
            }

    def run_classic_cpm(self):
        done_forward = set()
        done_backward = set()
        current_pred = {
            k: {
                "succs": set(self.predecessors_map[k]["succs"]),
                "nb": self.predecessors_map[k]["nb"],
            }
            for k in self.predecessors_map
        }
        available_activities = {
            n
            for n in current_pred
            if n not in done_forward and current_pred[n]["nb"] == 0
        }
        queue = [(0, self.node_to_index[n]) for n in available_activities]
        forward = True
        while queue:
            time, i_node = heappop(queue)
            node = self.index_to_node[i_node]
            if forward and node in done_forward:
                continue
            elif not forward and node in done_backward:
                continue
            if forward:
                self.map_node[node].set_earliest_start_date(time)
                done_forward.add(node)
            else:
                self.map_node[node].set_latest_finish_date(-time)
                done_backward.add(node)
            min_duration = min(
                [
                    self.problem.mode_details[node][k]["duration"]
                    for k in self.problem.mode_details[node]
                ]
            )
            if forward:
                self.map_node[node].set_earliest_finish_date(time + min_duration)
            else:
                self.map_node[node].set_latest_start_date(-time - min_duration)
            if forward:
                next_nodes = self.immediate_successors[node]
            else:
                next_nodes = self.immediate_predecessors[node]
            for next_node in next_nodes:
                pred = (
                    self.immediate_predecessors[next_node]
                    if forward
                    else self.immediate_successors[next_node]
                )
                if forward:
                    if all(self.map_node[n]._ESD is not None for n in pred):
                        max_esd = max([self.map_node[n]._EFD for n in pred])
                        heappush(queue, (max_esd, self.node_to_index[next_node]))
                else:
                    if all(self.map_node[n]._LSD is not None for n in pred):
                        max_esd = min([self.map_node[n]._LSD for n in pred])
                        heappush(queue, (-max_esd, self.node_to_index[next_node]))
            if node == self.sink:
                forward = False
                heappush(queue, (-self.map_node[node]._EFD, self.node_to_index[node]))

        critical_path = [self.sink]
        cur_node = self.sink
        while cur_node is not self.source:
            nodes = [
                n
                for n in self.immediate_predecessors[cur_node]
                if self.map_node[n]._ESD == self.map_node[n]._LSD
                and self.map_node[n]._EFD == self.map_node[cur_node]._ESD
            ]
            cur_node = nodes[0]
            critical_path += [cur_node]
        return critical_path[::-1]

    def return_order_cpm(self):
        order = sorted(
            self.map_node,
            key=lambda x: (
                self.map_node[x]._LSD,
                self.map_node[x]._LSD - self.map_node[x]._ESD,
            ),
        )
        return order

    def run_sgs_on_order(
        self,
        map_nodes: dict[Any, CpmObject],
        critical_path: list[Any],
        total_order: list[Any] = None,
        cut_sgs_by_critical=True,
    ):
        if total_order is None:
            total_order = self.return_order_cpm()
        index_in_order = {total_order[i]: i for i in range(len(total_order))}
        resource_avail_in_time = {}
        for res in list(self.problem.resources.keys()):
            if self.problem.is_varying_resource():
                resource_avail_in_time[res] = self.problem.resources[res][
                    : self.problem.horizon + 1
                ]
            else:
                resource_avail_in_time[res] = np.full(
                    self.problem.horizon,
                    self.problem.resources[res],
                    dtype=np.int_,
                ).tolist()
        done = set()
        ressource_usage = {
            res: {time: {} for time in range(self.problem.horizon)}
            for res in self.problem.resources.keys()
        }
        current_schedule = {}
        min_time_to_schedule = {n: self.map_node[n]._ESD for n in self.map_node}
        if cut_sgs_by_critical:
            index_critical = 0
            cur_critical_task_to_schedule = critical_path[index_critical]
            sorted_task_to_do_before = sorted(
                [
                    n
                    for n in self.predecessors_map[cur_critical_task_to_schedule][
                        "succs"
                    ]
                    if n not in done
                ],
                key=lambda x: index_in_order[x],
            )
        effects_on_delay = {}
        causes_of_delay = {}
        unlock_task_transition = {}
        while True:
            if cut_sgs_by_critical:
                sorted_task_to_do = sorted_task_to_do_before + [
                    cur_critical_task_to_schedule
                ]
            else:
                sorted_task_to_do = [t for t in total_order if t not in done]
            ll = list(sorted_task_to_do)
            while len(ll) > 0:
                j = next(
                    t
                    for t in ll
                    if all(task in done for task in self.immediate_predecessors[t])
                )
                ll.remove(j)
                early_start = min_time_to_schedule[j]
                ressource_consumption = {
                    r: self.problem.mode_details[j][1][r]
                    for r in self.problem.mode_details[j][1]
                    if r != "duration"
                }
                duration = self.problem.mode_details[j][1]["duration"]
                delayed_du_to_ressource = False
                for time in range(early_start, self.problem.horizon):
                    valid = True
                    for res in resource_avail_in_time:
                        for t in range(time, time + duration):
                            if resource_avail_in_time[res][
                                t
                            ] < ressource_consumption.get(res, 0):
                                valid = False
                                delayed_du_to_ressource = True
                                if j not in causes_of_delay:
                                    causes_of_delay[j] = {"res_t_other_task": []}
                                causes_of_delay[j]["res_t_other_task"] += [
                                    (
                                        res,
                                        t,
                                        set(
                                            [
                                                task
                                                for task in ressource_usage[res][t]
                                                if task
                                                not in self.predecessors_map[j]["succs"]
                                            ]
                                        ),
                                    )
                                ]
                                break
                        if not valid:
                            break
                    if valid:
                        if delayed_du_to_ressource:
                            ressource_blocking = [
                                res
                                for res in resource_avail_in_time
                                if resource_avail_in_time[res][time - 1]
                                < ressource_consumption.get(res, 0)
                            ]
                            task_blocking = [
                                task
                                for task in current_schedule
                                if current_schedule[task]["end_time"] == time
                                and any(
                                    res in ressource_blocking
                                    for res in ressource_usage
                                    if ressource_usage[res][time - 1].get(task, 0) > 0
                                )
                            ]
                            if j not in unlock_task_transition:
                                unlock_task_transition[j] = set()
                            unlock_task_transition[j].update(set(task_blocking))
                        current_schedule[j] = {
                            "start_time": time,
                            "end_time": time + duration,
                        }
                        done.add(j)
                        for res in ressource_consumption:
                            for t in range(time, time + duration):
                                resource_avail_in_time[res][t] -= ressource_consumption[
                                    res
                                ]
                                if ressource_consumption[res] > 0:
                                    ressource_usage[res][t][j] = ressource_consumption[
                                        res
                                    ]
                        for task in self.successors_map[j]["succs"]:
                            prev = min_time_to_schedule[task]
                            min_time_to_schedule[task] = max(
                                min_time_to_schedule[task],
                                current_schedule[j]["end_time"],
                            )
                            if (
                                min_time_to_schedule[task] > prev
                                and min_time_to_schedule[task]
                                > self.map_node[task]._LSD
                            ):
                                if task not in effects_on_delay:
                                    effects_on_delay[task] = {"task_causes": set()}
                                effects_on_delay[task]["task_causes"].add(
                                    j
                                )  # the task is delayed
                                # at least because of j
                        break
            if cut_sgs_by_critical:
                index_critical += 1
                if index_critical == len(critical_path):
                    break
                cur_critical_task_to_schedule = critical_path[index_critical]
                sorted_task_to_do_before = sorted(
                    [
                        n
                        for n in self.predecessors_map[cur_critical_task_to_schedule][
                            "succs"
                        ]
                        if n not in done
                    ],
                    key=lambda x: index_in_order[x],
                )
            else:
                break

        resource_links_to_add = []
        for j in causes_of_delay:
            delayed = current_schedule[j]["start_time"] > self.map_node[j]._ESD
            if delayed:
                for res, time, set_task in causes_of_delay[j]["res_t_other_task"]:
                    if time >= self.map_node[j]._LSD - 5:
                        for task in set_task:
                            resource_links_to_add += [(j, task)]
        logger.debug(f"Final time : {current_schedule[critical_path[-1]]}")
        self.unlock_task_transition = unlock_task_transition
        return (
            current_schedule,
            resource_links_to_add,
            effects_on_delay,
            causes_of_delay,
        )

    def get_first_time_to_do_one_task(self, resource_avail_in_time, task_id):
        early_start = self.map_node[task_id]._ESD
        ressource_consumption = {
            r: self.problem.mode_details[task_id][1][r]
            for r in self.problem.mode_details[task_id][1]
            if r != "duration"
        }
        duration = self.problem.mode_details[task_id][1]["duration"]
        time_start = None
        for time in range(early_start, self.problem.horizon):
            valid = True
            for res in resource_avail_in_time:
                for t in range(time, time + duration):
                    if resource_avail_in_time[res][t] < ressource_consumption[res]:
                        valid = False
                        time_start = time
                        break
                if not valid:
                    break
        return time_start, time_start + duration

    def run_sgs_time_loop(
        self,
        map_nodes: dict[Any, CpmObject],
        critical_path: list[Any],
        total_order: list[Any] = None,
    ):
        if total_order is None:
            total_order = self.return_order_cpm()
        resource_avail_in_time = {}
        for res in list(self.problem.resources.keys()):
            if self.problem.is_varying_resource():
                resource_avail_in_time[res] = self.problem.resources[res][
                    : self.problem.horizon + 1
                ]
            else:
                resource_avail_in_time[res] = np.full(
                    self.problem.horizon,
                    self.problem.resources[res],
                    dtype=np.int_,
                ).tolist()
        done = set()
        ressource_usage = {
            res: {time: {} for time in range(self.problem.horizon)}
            for res in self.problem.resources.keys()
        }
        current_schedule = {}
        min_time_to_schedule = {n: self.map_node[n]._ESD for n in self.map_node}
        effects_on_delay = {}
        causes_of_delay = {}
        cur_time = 0
        nb_task = len(self.map_node)
        while len(done) < nb_task:
            sorted_task_to_do = [t for t in total_order if t not in done]
            index = 0
            for j in sorted_task_to_do[:10]:
                early_start = min_time_to_schedule[j]
                if all(
                    t in done and current_schedule[t]["end_time"] <= cur_time
                    for t in self.immediate_predecessors[j]
                ):
                    if early_start <= cur_time:
                        ressource_consumption = {
                            r: self.problem.mode_details[j][1][r]
                            for r in self.problem.mode_details[j][1]
                            if r != "duration"
                        }
                        duration = self.problem.mode_details[j][1]["duration"]
                        for time in [cur_time]:
                            valid = True
                            for res in resource_avail_in_time:
                                for t in range(time, time + duration):
                                    if (
                                        resource_avail_in_time[res][t]
                                        < ressource_consumption[res]
                                    ):
                                        valid = False
                                        if j not in causes_of_delay:
                                            causes_of_delay[j] = {
                                                "res_t_other_task": []
                                            }
                                        causes_of_delay[j]["res_t_other_task"] += [
                                            (
                                                res,
                                                t,
                                                set(
                                                    [
                                                        task
                                                        for task in ressource_usage[
                                                            res
                                                        ][t]
                                                        if task
                                                        not in self.predecessors_map[j][
                                                            "succs"
                                                        ]
                                                    ]
                                                ),
                                            )
                                        ]
                                    break
                            if not valid:
                                break
                        if valid:
                            current_schedule[j] = {
                                "start_time": time,
                                "end_time": time + duration,
                            }
                            done.add(j)
                            for res in resource_avail_in_time:
                                for t in range(time, time + duration):
                                    resource_avail_in_time[res][
                                        t
                                    ] -= ressource_consumption[res]
                                    if ressource_consumption[res] > 0:
                                        ressource_usage[res][t][
                                            j
                                        ] = ressource_consumption[res]
                            for task in self.successors_map[j]["succs"]:
                                prev = min_time_to_schedule[task]
                                min_time_to_schedule[task] = max(
                                    min_time_to_schedule[task],
                                    current_schedule[j]["end_time"],
                                )
                                if (
                                    min_time_to_schedule[task] > prev
                                    and min_time_to_schedule[task]
                                    > self.map_node[task]._LSD
                                ):
                                    if task not in effects_on_delay:
                                        effects_on_delay[task] = {"task_causes": set()}
                                    effects_on_delay[task]["task_causes"].add(
                                        j
                                    )  # the task is delayed
                                    # at least because of j
                            break
                index += 1
            cur_time += 1
        resource_links_to_add = []
        for j in causes_of_delay:
            delayed = current_schedule[j]["start_time"] > self.map_node[j]._ESD
            if delayed:
                for res, time, set_task in causes_of_delay[j]["res_t_other_task"]:
                    if time >= self.map_node[j]._LSD - 5:
                        for task in set_task:
                            resource_links_to_add += [(j, task)]
        logger.debug(f"Final time : {current_schedule[critical_path[-1]]}")
        return (
            current_schedule,
            resource_links_to_add,
            effects_on_delay,
            causes_of_delay,
        )

    def solve(self, **kwargs) -> ResultStorage:
        cpath = self.run_classic_cpm()
        order = sorted(
            self.map_node,
            key=lambda x: (
                self.map_node[x]._LSD,
                self.map_node[x]._LSD - self.map_node[x]._ESD,
            ),
        )
        permutation_sgs = [
            self.problem.index_task_non_dummy[o]
            for o in order
            if o in self.problem.index_task_non_dummy
        ]
        solution_sgs_0 = RcpspSolution(
            problem=self.problem,
            rcpsp_permutation=permutation_sgs,
            rcpsp_modes=[1 for i in range(self.problem.n_jobs_non_dummy)],
        )
        fit_0 = self.aggreg_from_sol(solution_sgs_0)
        order = sorted(
            self.map_node,
            key=lambda x: (
                self.map_node[x]._LSD,
                self.map_node[x]._LSD - self.map_node[x]._ESD,
            ),
        )
        (
            schedule,
            link_to_add,
            effects_on_delay,
            causes_of_delay,
        ) = self.run_sgs_on_order(
            map_nodes=self.map_node, critical_path=cpath, total_order=order
        )
        solution_1 = RcpspSolution(
            problem=self.problem,
            rcpsp_schedule=schedule,
            rcpsp_modes=[1 for i in range(self.problem.n_jobs_non_dummy)],
        )
        fit_1 = self.aggreg_from_sol(solution_1)
        res = self.create_result_storage(
            [(solution_sgs_0, fit_0), (solution_1, fit_1)],
        )
        return res


def run_partial_classic_cpm(partial_schedule, cpm_solver):
    done_forward = set()
    done_backward = set()
    current_pred = {
        k: {
            "succs": set(cpm_solver.predecessors_map[k]["succs"]),
            "nb": cpm_solver.predecessors_map[k]["nb"],
        }
        for k in cpm_solver.predecessors_map
    }
    map_node: dict[Any, CpmObject] = {
        n: CpmObject(None, None, None, None) for n in cpm_solver.graph_nx.nodes()
    }
    forward = True

    for task in partial_schedule:
        map_node[task].set_earliest_start_date(partial_schedule[task][0])
        map_node[task].set_earliest_finish_date(partial_schedule[task][1])
        map_node[task].set_latest_start_date(partial_schedule[task][0])
        map_node[task].set_latest_finish_date(partial_schedule[task][1])
        done_forward.add(task)
        done_backward.add(task)
    available_activities = {
        n for n in current_pred if n not in done_forward and current_pred[n]["nb"] == 0
    }
    queue = [(0, n) for n in available_activities]
    for task in partial_schedule:
        next_nodes = cpm_solver.immediate_successors[task]
        for next_node in next_nodes:
            pred = (
                cpm_solver.immediate_predecessors[next_node]
                if forward
                else cpm_solver.immediate_successors[next_node]
            )
            if forward:
                if all(map_node[n]._ESD is not None for n in pred):
                    max_esd = max([map_node[n]._EFD for n in pred])
                    heappush(queue, (max_esd, next_node))
    while queue:
        time, node = heappop(queue)
        if forward and node in done_forward:
            continue
        elif not forward and node in done_backward:
            continue
        if forward:
            map_node[node].set_earliest_start_date(time)
            done_forward.add(node)
        else:
            map_node[node].set_latest_finish_date(-time)
            done_backward.add(node)
        min_duration = min(
            [
                cpm_solver.problem.mode_details[node][k]["duration"]
                for k in cpm_solver.problem.mode_details[node]
            ]
        )
        if forward:
            map_node[node].set_earliest_finish_date(time + min_duration)
        else:
            map_node[node].set_latest_start_date(-time - min_duration)
        if forward:
            next_nodes = cpm_solver.immediate_successors[node]
        else:
            next_nodes = cpm_solver.immediate_predecessors[node]
        for next_node in next_nodes:
            pred = (
                cpm_solver.immediate_predecessors[next_node]
                if forward
                else cpm_solver.immediate_successors[next_node]
            )
            if forward:
                if all(map_node[n]._ESD is not None for n in pred):
                    max_esd = max([map_node[n]._EFD for n in pred])
                    heappush(queue, (max_esd, next_node))
            else:
                if all(map_node[n]._LSD is not None for n in pred):
                    max_esd = min([map_node[n]._LSD for n in pred])
                    heappush(queue, (-max_esd, next_node))
        if node == cpm_solver.problem.sink_task:
            forward = False
            heappush(queue, (-map_node[node]._EFD, node))
    critical_path = [cpm_solver.problem.sink_task]
    cur_node = cpm_solver.problem.sink_task
    while cur_node is not cpm_solver.problem.source_task:
        nodes = [
            n
            for n in cpm_solver.immediate_predecessors[cur_node]
            if map_node[n]._ESD == map_node[n]._LSD
            and map_node[n]._EFD == map_node[cur_node]._ESD
        ]
        if len(nodes) == 0:
            break
        cur_node = nodes[0]
        critical_path += [cur_node]
    return critical_path[::-1], map_node
