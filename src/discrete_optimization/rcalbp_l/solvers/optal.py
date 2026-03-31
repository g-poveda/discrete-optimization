#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any

from discrete_optimization.generic_tasks_tools.solvers.optalcp_tasks_solver import (
    AllocationOptalSolver,
    SchedulingOptalSolver,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.rcalbp_l.problem import (
    RCALBPLProblem,
    RCALBPLSolution,
    Task,
    WorkStation,
)

try:
    import optalcp as cp
except ImportError:
    pass


class OptalRCALBPLSolver(
    AllocationOptalSolver[Task, WorkStation],
    SchedulingOptalSolver[Task],
    WarmstartMixin,
):
    problem: RCALBPLProblem
    variables: dict

    def init_model(
        self,
        minimize_used_cycle_time: bool = False,
        add_heuristic_constraint: bool = True,
        **kwargs: Any,
    ) -> None:
        super().init_model(**kwargs)
        self.cp_model = cp.Model(name="RCALBPL")
        self.variables = {}

        # 1. Create variables
        self.create_main_unfolded_intervals()
        self.create_resource_dispatch()
        self.create_cycle_time_variables()
        # 2. Post constraints
        self.constraint_only_one_station_allocation()
        self.create_cumulative_resource_constraint()
        self.create_zone_blocking()
        self.create_precedence_constraints()
        if add_heuristic_constraint:
            self.create_heuristic_target_reached_constraints(apply_heuristic=True)
        # 3. Post objective
        self.objective_value(minimize_used_cycle_time)

    def create_main_unfolded_intervals(self):
        dict_main_intervals = {}
        dict_opt_intervals = {}
        allocations = {}
        max_horizon = self.problem.c_max

        # Iterate over the periods list (not just integer)
        for p in self.problem.periods:
            # Unfold time ONLY by period. All workstations run in parallel within this window.
            p_lb_start = 0
            p_ub_start = max_horizon
            for task in self.problem.tasks:
                possible_durations = [
                    self.problem.get_duration(task, p, w) for w in self.problem.stations
                ]
                min_dur = min(possible_durations)
                max_dur = max(possible_durations)
                dict_main_intervals[(task, p)] = self.cp_model.interval_var(
                    start=(p_lb_start, p_ub_start),
                    end=(p_lb_start, p_ub_start),
                    length=(min_dur, max_dur),
                    optional=False,
                    name=f"{task}_{p}_interval_unfolded",
                )
                for w in self.problem.stations:
                    dur = self.problem.get_duration(task, p, w)
                    # Opt intervals share the exact same time window allowing parallel workstation overlap
                    opt_var = self.cp_model.interval_var(
                        start=(p_lb_start, p_ub_start),
                        end=(p_lb_start, p_ub_start),
                        length=dur,
                        optional=True,
                        name=f"{task}_{p}_{w}_interval_unfolded",
                    )
                    dict_opt_intervals[(task, p, w)] = opt_var
                    # Populate allocations dictionary with presence variables
                    allocations[(task, p, w)] = self.cp_model.presence(opt_var)
                    if p > self.problem.periods[0]:
                        self.cp_model.enforce(
                            allocations[(task, p, w)]
                            == allocations[(task, self.problem.periods[0], w)]
                        )
                # Link main interval with its workstation alternatives
                self.cp_model.alternative(
                    dict_main_intervals[(task, p)],
                    [dict_opt_intervals[(task, p, w)] for w in self.problem.stations],
                )

        self.variables["main_intervals"] = dict_main_intervals
        self.variables["opt_intervals"] = dict_opt_intervals
        self.variables["allocations"] = allocations

    def constraint_only_one_station_allocation(self):
        for task in self.problem.tasks:
            is_allocated = []
            for w in self.problem.stations:
                allocated_to_station = self.cp_model.max(
                    [
                        self.variables["allocations"][(task, p, w)]
                        for p in self.problem.periods
                    ]
                )
                is_allocated.append(allocated_to_station)
            self.cp_model.enforce(self.cp_model.sum(is_allocated) == 1)

    def create_resource_dispatch(self):
        resource = {}
        for r in self.problem.resources:
            capa = self.problem.capa_resources[r]
            for w in self.problem.stations:
                resource[(r, w)] = self.cp_model.int_var(
                    min=0, max=capa, name=f"capa_{r}_station_{w}"
                )
            self.cp_model.enforce(
                self.cp_model.sum([resource[r, w] for w in self.problem.stations])
                <= capa
            )
        self.variables["resource_dispatch"] = resource

    def create_cumulative_resource_constraint(self):
        max_horizon = self.problem.c_max
        p_lb = 0
        p_ub = max_horizon
        dummy_interval = self.cp_model.interval_var(
            start=p_lb,
            end=p_ub,
            length=max_horizon,
            optional=False,
            name=f"dummy_interval",
        )
        self.variables["dummy_interval"] = dummy_interval
        for r in self.problem.resources:
            capa = self.problem.capa_resources[r]
            for p in self.problem.periods:
                # Time bounds isolating this period

                # 1. Global capacity bound across all workstations
                pulses_global = []
                for task in self.problem.tasks:
                    req = self.problem.cons_resources[r][task]  # Indexing is [r][task]
                    if req > 0:
                        pulses_global.append(
                            self.cp_model.pulse(
                                self.variables["main_intervals"][(task, p)], req
                            )
                        )
                if pulses_global:
                    self.cp_model.enforce(self.cp_model.sum(pulses_global) <= capa)
                # 2. Local capacity bound per workstation (dispatch limit)
                for w in self.problem.stations:
                    pulses_local = []
                    for task in self.problem.tasks:
                        req = self.problem.cons_resources[r][task]
                        if req > 0:
                            pulses_local.append(
                                self.cp_model.pulse(
                                    self.variables["opt_intervals"][(task, p, w)], req
                                )
                            )

                    # Dummy interval trick: Consumes the unallocated portion of the resource dispatch
                    unallocated = capa - self.variables["resource_dispatch"][(r, w)]
                    pulses_local.append(
                        self.cp_model.pulse(dummy_interval, unallocated)
                    )
                    self.cp_model.enforce(self.cp_model.sum(pulses_local) <= capa)
        # 3. Zone capacities (local to workstations)
        for z in self.problem.zones:
            capa = self.problem.capa_zones[z]
            for p in self.problem.periods:
                for w in self.problem.stations:  # Zones are evaluated per workstation
                    pulses_zone = []
                    for task in self.problem.tasks:
                        req = self.problem.cons_zones[z][task]  # Indexing [z][task]
                        if req > 0:
                            pulses_zone.append(
                                self.cp_model.pulse(
                                    self.variables["opt_intervals"][(task, p, w)], req
                                )
                            )
                    if pulses_zone:
                        self.cp_model.enforce(self.cp_model.sum(pulses_zone) <= capa)

    def create_zone_blocking(self):
        for z in self.problem.zones:
            tasks_blocking = [
                t for t in self.problem.tasks if z in self.problem.neutr_zones[t]
            ]
            tasks_consuming = [
                t
                for t in self.problem.tasks
                if self.problem.cons_zones[z][t] > 0
                if t not in tasks_blocking
            ]
            # Since the zones have a capacity of 1 this sum threshold logic works perfectly
            if self.problem.capa_zones[z] == 1 and tasks_blocking:
                for p in self.problem.periods:
                    for w in self.problem.stations:
                        pulses = [
                            self.cp_model.pulse(
                                self.variables["opt_intervals"][(t, p, w)], 1
                            )
                            for t in tasks_blocking
                        ]
                        pulses.extend(
                            [
                                self.cp_model.pulse(
                                    self.variables["opt_intervals"][(t, p, w)],
                                    len(tasks_blocking),
                                )
                                for t in tasks_consuming
                            ]
                        )

                        if pulses:
                            self.cp_model.enforce(
                                self.cp_model.sum(pulses) <= len(tasks_blocking)
                            )

    def create_precedence_constraints(self):
        for t1, t2 in self.problem.precedences:
            # 1. Global Station Precedence: wks(t1) <= wks(t2)
            first_period = self.problem.periods[0]
            wks_t1 = self.cp_model.sum(
                [
                    w * self.variables["allocations"][(t1, first_period, w)]
                    for w in self.problem.stations
                ]
            )
            wks_t2 = self.cp_model.sum(
                [
                    w * self.variables["allocations"][(t2, first_period, w)]
                    for w in self.problem.stations
                ]
            )
            self.cp_model.enforce(wks_t1 <= wks_t2)

            # 2. Temporal Precedence: Valid ONLY if they share the same workstation
            for p in self.problem.periods:
                for w in self.problem.stations:
                    self.cp_model.end_before_start(
                        self.variables["opt_intervals"][(t1, p, w)],
                        self.variables["opt_intervals"][(t2, p, w)],
                    )

    def create_cycle_time_variables(self):
        cycle_time_used = {}
        cycle_time_chosen = {}
        max_horizon = self.problem.c_max

        for p in self.problem.periods:
            cycle_time_chosen[p] = self.cp_model.int_var(
                min=self.problem.c_target,
                max=max_horizon,
                name=f"cycle_time_chosen_{p}",
            )
            # Baseline offset applied for this specific period's unfolded timeline
            max_ends = []
            for task in self.problem.tasks:
                main_int = self.variables["main_intervals"][(task, p)]
                # Since all active tasks inside main_intervals perfectly overlap
                # within [p*max, (p+1)*max], we just subtract init_time to get the relative end.
                relative_end = self.cp_model.end(main_int)
                max_ends.append(relative_end)
            cycle_time_used[p] = self.cp_model.max(max_ends)
            self.cp_model.enforce(cycle_time_chosen[p] >= cycle_time_used[p])
            # For stable periods: if cycle_time_used <= c_target, then cycle_time_chosen == c_target
            if p >= self.problem.nb_stations:
                self.cp_model.enforce(
                    self.cp_model.implies(
                        cycle_time_used[p] <= self.problem.c_target,
                        cycle_time_chosen[p] == self.problem.c_target,
                    )
                )

        self.variables["cycle_time_used"] = cycle_time_used
        self.variables["cycle_time_chosen"] = cycle_time_chosen

        # Unstable periods logic: Cycle time remains constant [cite: 176, 178, 439]
        # Identify which periods in our window are unstable (< nb_stations)
        unstable_periods = [
            p for p in self.problem.periods if p < self.problem.nb_stations
        ]
        if len(unstable_periods) > 0:
            first_unstable = unstable_periods[0]
            self.cp_model.enforce(
                self.variables["cycle_time_chosen"][first_unstable]
                == self.cp_model.max(
                    [self.variables["cycle_time_used"][uns] for uns in unstable_periods]
                )
            )
            for i in range(1, len(unstable_periods)):
                p = unstable_periods[i]
                prev_p = unstable_periods[i - 1]
                self.cp_model.enforce(
                    self.variables["cycle_time_chosen"][p]
                    == self.variables["cycle_time_chosen"][prev_p]
                )
        # Stable periods logic: Cycle time is monotonically decreasing [cite: 488, 499]
        stable_periods = [
            p for p in self.problem.periods if p >= self.problem.nb_stations
        ]
        for i, p in enumerate(stable_periods):
            # Find the previous period (might be in unstable or stable)
            p_idx = self.problem.periods.index(p)
            if p_idx > 0:
                prev_p = self.problem.periods[p_idx - 1]
                self.cp_model.enforce(
                    self.variables["cycle_time_chosen"][p]
                    <= self.variables["cycle_time_chosen"][prev_p]
                )
                self.cp_model.enforce(
                    self.variables["cycle_time_used"][p]
                    <= self.variables["cycle_time_used"][prev_p]
                )
                self.cp_model.enforce(
                    self.cp_model.implies(
                        self.variables["cycle_time_chosen"][p]
                        < self.variables["cycle_time_chosen"][prev_p],
                        self.variables["cycle_time_chosen"][p]
                        == self.cp_model.max2(
                            self.variables["cycle_time_used"][p], self.problem.c_target
                        ),
                    )
                )

    def objective_value(self, minimize_used_cycle_time: bool = False):
        if minimize_used_cycle_time:
            # ALTERNATIVE OBJECTIVE: Squeeze the layout as tight as mathematically possible.
            obj_terms = []
            for p in self.problem.periods:
                obj_terms.append(self.variables["cycle_time_used"][p])
            self.cp_model.minimize(self.cp_model.sum(obj_terms))
        else:
            # ORIGINAL OBJECTIVE: Minimizing Ramp-up cost
            obj_terms = []
            self.variables["cost"] = {}
            for p in self.problem.periods:
                if p < self.problem.nb_stations:
                    obj_terms.append(self.variables["cycle_time_chosen"][p])
                else:
                    cost = self.cp_model.int_var(
                        min=0, max=self.problem.c_max, name=f"cost_{p}"
                    )
                    self.cp_model.enforce(
                        self.cp_model.implies(
                            (
                                self.variables["cycle_time_chosen"][p]
                                > self.problem.c_target
                            ),
                            cost == self.variables["cycle_time_chosen"][p],
                        )
                    )
                    self.cp_model.enforce(
                        self.cp_model.implies(
                            (
                                self.variables["cycle_time_chosen"][p]
                                == self.problem.c_target
                            ),
                            cost == 0,
                        )
                    )
                    obj_terms.append(cost)
                    self.variables["cost"][p] = cost
            self.cp_model.minimize(self.cp_model.sum(obj_terms))

    def create_heuristic_target_reached_constraints(self, apply_heuristic: bool = True):
        max_horizon = self.problem.c_max
        c_target = self.problem.c_target

        # 1. Constraint: If the "used" cycle time <= c_target, clamp the "chosen" cycle time to c_target.
        for p in self.problem.periods:
            # Link the boolean variable to the condition
            self.cp_model.enforce(
                (self.variables["cycle_time_used"][p] <= c_target)
                == (self.variables["cycle_time_chosen"][p] == c_target)
            )

        # 2. Heuristic: Freeze future schedules once target is reached
        if apply_heuristic:
            # We only apply this starting from the first stable period
            stable_periods = [
                p for p in self.problem.periods if p >= self.problem.nb_stations
            ]
            for i, p in enumerate(stable_periods[:-1]):  # Exclude the last period
                next_p = stable_periods[i + 1]
                for t in self.problem.tasks:
                    # Isolate the relative start time of task t for period p and next_p
                    rel_start_p = self.cp_model.start(
                        self.variables["main_intervals"][(t, p)]
                    )
                    rel_start_next = self.cp_model.start(
                        self.variables["main_intervals"][(t, next_p)]
                    )
                    # Implication: If target is reached at period p, enforce that the relative
                    # start time in next_p is strictly equal to the start time in period p.
                    self.cp_model.enforce(
                        self.cp_model.implies(
                            self.variables["cycle_time_chosen"][p] == c_target,
                            (rel_start_p == rel_start_next),
                        )
                    )

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: WorkStation
    ) -> "cp.BoolExpr":
        return self.variables["allocations"][(task[0], task[1], unary_resource)]

    def get_task_interval_variable(self, task: Task) -> "cp.IntervalVar":
        return self.variables["main_intervals"][task]

    def retrieve_solution(self, result: "cp.SolveResult") -> RCALBPLSolution:
        """
        Parses the OptalCP result back into a native RCALBPLSolution, translating
        absolute unfolded times back to relative workstation times.
        """
        wks = {}
        raw = {}
        start = {}
        cyc = {}

        # Retrieve allocations
        first_period = self.problem.periods[0]
        for t in self.problem.tasks:
            for w in self.problem.stations:
                # Task allocation is constant, so we can just check first period
                if result.solution.is_present(
                    self.variables["opt_intervals"][(t, first_period, w)]
                ):
                    wks[t] = w
                    break

        # Retrieve resource dispatch bounds
        for r in self.problem.resources:
            for w in self.problem.stations:
                raw[(r, w)] = result.solution.get_value(
                    self.variables["resource_dispatch"][(r, w)]
                )

        # Retrieve cycle times chosen
        for p in self.problem.periods:
            cyc[p] = result.solution.get_value(self.variables["cycle_time_chosen"][p])

        # Retrieve relative task start times
        for p in self.problem.periods:
            for t in self.problem.tasks:
                w = wks[t]
                # Fetch absolute start time and subtract baseline offset
                start[(t, p)] = result.solution.get_start(
                    self.variables["opt_intervals"][(t, p, w)]
                )
        return RCALBPLSolution(
            problem=self.problem, wks=wks, raw=raw, start=start, cyc=cyc
        )

    def set_warm_start(self, solution: RCALBPLSolution) -> None:
        """
        Injects a solution as a warm-start hint for the OptalCP solver.
        Creates a cp.Solution object with variable assignments from the given RCALBPLSolution.

        Note: We only set integer variable hints (resource dispatch, cycle times).
        We don't set allocation hints because:
        1. If allocations are fixed via fix_allocations_and_resources(), hints would conflict
        2. If allocations are free, the solver is better at finding them than us providing hints
        """
        if self.cp_model is None:
            return
        actual_cycle_times = self.problem.compute_actual_cycle_time_per_period(solution)
        # Create a solution builder
        sol = cp.Solution()

        # 1. Set Resource Dispatch
        for r in self.problem.resources:
            for w in self.problem.stations:
                var = self.variables["resource_dispatch"][(r, w)]
                val = solution.raw.get((r, w), 0)
                sol.set_value(var, val)

        # 2. Set Cycle Times and cost
        for p in self.problem.periods:
            cyc_val = solution.cyc.get(p, self.problem.c_max)
            sol.set_value(self.variables["cycle_time_chosen"][p], cyc_val)
            if p in self.variables["cost"]:
                if cyc_val <= self.problem.c_target:
                    sol.set_value(self.variables["cost"][p], 0)
                else:
                    sol.set_value(self.variables["cost"][p], cyc_val)
        # 3. Schedules
        for p in self.problem.periods:
            for t in self.problem.tasks:
                st = solution.get_start_time((t, p))
                end = solution.get_end_time((t, p))
                sol.set_value(self.variables["main_intervals"][(t, p)], st, end)
                for w in self.problem.stations:
                    if solution.is_allocated((t, p), w):
                        sol.set_value(
                            self.variables["opt_intervals"][(t, p, w)], st, end
                        )
                    else:
                        sol.set_absent(self.variables["opt_intervals"][(t, p, w)])
        # Dummy interval
        cost = self.problem.evaluate(solution)["ramp_up_duration"]
        sol.set_objective(cost)
        p_lb = 0
        p_ub = self.problem.c_max
        sol.set_value(self.variables["dummy_interval"], p_lb, p_ub)
        # Store the warm start solution
        self.use_warm_start = True
        self.warm_start_solution = sol

    def fix_allocations_and_resources(self, wks: dict, raw: dict):
        """
        Hard-fixes the allocation and resource variables to known values.
        This transforms the ALBP problem into a pure scheduling problem.
        """
        if self.cp_model is None:
            return

        # Lock Task Allocations
        for t, w_assigned in wks.items():
            for p in self.problem.periods:
                for w in self.problem.stations:
                    val = 1 if w == w_assigned else 0
                    self.cp_model.enforce(
                        self.variables["allocations"][(t, p, w)] == val
                    )

        # Lock Resource Dispatch
        if "resource_dispatch" in self.variables:
            for (r, w), val in raw.items():
                self.cp_model.enforce(
                    self.variables["resource_dispatch"][(r, w)] == val
                )

    def add_cycle_time_lower_bound(self, p: int, lower_bound: int):
        """
        Ensures cycle time monotonicity across independent chunk boundaries.
        Adds a constraint that the cycle time for period p must be >= lower_bound.
        """
        if (
            "cycle_time_chosen" in self.variables
            and p in self.variables["cycle_time_chosen"]
        ):
            self.cp_model.enforce(self.variables["cycle_time_chosen"][p] >= lower_bound)
