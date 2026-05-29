"""
solver.py
---------
Modelo de Programación Entera Mixta (MIP) para el problema de
Cross Docking de LogiFast CR.

Minimiza el makespan (tiempo total de operación) determinando:
  - El orden óptimo de atención de camiones de entrada y salida.
  - Las transferencias directas vs. paso por almacenamiento temporal.
  - Los tiempos de llegada/salida de cada camión al muelle.

Formulación basada en:
  Boysen, N. et al. (2010) "Scheduling inbound and outbound trucks at
  cross docking terminals" — adaptada a los parámetros del caso UCR.

Variables:
  x[i,j,k]  : int  — unidades del producto k del camión entrada i al camión salida j
  v[i,j]    : bin  — 1 si hay transferencia entre camión i y j
  u_in[i,p] : bin  — 1 si camión entrada i precede al camión entrada p
  u_out[j,q]: bin  — 1 si camión salida j precede al camión salida q
  a[i]       : cont — tiempo (min) en que el camión de entrada i llega al muelle
  d[j]       : cont — tiempo (min) en que el camión de salida j abandona el muelle
  C_max      : cont — makespan (objetivo a minimizar)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pulp

# ---------------------------------------------------------------------------
# Parámetros operativos (según el enunciado del caso)
# ---------------------------------------------------------------------------
T_UNIT   = 1    # minutos por unidad cargada/descargada
T_TRANSF = 5    # minutos por lote transferido internamente
T_CHANGE = 10   # minutos de cambio entre camiones en el muelle
BIG_M    = 99_999  # constante big-M


# ---------------------------------------------------------------------------
# Dataclasses de resultado
# ---------------------------------------------------------------------------

@dataclass
class TruckSchedule:
    truck_id: int
    truck_type: str          # 'inbound' | 'outbound'
    start_time: float
    end_time: float
    products: Dict[int, int] = field(default_factory=dict)   # {product_k: units}
    order_position: int = 0


@dataclass
class Transfer:
    inbound_truck: int
    outbound_truck: int
    product: int
    units: int
    via_storage: bool        # True → pasa por almacenamiento temporal


@dataclass
class SolverResult:
    status: str              # 'Optimal' | 'Feasible' | 'Infeasible' | 'Error'
    makespan: float
    inbound_sequence: List[int]
    outbound_sequence: List[int]
    inbound_schedule: List[TruckSchedule]
    outbound_schedule: List[TruckSchedule]
    transfers: List[Transfer]
    direct_units: int
    storage_units: int
    solve_time: float
    gap: float = 0.0
    model_info: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def solve_crossdocking(
    num_inbound: int,
    num_outbound: int,
    num_products: int,
    R: Dict[int, Dict[int, int]],   # R[i][k]
    S: Dict[int, Dict[int, int]],   # S[j][k]
    time_limit: int = 300,
    mip_gap: float = 0.01,
) -> SolverResult:
    """
    Resuelve el MIP de Cross Docking con PuLP (CBC).

    Parameters
    ----------
    num_inbound  : número de camiones de entrada (I)
    num_outbound : número de camiones de salida  (J)
    num_products : número de tipos de producto   (K)
    R            : unidades de producto k en camión entrada i
    S            : unidades de producto k requeridas por camión salida j
    time_limit   : límite de tiempo del solver en segundos
    mip_gap      : tolerancia de optimalidad relativa

    Returns
    -------
    SolverResult con todos los resultados.
    """
    start_clock = time.time()

    I = list(range(1, num_inbound + 1))
    J = list(range(1, num_outbound + 1))
    K = list(range(1, num_products + 1))

    # ------------------------------------------------------------------
    # Tiempo de descarga de cada camión de entrada
    # ------------------------------------------------------------------
    unload_time: Dict[int, int] = {
        i: T_UNIT * sum(R[i][k] for k in K) for i in I
    }

    # Tiempo de carga de cada camión de salida
    load_time: Dict[int, int] = {
        j: T_UNIT * sum(S[j][k] for k in K) for j in J
    }

    # Cota superior del makespan (heurística: todo en secuencia)
    upper_bound = (
        sum(unload_time.values())
        + sum(load_time.values())
        + len(I) * T_CHANGE
        + len(J) * T_CHANGE
        + len(I) * len(J) * T_TRANSF
        + 1000
    )

    # ------------------------------------------------------------------
    # Modelo PuLP
    # ------------------------------------------------------------------
    model = pulp.LpProblem("CrossDocking_LogiFast", pulp.LpMinimize)

    # ---- Variables de flujo ----
    x = {
        (i, j, k): pulp.LpVariable(
            f"x_{i}_{j}_{k}", lowBound=0, cat='Integer'
        )
        for i in I for j in J for k in K
    }

    # ---- Variables binarias de transferencia ----
    v = {
        (i, j): pulp.LpVariable(f"v_{i}_{j}", cat='Binary')
        for i in I for j in J
    }

    # ---- Variables de secuencia camiones de entrada ----
    u_in = {
        (i, p): pulp.LpVariable(f"uin_{i}_{p}", cat='Binary')
        for i in I for p in I if i != p
    }

    # ---- Variables de secuencia camiones de salida ----
    u_out = {
        (j, q): pulp.LpVariable(f"uout_{j}_{q}", cat='Binary')
        for j in J for q in J if j != q
    }

    # ---- Tiempos de muelle ----
    a = {i: pulp.LpVariable(f"a_{i}", lowBound=0) for i in I}
    d = {j: pulp.LpVariable(f"d_{j}", lowBound=0) for j in J}

    # ---- Makespan ----
    C_max = pulp.LpVariable("C_max", lowBound=0)

    # ---- Objetivo ----
    model += C_max, "Minimizar_Makespan"

    # ==================================================================
    # RESTRICCIONES
    # ==================================================================

    # R1: Makespan ≥ tiempo de salida de cada camión de salida
    for j in J:
        model += C_max >= d[j] + load_time[j], f"makespan_j{j}"

    # R2: Conservación de producto en camiones de entrada
    for i in I:
        for k in K:
            model += (
                pulp.lpSum(x[i, j, k] for j in J) == R[i][k],
                f"conserv_entrada_i{i}_k{k}"
            )

    # R3: Conservación de producto en camiones de salida
    for j in J:
        for k in K:
            model += (
                pulp.lpSum(x[i, j, k] for i in I) == S[j][k],
                f"conserv_salida_j{j}_k{k}"
            )

    # R4: Relación x → v  (si se transfiere producto, v debe ser 1)
    M_x = max(
        max(R[i].values()) if R[i] else 0
        for i in I
    ) + 1
    for i in I:
        for j in J:
            model += (
                pulp.lpSum(x[i, j, k] for k in K) <= M_x * v[i, j],
                f"relacion_xv_i{i}_j{j}"
            )

    # R5-R7: Secuencia válida de camiones de entrada
    # Si u_in[i,p]=1 entonces a[p] ≥ a[i] + unload_time[i] + T_CHANGE
    for i in I:
        for p in I:
            if i != p:
                model += (
                    a[p] >= a[i] + unload_time[i] + T_CHANGE
                    - BIG_M * (1 - u_in[i, p]),
                    f"seq_in_a_i{i}_p{p}"
                )

    # R6: Exactamente un orden entre cada par de camiones de entrada
    for i in I:
        for p in I:
            if i < p:
                model += (
                    u_in[i, p] + u_in[p, i] == 1,
                    f"orden_in_i{i}_p{p}"
                )

    # R7: Anti-reflexividad (no auto-precedencia) — garantizada por construcción
    # (no se crea u_in[i,i])

    # R8: No auto-precedencia de entrada (restricción explícita redundante, omitida)

    # R9-R11: Secuencia válida de camiones de salida
    for j in J:
        for q in J:
            if j != q:
                model += (
                    d[q] >= d[j] + T_CHANGE
                    - BIG_M * (1 - u_out[j, q]),
                    f"seq_out_d_j{j}_q{q}"
                )

    # R10: Exactamente un orden entre cada par de camiones de salida
    for j in J:
        for q in J:
            if j < q:
                model += (
                    u_out[j, q] + u_out[q, j] == 1,
                    f"orden_out_j{j}_q{q}"
                )

    # R12: No auto-precedencia de salida — garantizada por construcción

    # R13: Conexión entre llegada del camión de entrada y salida del de salida
    # d[j] ≥ a[i] + unload_time[i] + T_TRANSF  si v[i,j]=1
    for i in I:
        for j in J:
            model += (
                d[j] >= a[i] + unload_time[i] + T_TRANSF
                - BIG_M * (1 - v[i, j]),
                f"conexion_i{i}_j{j}"
            )

    # Cota inferior de tiempos de salida: al menos el tiempo de carga
    for j in J:
        model += d[j] >= load_time[j], f"min_salida_j{j}"

    # ------------------------------------------------------------------
    # Resolver
    # ------------------------------------------------------------------
    solver = pulp.PULP_CBC_CMD(
        timeLimit=time_limit,
        gapRel=mip_gap,
        msg=0,
    )

    solve_status = model.solve(solver)
    elapsed = time.time() - start_clock

    status_map = {
        1: "Optimal",
        2: "Infeasible",
        3: "Unbounded",
        -1: "Infeasible",
        0: "Not Solved",
    }
    status_str = status_map.get(solve_status, "Feasible")
    if pulp.value(model.objective) is not None and status_str == "Not Solved":
        status_str = "Feasible"

    if pulp.value(model.objective) is None:
        return SolverResult(
            status="Infeasible",
            makespan=0,
            inbound_sequence=[],
            outbound_sequence=[],
            inbound_schedule=[],
            outbound_schedule=[],
            transfers=[],
            direct_units=0,
            storage_units=0,
            solve_time=elapsed,
        )

    # ------------------------------------------------------------------
    # Extraer resultados
    # ------------------------------------------------------------------
    makespan_val = pulp.value(C_max)

    # Secuencias
    a_vals = {i: pulp.value(a[i]) or 0.0 for i in I}
    d_vals = {j: pulp.value(d[j]) or 0.0 for j in J}

    inbound_seq = sorted(I, key=lambda i: a_vals[i])
    outbound_seq = sorted(J, key=lambda j: d_vals[j])

    # Schedules de entrada
    inbound_sched = []
    for pos, i in enumerate(inbound_seq, 1):
        prods = {k: R[i][k] for k in K if R[i][k] > 0}
        inbound_sched.append(TruckSchedule(
            truck_id=i,
            truck_type='inbound',
            start_time=round(a_vals[i], 2),
            end_time=round(a_vals[i] + unload_time[i], 2),
            products=prods,
            order_position=pos,
        ))

    # Schedules de salida
    outbound_sched = []
    for pos, j in enumerate(outbound_seq, 1):
        prods = {k: S[j][k] for k in K if S[j][k] > 0}
        outbound_sched.append(TruckSchedule(
            truck_id=j,
            truck_type='outbound',
            start_time=round(d_vals[j], 2),
            end_time=round(d_vals[j] + load_time[j], 2),
            products=prods,
            order_position=pos,
        ))

    # Transferencias
    transfers = []
    direct_units = 0
    storage_units = 0

    for i in I:
        ai_end = a_vals[i] + unload_time[i]
        for j in J:
            dj = d_vals[j]
            for k in K:
                units = int(round(pulp.value(x[i, j, k]) or 0))
                if units > 0:
                    # Determinar si va directo o por almacén:
                    # Directo si el camión de entrada termina de descargar
                    # antes de que el camión de salida empiece a cargar.
                    via_storage = ai_end > dj + 1e-3
                    transfers.append(Transfer(
                        inbound_truck=i,
                        outbound_truck=j,
                        product=k,
                        units=units,
                        via_storage=via_storage,
                    ))
                    if via_storage:
                        storage_units += units
                    else:
                        direct_units += units

    gap_val = abs(model.solverModel.bestBound - makespan_val) / (abs(makespan_val) + 1e-9) \
        if hasattr(model, 'solverModel') else 0.0

    model_info = {
        'num_variables': len(model.variables()),
        'num_constraints': len(model.constraints),
        'objective_value': makespan_val,
    }

    return SolverResult(
        status=status_str,
        makespan=round(makespan_val, 2),
        inbound_sequence=inbound_seq,
        outbound_sequence=outbound_seq,
        inbound_schedule=inbound_sched,
        outbound_schedule=outbound_sched,
        transfers=transfers,
        direct_units=direct_units,
        storage_units=storage_units,
        solve_time=round(elapsed, 2),
        gap=round(gap_val * 100, 2),
        model_info=model_info,
    )


# ---------------------------------------------------------------------------
# Heurística constructiva (fallback / comparación)
# ---------------------------------------------------------------------------

def solve_heuristic(
    num_inbound: int,
    num_outbound: int,
    num_products: int,
    R: Dict[int, Dict[int, int]],
    S: Dict[int, Dict[int, int]],
) -> SolverResult:
    """
    Heurística SPT (Shortest Processing Time) como cota de referencia.
    Ordena los camiones de entrada por tiempo de descarga ascendente
    y los de salida por tiempo de carga ascendente.
    """
    start_clock = time.time()

    I = list(range(1, num_inbound + 1))
    J = list(range(1, num_outbound + 1))
    K = list(range(1, num_products + 1))

    unload_time = {i: T_UNIT * sum(R[i][k] for k in K) for i in I}
    load_time   = {j: T_UNIT * sum(S[j][k] for k in K) for j in J}

    # Ordenar SPT
    inbound_seq  = sorted(I, key=lambda i: unload_time[i])
    outbound_seq = sorted(J, key=lambda j: load_time[j])

    # Calcular tiempos simples
    t = 0.0
    a_vals: Dict[int, float] = {}
    for i in inbound_seq:
        a_vals[i] = t
        t += unload_time[i] + T_CHANGE

    t = 0.0
    d_vals: Dict[int, float] = {}
    for j in outbound_seq:
        d_vals[j] = t
        t += load_time[j] + T_CHANGE

    # Asignar flujos greedy (proporcional)
    x_vals: Dict[Tuple[int,int,int], int] = {}
    transfers = []
    direct_units = 0
    storage_units = 0

    for k in K:
        supply = {i: R[i][k] for i in I}
        demand = {j: S[j][k] for j in J}
        for i in inbound_seq:
            for j in outbound_seq:
                amt = min(supply[i], demand[j])
                if amt > 0:
                    x_vals[(i, j, k)] = amt
                    supply[i] -= amt
                    demand[j] -= amt
                    via = a_vals[i] + unload_time[i] > d_vals[j] + 1e-3
                    transfers.append(Transfer(i, j, k, amt, via))
                    if via:
                        storage_units += amt
                    else:
                        direct_units += amt

    makespan = max(
        d_vals[j] + load_time[j] for j in J
    )

    inbound_sched = [
        TruckSchedule(i, 'inbound', a_vals[i], a_vals[i]+unload_time[i],
                      {k: R[i][k] for k in K if R[i][k]>0}, pos+1)
        for pos, i in enumerate(inbound_seq)
    ]
    outbound_sched = [
        TruckSchedule(j, 'outbound', d_vals[j], d_vals[j]+load_time[j],
                      {k: S[j][k] for k in K if S[j][k]>0}, pos+1)
        for pos, j in enumerate(outbound_seq)
    ]

    return SolverResult(
        status="Heuristic",
        makespan=round(makespan, 2),
        inbound_sequence=inbound_seq,
        outbound_sequence=outbound_seq,
        inbound_schedule=inbound_sched,
        outbound_schedule=outbound_sched,
        transfers=transfers,
        direct_units=direct_units,
        storage_units=storage_units,
        solve_time=round(time.time() - start_clock, 4),
        model_info={'method': 'SPT Heuristic'},
    )
