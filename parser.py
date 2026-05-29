"""
utils/parser.py
---------------
Parser para archivos con formato TS (Cross Docking LogiFast CR).

Formato del archivo:
  i  <num_camiones_entrada>
  o  <num_camiones_salida>
  n  <num_productos>
  r  <id_camion_entrada>  <id_producto>  <cantidad>
  s  <id_camion_salida>   <id_producto>  <cantidad>

Todos los tokens pueden estar en una sola línea o en múltiples líneas.
"""

import re
from typing import Dict, Tuple


def parse_ts_file(content: str) -> Dict:
    """
    Parsea el contenido de un archivo TS y devuelve un diccionario
    con los parámetros de la instancia.

    Args:
        content: Contenido del archivo como string.

    Returns:
        {
            'num_inbound':  int,           # número de camiones de entrada
            'num_outbound': int,           # número de camiones de salida
            'num_products': int,           # número de tipos de producto
            'inbound':  {(i,k): qty},      # unidades de producto k en camión i
            'outbound': {(j,k): qty},      # unidades de producto k requeridas por camión j
        }
    """
    # Normalizar: reemplazar separadores múltiples y split general
    tokens = re.split(r'[\s\t\n\r]+', content.strip())

    num_inbound = 0
    num_outbound = 0
    num_products = 0
    inbound: Dict[Tuple[int, int], int] = {}
    outbound: Dict[Tuple[int, int], int] = {}

    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]

        if tok == 'i':
            num_inbound = int(tokens[idx + 1])
            idx += 2
        elif tok == 'o':
            num_outbound = int(tokens[idx + 1])
            idx += 2
        elif tok == 'n':
            num_products = int(tokens[idx + 1])
            idx += 2
        elif tok == 'r':
            truck_id = int(tokens[idx + 1])
            product_id = int(tokens[idx + 2])
            quantity = int(tokens[idx + 3])
            inbound[(truck_id, product_id)] = quantity
            idx += 4
        elif tok == 's':
            truck_id = int(tokens[idx + 1])
            product_id = int(tokens[idx + 2])
            quantity = int(tokens[idx + 3])
            outbound[(truck_id, product_id)] = quantity
            idx += 4
        else:
            idx += 1  # token inesperado, saltar

    return {
        'num_inbound': num_inbound,
        'num_outbound': num_outbound,
        'num_products': num_products,
        'inbound': inbound,
        'outbound': outbound,
    }


def build_matrices(data: Dict):
    """
    Construye matrices densas a partir del diccionario parseado.

    Returns:
        R[i][k]: unidades del producto k en camión de entrada i  (índices 1-based)
        S[j][k]: unidades del producto k requeridas por camión j (índices 1-based)
    """
    I = data['num_inbound']
    J = data['num_outbound']
    K = data['num_products']

    # Matrices inicializadas en 0
    R = {i: {k: 0 for k in range(1, K + 1)} for i in range(1, I + 1)}
    S = {j: {k: 0 for k in range(1, K + 1)} for j in range(1, J + 1)}

    for (i, k), qty in data['inbound'].items():
        R[i][k] = qty

    for (j, k), qty in data['outbound'].items():
        S[j][k] = qty

    return R, S


def validate_data(data: Dict) -> Tuple[bool, str]:
    """
    Verifica que la demanda y la oferta sean balanceadas por producto.

    Returns:
        (is_valid, message)
    """
    R, S = build_matrices(data)
    K = data['num_products']
    I = data['num_inbound']
    J = data['num_outbound']

    issues = []
    for k in range(1, K + 1):
        supply = sum(R[i][k] for i in range(1, I + 1))
        demand = sum(S[j][k] for j in range(1, J + 1))
        if supply != demand:
            issues.append(
                f"Producto {k}: oferta={supply} ≠ demanda={demand}"
            )

    if issues:
        return False, "Desbalance detectado:\n" + "\n".join(issues)
    return True, "Datos balanceados correctamente."


def summarize(data: Dict) -> str:
    """Resumen legible de la instancia."""
    lines = [
        f"Camiones de entrada : {data['num_inbound']}",
        f"Camiones de salida  : {data['num_outbound']}",
        f"Tipos de producto   : {data['num_products']}",
        f"Registros entrada   : {len(data['inbound'])}",
        f"Registros salida    : {len(data['outbound'])}",
    ]
    return "\n".join(lines)
