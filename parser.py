"""
utils/parser.py
---------------
Módulo para leer y parsear archivos de datos en formato TS (TS5.txt, etc.)
del sistema Cross Docking de LogiFast CR.

Formato del archivo:
  i  <num_inbound>   o  <num_outbound>   n  <num_products>
  r  <truck_id>  <product_id>  <quantity>   ...
  s  <truck_id>  <product_id>  <quantity>   ...
"""

import re
import pandas as pd
from io import StringIO


def parse_ts_content(content: str) -> dict:
    """
    Parsea el contenido de un archivo TS en formato compacto o con saltos de línea.

    Parámetros
    ----------
    content : str
        Texto crudo del archivo TS.

    Retorna
    -------
    dict con claves:
        num_inbound   : int  – número de camiones de entrada
        num_outbound  : int  – número de camiones de salida
        num_products  : int  – número de tipos de producto
        inbound       : dict  {truck_id -> {product_id -> quantity}}
        outbound      : dict  {truck_id -> {product_id -> quantity}}
        df_inbound    : pd.DataFrame
        df_outbound   : pd.DataFrame
    """
    # Normalizar: separar tokens por espacios y saltos de línea
    raw_tokens = re.split(r'[\s\t\n\r]+', content.strip())

    # El archivo puede estar compactado: "170r" significa valor 170 seguido del token 'r'.
    # Necesitamos dividir tokens que mezclan dígitos y letras.
    tokens = []
    for tok in raw_tokens:
        if not tok:
            continue
        # Dividir en partes de dígitos y partes de letras
        parts = re.findall(r'[a-zA-Z]+|\d+', tok)
        tokens.extend(parts)

    idx = 0

    def next_token():
        nonlocal idx
        val = tokens[idx]
        idx += 1
        return val

    # ---------------------------------------------------------------
    # Cabecera: i <n_i>  o <n_o>  n <n_prod>
    # ---------------------------------------------------------------
    num_inbound = None
    num_outbound = None
    num_products = None

    while idx < len(tokens):
        tok = next_token()
        if tok == 'i':
            num_inbound = int(next_token())
        elif tok == 'o':
            num_outbound = int(next_token())
        elif tok == 'n':
            num_products = int(next_token())
        elif tok in ('r', 's'):
            # Ya empezaron los datos → retroceder
            idx -= 1
            break

    # ---------------------------------------------------------------
    # Filas de datos
    # ---------------------------------------------------------------
    inbound = {i: {} for i in range(1, num_inbound + 1)}
    outbound = {j: {} for j in range(1, num_outbound + 1)}

    rows_in = []
    rows_out = []

    while idx < len(tokens):
        tok = next_token()
        if tok == 'r':
            truck = int(next_token())
            product = int(next_token())
            qty = int(next_token())
            inbound[truck][product] = inbound[truck].get(product, 0) + qty
            rows_in.append({'truck': truck, 'product': product, 'quantity': qty})
        elif tok == 's':
            truck = int(next_token())
            product = int(next_token())
            qty = int(next_token())
            outbound[truck][product] = outbound[truck].get(product, 0) + qty
            rows_out.append({'truck': truck, 'product': product, 'quantity': qty})

    df_inbound = pd.DataFrame(rows_in) if rows_in else pd.DataFrame(columns=['truck', 'product', 'quantity'])
    df_outbound = pd.DataFrame(rows_out) if rows_out else pd.DataFrame(columns=['truck', 'product', 'quantity'])

    return {
        'num_inbound': num_inbound,
        'num_outbound': num_outbound,
        'num_products': num_products,
        'inbound': inbound,
        'outbound': outbound,
        'df_inbound': df_inbound,
        'df_outbound': df_outbound,
    }


def parse_ts_file(filepath: str) -> dict:
    """Lee un archivo TS desde disco y lo parsea."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return parse_ts_content(content)


def build_supply_demand_matrix(data: dict) -> pd.DataFrame:
    """
    Construye una matriz (camiones_entrada x productos) con cantidades ofrecidas,
    y otra (camiones_salida x productos) con cantidades demandadas.

    Retorna un DataFrame pivotado para visualización.
    """
    n_in = data['num_inbound']
    n_out = data['num_outbound']
    n_prod = data['num_products']

    # Matriz oferta
    supply = pd.DataFrame(0, index=range(1, n_in + 1), columns=range(1, n_prod + 1))
    supply.index.name = 'Camión Entrada'
    supply.columns.name = 'Producto'
    for i, prods in data['inbound'].items():
        for k, q in prods.items():
            supply.loc[i, k] = q

    # Matriz demanda
    demand = pd.DataFrame(0, index=range(1, n_out + 1), columns=range(1, n_prod + 1))
    demand.index.name = 'Camión Salida'
    demand.columns.name = 'Producto'
    for j, prods in data['outbound'].items():
        for k, q in prods.items():
            demand.loc[j, k] = q

    return supply, demand


def validate_data(data: dict) -> list:
    """
    Valida que la oferta total de cada producto iguale la demanda total.
    Retorna lista de mensajes de advertencia (vacía si todo está OK).
    """
    warnings = []
    n_prod = data['num_products']

    for k in range(1, n_prod + 1):
        supply_k = sum(data['inbound'][i].get(k, 0) for i in data['inbound'])
        demand_k = sum(data['outbound'][j].get(k, 0) for j in data['outbound'])
        if supply_k != demand_k:
            warnings.append(
                f"⚠️ Producto {k}: oferta ({supply_k}) ≠ demanda ({demand_k})"
            )

    return warnings
