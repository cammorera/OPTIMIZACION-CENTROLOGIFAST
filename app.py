"""
app.py
------
Aplicación Streamlit para el caso de optimización Cross Docking — LogiFast CR
Universidad de Costa Rica · Ingeniería Industrial · MIP I-2026
"""

import io
import os
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from solver import solve_crossdocking, solve_heuristic
from utils.parser import parse_ts_file, build_matrices, validate_data, summarize

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LogiFast CR — Cross Docking Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS personalizado
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  /* Fuente principal */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
      font-family: 'IBM Plex Sans', sans-serif;
      color: #1a1a2e;
  }

  /* Fondo principal */
  .stApp {
      background-color: #f8f9fc;
  }

  /* Barra lateral */
  [data-testid="stSidebar"] {
      background-color: #1a1a2e;
      color: #e8eaf6;
  }
  [data-testid="stSidebar"] * {
      color: #e8eaf6 !important;
  }
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {
      color: #7eb8f7 !important;
  }

  /* Tarjetas de métricas */
  .metric-card {
      background: white;
      border-radius: 12px;
      padding: 20px 24px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.07);
      border-left: 4px solid #2563eb;
      margin-bottom: 12px;
  }
  .metric-card.green  { border-left-color: #16a34a; }
  .metric-card.orange { border-left-color: #ea580c; }
  .metric-card.purple { border-left-color: #7c3aed; }
  .metric-card.red    { border-left-color: #dc2626; }

  .metric-label {
      font-size: 0.78rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #6b7280;
      margin-bottom: 4px;
  }
  .metric-value {
      font-size: 2rem;
      font-weight: 700;
      color: #1a1a2e;
      line-height: 1.1;
  }
  .metric-sub {
      font-size: 0.8rem;
      color: #9ca3af;
      margin-top: 2px;
  }

  /* Header banner */
  .hero-banner {
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
      color: white;
      padding: 36px 40px;
      border-radius: 16px;
      margin-bottom: 28px;
  }
  .hero-banner h1 {
      font-size: 2.1rem;
      font-weight: 700;
      margin: 0 0 8px 0;
      color: white !important;
  }
  .hero-banner p {
      font-size: 1.05rem;
      opacity: 0.8;
      margin: 0;
      color: #c7d2fe !important;
  }

  /* Sección de tabs */
  .stTabs [data-baseweb="tab"] {
      font-weight: 500;
      font-size: 0.9rem;
  }

  /* Tablas */
  .dataframe thead th {
      background-color: #1a1a2e !important;
      color: white !important;
      font-weight: 600 !important;
  }

  /* Botón principal */
  .stButton > button[kind="primary"] {
      background-color: #2563eb;
      border: none;
      border-radius: 8px;
      font-weight: 600;
      padding: 0.6rem 2rem;
      font-size: 1rem;
  }
  .stButton > button[kind="primary"]:hover {
      background-color: #1d4ed8;
  }

  /* Status boxes */
  .status-optimal {
      background: #dcfce7;
      border: 1px solid #16a34a;
      color: #15803d;
      padding: 10px 18px;
      border-radius: 8px;
      font-weight: 600;
      display: inline-block;
  }
  .status-heuristic {
      background: #fef9c3;
      border: 1px solid #ca8a04;
      color: #854d0e;
      padding: 10px 18px;
      border-radius: 8px;
      font-weight: 600;
      display: inline-block;
  }

  /* Código matemático */
  .math-block {
      background: #1e293b;
      color: #e2e8f0;
      padding: 18px 22px;
      border-radius: 10px;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.85rem;
      line-height: 1.6;
      overflow-x: auto;
  }

  /* Dividers */
  hr {
      border: none;
      border-top: 1px solid #e5e7eb;
      margin: 20px 0;
  }

  /* Section headers */
  h2 { color: #1a1a2e; font-weight: 700; }
  h3 { color: #374151; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers de UI
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, sub: str = "", color: str = "") -> str:
    cls = f"metric-card {color}".strip()
    return f"""
    <div class="{cls}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """


def minutes_to_hhmm(minutes: float) -> str:
    h = int(minutes) // 60
    m = int(minutes) % 60
    return f"{h}h {m:02d}min" if h > 0 else f"{m} min"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚚 LogiFast CR")
    st.markdown("**Cross Docking Optimizer**")
    st.markdown("---")

    st.markdown("### 📂 Datos de entrada")
    upload_mode = st.radio(
        "Fuente de datos",
        ["Archivo TS (demo TS5)", "Subir archivo .txt", "Editar manualmente"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Parámetros operativos")

    T_UNIT_UI   = st.number_input("Tiempo por unidad (min)", 1, 10, 1)
    T_TRANSF_UI = st.number_input("Tiempo traslado interno (min)", 1, 30, 5)
    T_CHANGE_UI = st.number_input("Tiempo cambio de camión (min)", 1, 60, 10)

    st.markdown("---")
    st.markdown("### 🔧 Solver")
    time_limit  = st.slider("Límite de tiempo (seg)", 30, 600, 180, 30)
    mip_gap_pct = st.slider("Gap MIP (%)", 0.1, 10.0, 1.0, 0.1)
    run_heuristic_flag = st.checkbox("Comparar con heurística SPT", value=True)

    st.markdown("---")
    st.markdown(
        "<small style='color:#6b7280'>UCR · Ing. Industrial · MIP I-2026</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
  <h1>🚚 Optimización Cross Docking — LogiFast CR</h1>
  <p>Programación Entera Mixta para minimizar el makespan del centro de distribución</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs principales
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "📋 Caso & Modelo",
    "📊 Datos",
    "🚀 Optimización",
    "📈 Resultados",
    "🗺️ Gantt",
    "📝 Conclusiones",
])


# ===========================================================================
# TAB 1 — CASO & MODELO MATEMÁTICO
# ===========================================================================
with tabs[0]:
    col_desc, col_math = st.columns([1, 1], gap="large")

    with col_desc:
        st.markdown("## 🏭 Descripción del problema")
        st.markdown("""
**LogiFast CR** opera un centro de distribución *cross docking* en el Valle Central
de Costa Rica para abastecer tiendas minoristas nacionales.

#### Operación
1. Los camiones de **entrada** llegan y descargan productos.
2. Los productos se trasladan al área de despacho (directamente o vía almacenamiento temporal).
3. Los camiones de **salida** esperan hasta completar su carga y entonces parten.

#### Problema central
> ¿En qué **orden** deben atenderse los camiones de entrada y salida para
> completar todas las operaciones en el **menor tiempo total posible**?

#### Restricciones operativas clave
- Un camión en el muelle no puede salir hasta terminar su operación.
- Solo un muelle de recepción y uno de despacho.
- No se puede cargar desde el proveedor y desde almacenamiento temporal simultáneamente.
- Todos los productos que entran deben salir el mismo día.
        """)

        st.markdown("#### Tipos de decisión")
        decision_df = pd.DataFrame({
            "Decisión": [
                "Orden de camiones de entrada",
                "Orden de camiones de salida",
                "Transferencia directa vs. almacén",
                "Unidades asignadas entre camiones",
            ],
            "Tipo": ["Entera binaria", "Entera binaria", "Entera binaria", "Entera positiva"],
            "Variable": ["u_in[i,p]", "u_out[j,q]", "v[i,j]", "x[i,j,k]"],
        })
        st.dataframe(decision_df, use_container_width=True, hide_index=True)

    with col_math:
        st.markdown("## 📐 Modelo matemático")

        st.markdown("#### Conjuntos y parámetros")
        st.markdown("""
| Símbolo | Descripción |
|---------|-------------|
| **I** | Conjunto de camiones de entrada {1..i} |
| **J** | Conjunto de camiones de salida {1..o} |
| **K** | Conjunto de tipos de producto {1..n} |
| r[i,k] | Unidades del producto k en camión entrada i |
| s[j,k] | Unidades del producto k requeridas por camión salida j |
| t_u | Tiempo de carga/descarga por unidad (1 min) |
| t_t | Tiempo de traslado interno por lote (5 min) |
| t_c | Tiempo de cambio entre camiones (10 min) |
        """)

        st.markdown("#### Variables de decisión")
        st.markdown("""
| Variable | Tipo | Descripción |
|----------|------|-------------|
| x[i,j,k] | Entera ≥ 0 | Unidades de producto k del camión i al camión j |
| v[i,j] | Binaria | 1 si hay transferencia entre camión i y j |
| u_in[i,p] | Binaria | 1 si camión entrada i precede a p |
| u_out[j,q] | Binaria | 1 si camión salida j precede a q |
| a[i] | Continua ≥ 0 | Tiempo de llegada al muelle del camión i |
| d[j] | Continua ≥ 0 | Tiempo de salida del muelle del camión j |
| C_max | Continua ≥ 0 | Makespan (tiempo total de operación) |
        """)

        st.markdown("#### Función objetivo")
        st.markdown("""
<div class="math-block">
min  C_max
</div>
        """, unsafe_allow_html=True)

        st.markdown("#### Restricciones (resumen)")
        st.markdown("""
<div class="math-block">
R1:  C_max ≥ d[j] + load_time[j]        ∀j ∈ J
R2:  Σ_j x[i,j,k] = r[i,k]             ∀i ∈ I, k ∈ K
R3:  Σ_i x[i,j,k] = s[j,k]             ∀j ∈ J, k ∈ K
R4:  Σ_k x[i,j,k] ≤ M · v[i,j]         ∀i,j
R5:  a[p] ≥ a[i] + t_unload[i] + t_c
         − M·(1−u_in[i,p])              ∀i≠p
R6:  u_in[i,p] + u_in[p,i] = 1         ∀i&lt;p
R7:  u_in[i,i] = 0  (no auto-prec.)
R8:  d[q] ≥ d[j] + t_c − M·(1−u_out[j,q])  ∀j≠q
R9:  u_out[j,q] + u_out[q,j] = 1        ∀j&lt;q
R10: u_out[j,j] = 0  (no auto-prec.)
R11: d[j] ≥ a[i] + t_unload[i] + t_t
          − M·(1−v[i,j])               ∀i,j
R12: d[j] ≥ load_time[j]               ∀j
</div>
        """, unsafe_allow_html=True)


# ===========================================================================
# TAB 2 — DATOS
# ===========================================================================
with tabs[1]:
    st.markdown("## 📊 Carga y visualización de datos")

    # Cargar datos según modo
    raw_content = None

    if upload_mode == "Archivo TS (demo TS5)":
        demo_path = os.path.join(os.path.dirname(__file__), "data", "TS5.txt")
        if os.path.exists(demo_path):
            with open(demo_path, "r") as f:
                raw_content = f.read()
            st.success("✅ Datos del archivo TS5 cargados correctamente.")
        else:
            st.error("No se encontró data/TS5.txt")

    elif upload_mode == "Subir archivo .txt":
        uploaded = st.file_uploader("Selecciona un archivo TS (.txt)", type=["txt"])
        if uploaded:
            raw_content = uploaded.read().decode("utf-8")
            st.success(f"✅ Archivo '{uploaded.name}' cargado.")

    else:  # Editar manualmente
        st.info("Edita los tokens del archivo TS directamente.")
        raw_content = st.text_area(
            "Contenido del archivo TS",
            value="i\t5\t\to\t3\t\tn\t8\t\t"
                  "r\t1\t1\t170\tr\t2\t1\t6\tr\t2\t2\t6\t"
                  "r\t2\t3\t19\tr\t2\t4\t50\tr\t2\t5\t38\t"
                  "r\t2\t6\t6\tr\t2\t7\t19\tr\t2\t8\t56\t"
                  "r\t3\t1\t49\tr\t3\t2\t31\tr\t3\t3\t60\t"
                  "r\t3\t6\t12\tr\t3\t7\t37\tr\t3\t8\t31\t"
                  "r\t4\t5\t143\tr\t4\t7\t47\t"
                  "r\t5\t4\t58\tr\t5\t5\t36\tr\t5\t7\t72\tr\t5\t8\t14\t"
                  "s\t1\t1\t75\ts\t1\t2\t12\ts\t1\t3\t59\t"
                  "s\t1\t6\t9\ts\t1\t7\t98\ts\t1\t8\t40\t"
                  "s\t2\t1\t150\ts\t2\t5\t217\t"
                  "s\t3\t2\t25\ts\t3\t3\t20\ts\t3\t4\t108\t"
                  "s\t3\t6\t9\ts\t3\t7\t77\ts\t3\t8\t61",
            height=200,
        )

    if raw_content:
        try:
            data = parse_ts_file(raw_content)
            st.session_state['data'] = data
            R, S = build_matrices(data)
            st.session_state['R'] = R
            st.session_state['S'] = S

            is_valid, msg = validate_data(data)

            col_sum, col_val = st.columns([1, 1])
            with col_sum:
                st.markdown("### 📌 Resumen de la instancia")
                for line in summarize(data).split("\n"):
                    st.markdown(f"- {line}")

            with col_val:
                st.markdown("### ✔️ Validación de balance")
                if is_valid:
                    st.success(msg)
                else:
                    st.error(msg)

            st.markdown("---")
            col_r, col_s = st.columns(2)

            with col_r:
                st.markdown("#### 🚛 Camiones de entrada (oferta por producto)")
                r_data = {
                    f"Producto {k}": [R[i][k] for i in range(1, data['num_inbound']+1)]
                    for k in range(1, data['num_products']+1)
                }
                r_df = pd.DataFrame(r_data, index=[f"Camión E{i}" for i in range(1, data['num_inbound']+1)])
                r_df["TOTAL"] = r_df.sum(axis=1)
                st.dataframe(r_df.style.highlight_max(axis=0, color="#dbeafe"), use_container_width=True)

            with col_s:
                st.markdown("#### 📦 Camiones de salida (demanda por producto)")
                s_data = {
                    f"Producto {k}": [S[j][k] for j in range(1, data['num_outbound']+1)]
                    for k in range(1, data['num_products']+1)
                }
                s_df = pd.DataFrame(s_data, index=[f"Camión S{j}" for j in range(1, data['num_outbound']+1)])
                s_df["TOTAL"] = s_df.sum(axis=1)
                st.dataframe(s_df.style.highlight_max(axis=0, color="#dcfce7"), use_container_width=True)

            # Heatmap de flujos potenciales
            st.markdown("#### 🔥 Heatmap — Oferta vs. Demanda por producto")
            prod_summary = pd.DataFrame({
                "Oferta total (entrada)": [
                    sum(R[i][k] for i in range(1, data['num_inbound']+1))
                    for k in range(1, data['num_products']+1)
                ],
                "Demanda total (salida)": [
                    sum(S[j][k] for j in range(1, data['num_outbound']+1))
                    for k in range(1, data['num_products']+1)
                ],
            }, index=[f"P{k}" for k in range(1, data['num_products']+1)])

            fig_heat = px.bar(
                prod_summary.reset_index().melt(id_vars="index"),
                x="index", y="value", color="variable",
                barmode="group",
                labels={"index": "Producto", "value": "Unidades", "variable": ""},
                color_discrete_map={
                    "Oferta total (entrada)": "#2563eb",
                    "Demanda total (salida)": "#16a34a",
                },
                template="plotly_white",
            )
            fig_heat.update_layout(
                height=320, margin=dict(t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        except Exception as e:
            st.error(f"Error al parsear el archivo: {e}")
    else:
        st.info("Selecciona o sube un archivo TS para continuar.")


# ===========================================================================
# TAB 3 — OPTIMIZACIÓN
# ===========================================================================
with tabs[2]:
    st.markdown("## 🚀 Ejecutar optimización")

    if 'data' not in st.session_state:
        st.warning("⚠️ Primero carga los datos en la pestaña **📊 Datos**.")
    else:
        data = st.session_state['data']
        R    = st.session_state['R']
        S    = st.session_state['S']

        col_info, col_btn = st.columns([2, 1])
        with col_info:
            st.markdown(f"""
**Instancia cargada:**
- Camiones de entrada: **{data['num_inbound']}**
- Camiones de salida: **{data['num_outbound']}**
- Tipos de producto: **{data['num_products']}**
- Tiempo límite del solver: **{time_limit} segundos**
- Gap MIP tolerado: **{mip_gap_pct}%**
            """)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            run_button = st.button("▶️ Ejecutar modelo MIP", type="primary", use_container_width=True)

        if run_button:
            with st.spinner("⏳ Resolviendo modelo MIP con PuLP/CBC..."):
                result = solve_crossdocking(
                    num_inbound=data['num_inbound'],
                    num_outbound=data['num_outbound'],
                    num_products=data['num_products'],
                    R=R, S=S,
                    time_limit=time_limit,
                    mip_gap=mip_gap_pct / 100,
                )
                st.session_state['result'] = result

            if run_heuristic_flag:
                with st.spinner("⏳ Calculando heurística SPT de referencia..."):
                    heuristic = solve_heuristic(
                        data['num_inbound'], data['num_outbound'],
                        data['num_products'], R, S,
                    )
                    st.session_state['heuristic'] = heuristic

            status = st.session_state['result'].status
            if status in ("Optimal", "Feasible"):
                st.success(f"✅ Solución encontrada — Estado: **{status}**")
            elif status == "Heuristic":
                st.warning("🟡 Solo heurística disponible.")
            else:
                st.error(f"❌ El solver reportó: {status}")

            st.info("📈 Ve a la pestaña **Resultados** y **Gantt** para explorar la solución.")

        if 'result' in st.session_state:
            st.markdown("---")
            st.markdown("### ℹ️ Información del modelo")
            result = st.session_state['result']
            info = result.model_info
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Variables", info.get('num_variables', '—'))
            col_b.metric("Restricciones", info.get('num_constraints', '—'))
            col_c.metric("Tiempo de resolución", f"{result.solve_time} seg")


# ===========================================================================
# TAB 4 — RESULTADOS
# ===========================================================================
with tabs[3]:
    st.markdown("## 📈 Resultados óptimos")

    if 'result' not in st.session_state:
        st.warning("⚠️ Ejecuta el modelo primero en la pestaña **🚀 Optimización**.")
    else:
        result = st.session_state['result']
        heuristic = st.session_state.get('heuristic', None)

        # ---- KPIs principales ----
        st.markdown("### 🎯 Indicadores clave")
        total_units = result.direct_units + result.storage_units
        pct_direct = (result.direct_units / total_units * 100) if total_units else 0

        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            st.markdown(metric_card(
                "Makespan óptimo",
                f"{result.makespan:.0f} min",
                minutes_to_hhmm(result.makespan),
                "blue",
            ), unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown(metric_card(
                "Unidades directas",
                f"{result.direct_units:,}",
                f"{pct_direct:.1f}% del total",
                "green",
            ), unsafe_allow_html=True)
        with kpi_cols[2]:
            st.markdown(metric_card(
                "Unidades vía almacén",
                f"{result.storage_units:,}",
                f"{100-pct_direct:.1f}% del total",
                "orange",
            ), unsafe_allow_html=True)
        with kpi_cols[3]:
            savings = ""
            if heuristic:
                diff = heuristic.makespan - result.makespan
                pct = diff / heuristic.makespan * 100 if heuristic.makespan else 0
                savings = f"{pct:.1f}% mejor que SPT"
            st.markdown(metric_card(
                "Mejora vs. Heurística",
                f"{savings.split('%')[0]}%" if savings else "—",
                savings,
                "purple",
            ), unsafe_allow_html=True)

        st.markdown("---")

        # ---- Secuencias ----
        col_seq_in, col_seq_out = st.columns(2)

        with col_seq_in:
            st.markdown("### 🚛 Secuencia óptima — Camiones de entrada")
            seq_in_df = pd.DataFrame([
                {
                    "Posición": s.order_position,
                    "Camión": f"E{s.truck_id}",
                    "Inicio (min)": s.start_time,
                    "Fin (min)": s.end_time,
                    "Duración (min)": round(s.end_time - s.start_time, 1),
                    "Unidades totales": sum(s.products.values()),
                }
                for s in result.inbound_schedule
            ])
            st.dataframe(seq_in_df, use_container_width=True, hide_index=True)

        with col_seq_out:
            st.markdown("### 📦 Secuencia óptima — Camiones de salida")
            seq_out_df = pd.DataFrame([
                {
                    "Posición": s.order_position,
                    "Camión": f"S{s.truck_id}",
                    "Inicio (min)": s.start_time,
                    "Fin (min)": s.end_time,
                    "Duración (min)": round(s.end_time - s.start_time, 1),
                    "Unidades totales": sum(s.products.values()),
                }
                for s in result.outbound_schedule
            ])
            st.dataframe(seq_out_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ---- Tabla de transferencias ----
        st.markdown("### 🔄 Tabla de transferencias")
        if result.transfers:
            trans_df = pd.DataFrame([
                {
                    "Camión entrada": f"E{t.inbound_truck}",
                    "Camión salida": f"S{t.outbound_truck}",
                    "Producto": f"P{t.product}",
                    "Unidades": t.units,
                    "Ruta": "📦 Almacén temporal" if t.via_storage else "✅ Directa",
                }
                for t in result.transfers
            ])

            def color_ruta(val):
                if "Directa" in str(val):
                    return "background-color: #dcfce7; color: #15803d; font-weight: 600"
                return "background-color: #fef3c7; color: #92400e; font-weight: 600"

            st.dataframe(
                trans_df.style.applymap(color_ruta, subset=["Ruta"]),
                use_container_width=True, hide_index=True,
            )

            # Gráfico de transferencias
            trans_agg = (
                trans_df.groupby(["Camión entrada", "Camión salida"])["Unidades"]
                .sum()
                .reset_index()
            )
            fig_trans = px.bar(
                trans_agg, x="Camión entrada", y="Unidades",
                color="Camión salida", barmode="stack",
                labels={"Unidades": "Unidades transferidas"},
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_trans.update_layout(height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig_trans, use_container_width=True)

        # ---- Comparación con heurística ----
        if heuristic:
            st.markdown("---")
            st.markdown("### 🆚 Comparación MIP vs. Heurística SPT")
            comp_df = pd.DataFrame({
                "Método": ["MIP (Óptimo)", "Heurística SPT"],
                "Makespan (min)": [result.makespan, heuristic.makespan],
                "Transferencias directas": [result.direct_units, heuristic.direct_units],
                "Vía almacén": [result.storage_units, heuristic.storage_units],
                "Tiempo de cómputo (seg)": [result.solve_time, heuristic.solve_time],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            fig_comp = px.bar(
                comp_df, x="Método", y="Makespan (min)",
                color="Método",
                color_discrete_map={"MIP (Óptimo)": "#2563eb", "Heurística SPT": "#f59e0b"},
                template="plotly_white",
                text="Makespan (min)",
            )
            fig_comp.update_traces(textposition="outside")
            fig_comp.update_layout(
                showlegend=False, height=320,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_comp, use_container_width=True)


# ===========================================================================
# TAB 5 — GANTT
# ===========================================================================
with tabs[4]:
    st.markdown("## 🗺️ Diagrama de Gantt — Programación de muelles")

    if 'result' not in st.session_state:
        st.warning("⚠️ Ejecuta el modelo primero.")
    else:
        result = st.session_state['result']

        # Construir datos para Gantt
        gantt_rows = []

        for s in result.inbound_schedule:
            gantt_rows.append({
                "Tarea": f"E{s.truck_id} (Descarga)",
                "Muelle": "Muelle Entrada",
                "Inicio": s.start_time,
                "Fin": s.end_time,
                "Tipo": "Entrada",
                "Unidades": sum(s.products.values()),
            })

        for s in result.outbound_schedule:
            gantt_rows.append({
                "Tarea": f"S{s.truck_id} (Carga)",
                "Muelle": "Muelle Salida",
                "Inicio": s.start_time,
                "Fin": s.end_time,
                "Tipo": "Salida",
                "Unidades": sum(s.products.values()),
            })

        gantt_df = pd.DataFrame(gantt_rows)

        # Gantt con Plotly
        colors = {"Entrada": "#2563eb", "Salida": "#16a34a"}
        fig_gantt = go.Figure()

        for _, row in gantt_df.iterrows():
            color = colors.get(row["Tipo"], "#6b7280")
            fig_gantt.add_trace(go.Bar(
                name=row["Muelle"],
                x=[row["Fin"] - row["Inicio"]],
                y=[row["Muelle"]],
                base=row["Inicio"],
                orientation="h",
                marker=dict(color=color, opacity=0.85, line=dict(color="white", width=1.5)),
                text=row["Tarea"],
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=(
                    f"<b>{row['Tarea']}</b><br>"
                    f"Inicio: {row['Inicio']:.1f} min<br>"
                    f"Fin: {row['Fin']:.1f} min<br>"
                    f"Duración: {row['Fin']-row['Inicio']:.1f} min<br>"
                    f"Unidades: {row['Unidades']}<extra></extra>"
                ),
                showlegend=False,
            ))

        # Línea de makespan
        fig_gantt.add_vline(
            x=result.makespan,
            line_dash="dash", line_color="red", line_width=2,
        )
        fig_gantt.add_annotation(
            x=result.makespan, y=1.05, yref="paper",
            text=f"Makespan = {result.makespan:.0f} min",
            showarrow=False,
            font=dict(color="red", size=12, family="IBM Plex Sans"),
            xanchor="left",
        )

        fig_gantt.update_layout(
            barmode="stack",
            xaxis=dict(title="Tiempo (minutos)", showgrid=True, gridcolor="#e5e7eb"),
            yaxis=dict(title="", categoryorder="array", categoryarray=["Muelle Salida", "Muelle Entrada"]),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="IBM Plex Sans", color="#1a1a2e"),
            height=380,
            margin=dict(l=140, r=40, t=40, b=60),
        )

        # Leyenda manual
        for label, color in colors.items():
            fig_gantt.add_trace(go.Bar(
                x=[None], y=[None],
                marker=dict(color=color),
                name=f"Camión de {'entrada' if label == 'Entrada' else 'salida'}",
                showlegend=True,
            ))

        fig_gantt.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
        )

        st.plotly_chart(fig_gantt, use_container_width=True)

        # Timeline de flujos
        st.markdown("### 🔄 Flujo de transferencias en el tiempo")

        if result.transfers:
            flow_data = []
            in_times = {s.truck_id: s.start_time for s in result.inbound_schedule}
            out_times = {s.truck_id: s.start_time for s in result.outbound_schedule}

            for t in result.transfers:
                flow_data.append({
                    "Par": f"E{t.inbound_truck}→S{t.outbound_truck} P{t.product}",
                    "Tiempo entrada": in_times.get(t.inbound_truck, 0),
                    "Tiempo salida": out_times.get(t.outbound_truck, 0),
                    "Unidades": t.units,
                    "Ruta": "Almacén" if t.via_storage else "Directa",
                })

            flow_df = pd.DataFrame(flow_data)
            fig_flow = px.scatter(
                flow_df,
                x="Tiempo entrada", y="Tiempo salida",
                size="Unidades", color="Ruta",
                hover_name="Par",
                labels={
                    "Tiempo entrada": "Camión entrada llega (min)",
                    "Tiempo salida": "Camión salida parte (min)",
                },
                color_discrete_map={"Directa": "#16a34a", "Almacén": "#f59e0b"},
                template="plotly_white",
                size_max=35,
            )
            fig_flow.add_shape(
                type="line",
                x0=0, y0=0,
                x1=max(flow_df["Tiempo entrada"].max(), flow_df["Tiempo salida"].max()) + 10,
                y1=max(flow_df["Tiempo entrada"].max(), flow_df["Tiempo salida"].max()) + 10,
                line=dict(color="#94a3b8", dash="dot"),
            )
            fig_flow.update_layout(height=350, margin=dict(t=20))
            st.plotly_chart(fig_flow, use_container_width=True)
            st.caption("Puntos por encima de la diagonal = flujo vía almacenamiento temporal.")


# ===========================================================================
# TAB 6 — CONCLUSIONES
# ===========================================================================
with tabs[5]:
    st.markdown("## 📝 Conclusiones automáticas")

    if 'result' not in st.session_state:
        st.warning("⚠️ Ejecuta el modelo primero.")
    else:
        result = st.session_state['result']
        heuristic = st.session_state.get('heuristic', None)
        data = st.session_state['data']
        total_units = result.direct_units + result.storage_units
        pct_direct = result.direct_units / total_units * 100 if total_units else 0

        st.markdown(f"""
### 🔍 Análisis de la solución óptima

#### 1. Tiempo de operación
El makespan óptimo encontrado es **{result.makespan:.0f} minutos** ({minutes_to_hhmm(result.makespan)}),
lo que representa el tiempo mínimo necesario para procesar todos los camiones de entrada
y salida en el centro de cross docking de LogiFast CR.

#### 2. Secuencia de camiones de entrada
El orden óptimo para atender los camiones de **entrada** es:
**{' → '.join(f'E{i}' for i in result.inbound_sequence)}**

Esta secuencia minimiza los tiempos muertos en el muelle de recepción y permite
que los productos estén disponibles en el momento correcto para los camiones de salida.

#### 3. Secuencia de camiones de salida
El orden óptimo para los camiones de **salida** es:
**{' → '.join(f'S{j}' for j in result.outbound_sequence)}**

#### 4. Uso del almacenamiento temporal
- Unidades transferidas **directamente**: {result.direct_units:,} ({pct_direct:.1f}%)
- Unidades vía **almacenamiento temporal**: {result.storage_units:,} ({100-pct_direct:.1f}%)

{'✅ La mayoría de los flujos son directos, lo que indica una buena sincronización entre camiones.' if pct_direct >= 60 else '⚠️ Una parte significativa de los productos pasa por almacenamiento temporal, lo que genera tiempos adicionales. Considerar ajustar los horarios de llegada.'}

#### 5. Comparación con heurística
        """)

        if heuristic:
            diff = heuristic.makespan - result.makespan
            pct_imp = diff / heuristic.makespan * 100
            st.markdown(f"""
La heurística SPT (Shortest Processing Time) produciría un makespan de **{heuristic.makespan:.0f} min**.
El modelo MIP logra una **reducción de {diff:.0f} minutos ({pct_imp:.1f}%)**, demostrando
el valor de la optimización matemática sobre una regla simple.
            """)

        st.markdown("""
#### 6. Recomendaciones operativas
        """)

        recommendations = []
        if pct_direct < 50:
            recommendations.append(
                "📌 **Coordinar horarios de llegada**: Negociar ventanas de tiempo con proveedores "
                "para que lleguen más cerca del momento en que sus productos son necesarios."
            )
        recommendations.append(
            "📌 **Respetar la secuencia calculada**: Cualquier desviación del orden óptimo "
            "puede incrementar significativamente el makespan total."
        )
        recommendations.append(
            "📌 **Monitorear el almacenamiento temporal**: Aunque la capacidad es ilimitada, "
            "cada unidad en almacén agrega 5 minutos de traslado. Minimizar su uso es clave."
        )
        if result.storage_units > 0:
            recommendations.append(
                "📌 **Analizar restricciones de muelle único**: Con un solo muelle de entrada "
                "y uno de salida, la solución es inherentemente secuencial. Evaluar la inversión "
                "en capacidad adicional si el volumen crece."
            )

        for rec in recommendations:
            st.markdown(rec)

        st.markdown("---")
        st.markdown("#### 📌 Supuestos del modelo")
        st.info("""
**Supuestos realizados:**
1. Todos los camiones están disponibles desde el inicio (t=0).
2. La transferencia directa es posible si el camión de entrada termina antes de que el camión de salida inicie su carga.
3. El tiempo de traslado interno (5 min) aplica por cada par camión entrada-salida con transferencia.
4. La capacidad del almacenamiento temporal es ilimitada en cantidad, pero costosa en tiempo.
5. No se consideran prioridades explícitas entre clientes.
6. El tiempo de carga/descarga es estrictamente 1 min/unidad.
        """)

        # Exportar resultados
        st.markdown("---")
        st.markdown("#### 💾 Exportar resultados")

        # Construir Excel en memoria
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            seq_in_df = pd.DataFrame([
                {"Posición": s.order_position, "Camión": f"E{s.truck_id}",
                 "Inicio (min)": s.start_time, "Fin (min)": s.end_time}
                for s in result.inbound_schedule
            ])
            seq_out_df = pd.DataFrame([
                {"Posición": s.order_position, "Camión": f"S{s.truck_id}",
                 "Inicio (min)": s.start_time, "Fin (min)": s.end_time}
                for s in result.outbound_schedule
            ])
            trans_df = pd.DataFrame([
                {"Camión entrada": f"E{t.inbound_truck}",
                 "Camión salida": f"S{t.outbound_truck}",
                 "Producto": f"P{t.product}",
                 "Unidades": t.units,
                 "Ruta": "Almacén" if t.via_storage else "Directa"}
                for t in result.transfers
            ])
            kpi_df = pd.DataFrame({
                "Indicador": ["Makespan (min)", "Unidades directas", "Unidades almacén"],
                "Valor": [result.makespan, result.direct_units, result.storage_units],
            })

            seq_in_df.to_excel(writer, sheet_name="Secuencia Entrada", index=False)
            seq_out_df.to_excel(writer, sheet_name="Secuencia Salida", index=False)
            trans_df.to_excel(writer, sheet_name="Transferencias", index=False)
            kpi_df.to_excel(writer, sheet_name="KPIs", index=False)

        st.download_button(
            label="⬇️ Descargar resultados (Excel)",
            data=output.getvalue(),
            file_name="logifast_resultado_optimo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
