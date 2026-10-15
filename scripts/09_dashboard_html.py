#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_dashboard_html.py
====================
Genera `resultados/dashboard_turismo_SUP.html`: un dashboard interactivo y
AUTOCONTENIDO (CSS + JavaScript + datos embebidos, sin recursos externos ni
servidor) para explorar los mercados turísticos internacionales de Bolivia.

Estructura del dashboard (6 pestañas; navegación por pestañas con `showTab`):
  * Pestaña 1 MERCADOS PRIORITARIOS (Pregunta 1), en dos niveles: Nivel 1 compara
    las 7 regiones de procedencia (volumen, participación, variación anual y
    evolución histórica); Nivel 2 muestra los países de la región elegida
    (top-5 por volumen, tabla con participación, variación, tendencia histórica
    y estacionalidad). La conclusión se deriva de los datos y de los filtros.
  * Pestaña 2 CALENDARIO DE CAMPAÑAS (Pregunta 2): distribución mensual del
    mercado o región elegido, comparación de hasta 5 países, meses de mayor y
    menor afluencia y recomendación comercial basada en los contrastes existentes
    entre regiones y países, sin inferir plazos de anticipación publicitaria.
  * Pestaña 3 MERCADOS PARA INVESTIGAR (Pregunta 3): señales de tendencia
    separadas para llegadas por nacionalidad y gasto por país de residencia,
    con interpretación exploratoria, escenarios y contexto del gasto total.
  * Pestaña 4 DATOS Y DETALLE: llegadas por nacionalidad (KPIs, evolución,
    ranking, estacionalidad y tabla), estacionalidad regional (vistas A/B/C),
    gasto por país de residencia y resumen descriptivo.
  * Pestaña 5 PRUEBAS ESTADÍSTICAS: chi-cuadrado región x mes, prueba europea
    previa y análisis inferencial complementario (Etapa 5).
  * Pestaña 6 FUENTES Y METODOLOGÍA: fuentes y archivos, definición de los
    indicadores, nacionalidad vs. residencia, categorías residuales, datos (p),
    las advertencias metodológicas, criterios de las pruebas y limitaciones.

Cada pestaña tiene sus propios filtros: elegir una región en la Pregunta 1 NO
cambia las Preguntas 2 y 3. La prioridad de mercado reutiliza los criterios
explícitos V/C/T de `tabla_decision_mercados_potenciales.csv`.

Reglas de integridad:
  - Sólo LECTURA sobre `resultados/*.csv`. No se modifican CSV, gráficas ni el
    informe `informe_final_SUP.html`.
  - Criterio de "visitantes extranjeros" (idéntico a
    `scripts/03_analisis_descriptivo.py`): filas `nivel=="detalle"` EXCLUYENDO
    `Bolivia` (nacionales). NO se usa `nivel=="total"` como extranjeros, porque
    incluye a los nacionales. La categoría residual `Resto de países` se cuenta
    en el TOTAL extranjero pero NO se trata como país ni entra en rankings ni en
    agregados regionales.
  - Visitantes = nacionalidad; gasto = país de residencia. NUNCA se cruzan.
  - Los valores faltantes (NaN) se conservan como "n/d"; no se convierten en 0.
  - No se infiere gasto mensual, gasto por visitante, demanda SUP ni
    disposición a pagar.

Uso:
    python3 scripts/09_dashboard_html.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm, rankdata

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "resultados"

F_MENSUAL = RES / "visitantes_mensual_largo.csv"
F_ANUAL = RES / "visitantes_anual_largo.csv"
F_GASTO = RES / "gasto_pais_anual_largo.csv"
F_PRINCIPAL = RES / "visitantes_principal_2022_2025.csv"
F_TXT_EU = RES / "resultado_prueba_inferencial.txt"
# Etapa 5 (análisis inferencial complementario); pueden no existir todavía.
F_INF_TEND = RES / "tendencias_mann_kendall_por_pais.csv"
F_INF_HOMOG = RES / "homogeneidad_estacional_paises.csv"
F_INF_DEC = RES / "tabla_decision_mercados_potenciales.csv"
F_INF_RES = RES / "homogeneidad_estacional_residuos.csv"
F_EVOL = RES / "mercados_evolucion_reciente_2022_2025.csv"

OUT = RES / "dashboard_turismo_SUP_final.html"

MESES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]

# --------------------------------------------------------------------------- #
# Clasificación geográfica (regiones). Convenciones documentadas:
#   * México -> Norteamérica (decisión explícita del proyecto).
#   * Rusia y Turquía -> Europa por convención transcontinental, coherente con
#     la lista EUROPA de scripts/03, 05 y 06 del proyecto.
#   * Israel -> Asia (decisión explícita del proyecto).
#   * Rusia, Turquía e Israel son transcontinentales; se listan aquí para
#     constancia. No hay países sin clasificar: 51 mercados reales en total.
# --------------------------------------------------------------------------- #
REGIONES: dict[str, list[str]] = {
    "Europa": [
        "Alemania", "Austria", "Bélgica", "Dinamarca", "Eslovaquia", "España",
        "Finlandia", "Francia", "Grecia", "Irlanda", "Italia", "Noruega",
        "Países Bajos", "Polonia", "Portugal", "Reino Unido", "República Checa",
        "Rumania", "Suecia", "Suiza", "Rusia", "Turquia",
    ],
    "Norteamérica": ["Canadá", "Estados Unidos", "México"],
    "Sudamérica": [
        "Argentina", "Brasil", "Chile", "Colombia", "Ecuador", "Paraguay",
        "Perú", "Uruguay", "Venezuela",
    ],
    "Centroamérica y Caribe": [
        "Costa Rica", "Cuba", "El Salvador", "Guatemala", "Honduras",
        "Nicaragua", "Panamá", "República Dominicana",
    ],
    "Asia": ["China", "Corea del Sur", "India", "Israel", "Japón", "Taiwan"],
    "África": ["Sudáfrica"],
    "Oceanía": ["Australia", "Nueva Zelandia"],
}
MAPA_REGION = {p: r for r, ps in REGIONES.items() for p in ps}

# El archivo de gasto usa "Irlanda (EIRE)"; se asigna a la misma región.
GASTO_ALIAS = {"Irlanda (EIRE)": "Irlanda"}

# Los 8 mercados europeos usados en la prueba inferencial previa (05_pruebas).
EUROPA_TOP8_DOC = ["España", "Francia", "Alemania", "Reino Unido",
                   "Italia", "Suiza", "Países Bajos", "Bélgica"]


def _num(x):
    """Convierte a float nativo o None (conserva los faltantes como None)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return float(x)


def _region_de_gasto(pais: str) -> str | None:
    return MAPA_REGION.get(pais) or MAPA_REGION.get(GASTO_ALIAS.get(pais, ""))


def _chi2_tabla(matriz: np.ndarray) -> dict:
    """Chi-cuadrado de independencia + V de Cramér sobre una matriz de conteos."""
    chi2, p_val, dof, esperado = chi2_contingency(matriz, correction=False)
    n = float(matriz.sum())
    v = float(np.sqrt((chi2 / n) / min(matriz.shape[0] - 1, matriz.shape[1] - 1)))
    return {
        "chi2": float(chi2), "gl": int(dof), "p": float(p_val), "n": n,
        "cramerV": v, "minEsperado": float(esperado.min()),
        "pctEsperadoLt5": float((esperado < 5).mean() * 100),
    }


def _bh(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg sobre los p-valores disponibles."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.size, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out.tolist()
    ix = np.flatnonzero(valid)
    order = np.argsort(p[ix])
    ranked = p[ix][order]
    q = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[ix[order]] = np.clip(q, 0, 1)
    return out.tolist()


def _lag1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    den = float(np.sum(x * x))
    return float(np.sum(x[:-1] * x[1:]) / den) if den > 0 and x.size > 1 else 0.0


def _mk_gasto(x: np.ndarray) -> dict:
    """Mann–Kendall con empates y continuidad; mismo diseño que Etapa 5."""
    x = np.asarray(x, dtype=float)
    n = x.size
    s = 0.0
    for i in range(n - 1):
        s += float(np.sign(x[i + 1:] - x[i]).sum())
    _, counts = np.unique(x, return_counts=True)
    tie = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    var_s = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    if var_s <= 0 or s == 0:
        z = 0.0
    else:
        z = (s - np.sign(s)) / np.sqrt(var_s)
    return {"S": s, "varS": var_s, "p": float(2 * norm.sf(abs(z))), "n": int(n)}


def _hamed_rao_gasto(x: np.ndarray, var_s: float) -> dict:
    """Ajusta Mann–Kendall por autocorrelación en los rangos (Hamed–Rao)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    ranked = rankdata(x)
    cf = 1.0
    if n >= 4:
        # Reproduce el factor de Hamed–Rao usado en scripts/10, calculando la
        # autocorrelación de rango a cada retardo sobre el rango completo.
        def rho_lag(z, k):
            z = z - z.mean()
            den = float(np.sum(z * z))
            return float(np.sum(z[:-k] * z[k:]) / den) if den > 0 else 0.0
        total = sum((n - lag) * (n - lag - 1) * (n - lag - 2) * rho_lag(ranked, lag)
                    for lag in range(1, n - 1))
        cf = max(1.0 + (2.0 / (n * (n - 1) * (n - 2))) * total, 1.0)
    var_hr = var_s * cf
    mk = _mk_gasto(x)
    s = mk["S"]
    z = 0.0 if var_hr <= 0 or s == 0 else (s - np.sign(s)) / np.sqrt(var_hr)
    return {"rho1": _lag1(x), "factor": float(cf), "p": float(2 * norm.sf(abs(z))), "varS": float(var_hr)}


def _sen_gasto(years: list[int], x: np.ndarray, var_s: float) -> tuple[float, float | None, float | None]:
    """Pendiente de Sen por año; IC95% con varianza ajustada cuando hay autocorr."""
    slopes = np.asarray([(x[j] - x[i]) / (years[j] - years[i])
                         for i in range(len(x) - 1) for j in range(i + 1, len(x))], dtype=float)
    slopes.sort()
    if not len(slopes):
        return float("nan"), None, None
    sen = float(np.median(slopes))
    c = norm.ppf(0.975) * np.sqrt(max(var_s, 0.0))
    lo = int(np.ceil((len(slopes) - c) / 2.0))
    hi = int(np.ceil((len(slopes) + c) / 2.0))
    if lo < 1 or hi > len(slopes):
        return sen, None, None
    return sen, float(slopes[lo - 1]), float(slopes[hi - 1])


def construir_tendencias_gasto(gasto_pais: dict[str, dict[str, float]]) -> dict:
    """Mann–Kendall y Sen en gasto por residencia; BH por periodo y HR por serie."""
    periodos = {"pre": (2008, 2019), "hist": (2008, 2025)}
    resultado: dict[str, dict] = {p: {} for p in gasto_pais}
    for clave, (ini, fin) in periodos.items():
        pendientes = []
        for pais, serie in gasto_pais.items():
            years = [y for y in range(ini, fin + 1) if serie.get(str(y)) is not None]
            vals = np.asarray([serie[str(y)] for y in years], dtype=float)
            completos = len(years) == (fin - ini + 1) and years == list(range(ini, fin + 1))
            if len(vals) < 8 or not completos:
                resultado[pais][clave] = {"evaluable": False, "n": int(len(vals)), "desde": ini, "hasta": fin}
                pendientes.append(np.nan)
                continue
            mk = _mk_gasto(vals)
            hr = _hamed_rao_gasto(vals, mk["varS"])
            sen, lo, hi = _sen_gasto(years, vals, hr["varS"])
            resultado[pais][clave] = {"evaluable": True, "n": int(len(vals)), "desde": ini, "hasta": fin,
                "S": mk["S"], "sen": sen, "icInf": lo, "icSup": hi, "p": mk["p"],
                "rho1": hr["rho1"], "factorHR": hr["factor"], "pHR": hr["p"], "pBH": None,
                "pBHHR": None, "concl": False, "direccion": "sin evidencia suficiente"}
            pendientes.append(mk["p"])
        q = _bh(pendientes)
        qhr = _bh([resultado[p][clave]["pHR"] if resultado[p][clave].get("evaluable") else np.nan
                   for p in gasto_pais])
        for idx, pais in enumerate(gasto_pais):
            row = resultado[pais][clave]
            if not row.get("evaluable"):
                continue
            row["pBH"], row["pBHHR"] = q[idx], qhr[idx]
            nominal = q[idx] < 0.05
            material = row["factorHR"] > 1.05
            concl = bool(nominal and (not material or qhr[idx] < 0.05))
            if not nominal:
                direc = "sin evidencia suficiente"
            elif row["sen"] > 0:
                direc = "creciente" if concl else "creciente, no concluyente tras autocorrelación"
            elif row["sen"] < 0:
                direc = "decreciente" if concl else "decreciente, no concluyente tras autocorrelación"
            else:
                direc = "sin evidencia suficiente"
            row["concl"], row["direccion"] = concl, direc
    return resultado


def construir_inferencial() -> dict:
    """Resumen COMPACTO de la Etapa 5 para el dashboard (sin recargar la interfaz).

    Lee las salidas ya producidas por scripts/10_analisis_inferencial_complementario.py.
    Si no existen, devuelve disponible=False (la sección muestra un aviso).
    """
    archivos = [F_INF_TEND.name, F_INF_HOMOG.name, F_INF_DEC.name,
                "homogeneidad_estacional_residuos.csv",
                "mercados_evolucion_reciente_2022_2025.csv",
                "interpretacion_inferencial_complementaria.txt"]
    if not (F_INF_TEND.exists() and F_INF_HOMOG.exists() and F_INF_DEC.exists()):
        return {"disponible": False, "archivos": archivos}
    tend = pd.read_csv(F_INF_TEND)
    hom = pd.read_csv(F_INF_HOMOG)
    dec = pd.read_csv(F_INF_DEC)
    conci = "concluyente_autocorr"

    def resumen_periodo(periodo):
        ev = tend[(tend.periodo == periodo) & (tend.p_BH.notna())]
        up = ev[(ev.p_BH < 0.05) & (ev.sen_pendiente > 0)]
        dn = ev[(ev.p_BH < 0.05) & (ev.sen_pendiente < 0)]
        top = up.sort_values("sen_pendiente", ascending=False)
        return {
            "evaluados": int(len(ev)),
            "nominalArriba": int(len(up)), "nominalAbajo": int(len(dn)),
            "conclArriba": int(up[conci].sum()) if conci in up else 0,
            "conclAbajo": int(dn[conci].sum()) if conci in dn else 0,
            "top": [{"pais": r.pais, "sen": _num(r.sen_pendiente), "pBH": _num(r.p_BH),
                     "concl": bool(getattr(r, conci))} for r in top.head(8).itertuples()],
        }

    homog = []
    for _, r in hom.iterrows():
        homog.append({"ambito": r.ambito, "tipo": r.tipo, "paises": r.paises_incluidos,
                      "nPaises": int(r.n_paises), "chi2": _num(r.chi2),
                      "gl": None if pd.isna(r.gl) else int(r.gl), "p": _num(r.p_valor),
                      "pBH": _num(r.p_BH), "V": _num(r.V_cramer), "minEsp": _num(r.min_esperado),
                      "pctLt5": _num(r.pct_celdas_lt5), "decision": r.decision})

    orden = {"Alta": 0, "Media": 1, "Baja": 2}
    dec = dec.assign(_o=dec.prioridad_sugerida.map(orden))
    dec = dec.sort_values(["_o", "llegadas_2025"], ascending=[True, False])
    detalle = [{"pais": r.pais, "region": r.region,
                "v2025": _num(r.llegadas_2025), "var": _num(r.var_2024_2025_pct),
                "prioridad": r.prioridad_sugerida, "patron": r.patron_estacional,
                "tendencia": r.tendencia_historica}
               for r in dec.head(15).itertuples()]
    # --- Detalle ESTRUCTURADO por mercado (51) para los módulos de decisión - #
    # No se recalcula nada nuevo: se reordenan las salidas ya producidas por
    # scripts/10_analisis_inferencial_complementario.py.
    tend_est: dict[str, dict] = {}
    for _, r in tend.iterrows():
        clave = "pre" if r["periodo"] == "pre_pandemia_2008_2019" else "hist"
        tend_est.setdefault(r["pais"], {})[clave] = {
            "sen": _num(r["sen_pendiente"]), "icInf": _num(r["sen_ic_inf"]),
            "icSup": _num(r["sen_ic_sup"]), "pBH": _num(r["p_BH"]),
            "concl": bool(r["concluyente_autocorr"]), "etq": r["tendencia"],
            "n": None if pd.isna(r["n_obs_validas"]) else int(r["n_obs_validas"]),
            "autocorr": _num(r["autocorr_lag1"]),
        }
    evol_est: dict[str, dict] = {}
    if F_EVOL.exists():
        for _, r in pd.read_csv(F_EVOL).iterrows():
            evol_est[r["pais"]] = {"total2225": _num(r["total_2022_2025"]),
                                   "v2025": _num(r["llegadas_2025"]),
                                   "var": _num(r["var_2025_vs_2024_pct"])}
    mercados: dict[str, dict] = {}
    for _, r in dec.iterrows():
        p = r["pais"]
        ev = evol_est.get(p, {})
        mercados[p] = {
            "region": r["region"], "v2025": _num(r["llegadas_2025"]),
            "var": _num(r["var_2024_2025_pct"]), "patron": r["patron_estacional"],
            "tendencia": r["tendencia_historica"], "inferencia": r["resultado_inferencial"],
            "limitaciones": r["limitaciones"], "prioridad": r["prioridad_sugerida"],
            "criterios": r["criterios_prioridad"],
            "total2225": ev.get("total2225"), "tend": tend_est.get(p, {}),
        }

    # --- Residuos estacionales con mayor aporte por región (máx. 6) --------- #
    residuos: dict[str, list] = {}
    if F_INF_RES.exists():
        rr = pd.read_csv(F_INF_RES)
        for reg, sub in rr.groupby("region"):
            sel = sub.assign(_a=sub["residuo_ajustado"].abs()).sort_values(
                "_a", ascending=False).head(6)
            residuos[reg] = [{"pais": x.pais, "mes": x.mes, "r": _num(x.residuo_ajustado),
                              "signif": bool(x.significativo_BH)} for x in sel.itertuples()]

    prior = {k: int(v) for k, v in dec.prioridad_sugerida.value_counts().items()}
    return {"disponible": True, "archivos": archivos,
            "pre": resumen_periodo("pre_pandemia_2008_2019"),
            "hist": resumen_periodo("historico_2008_2025"),
            "homogeneidad": homog, "prioridades": prior, "detalle": detalle,
            "mercados": mercados, "residuos": residuos,
            "periodoPre": "pre_pandemia_2008_2019", "periodoHist": "historico_2008_2025"}


def construir_datos() -> dict:
    mensual = pd.read_csv(F_MENSUAL)
    anual = pd.read_csv(F_ANUAL)
    gasto = pd.read_csv(F_GASTO)

    # --- Visitantes extranjeros -------------------------------------------- #
    det_ext = mensual[(mensual["nivel"] == "detalle") & (mensual["nacionalidad"] != "Bolivia")].copy()
    mercados_df = det_ext[~det_ext["es_residual"]].copy()          # 51 países reales

    nacionalidades = sorted(mercados_df["nacionalidad"].dropna().unique().tolist())
    anios_vis = [int(a) for a in sorted(mensual["anio"].unique())]

    mapa_mes = (
        mensual.dropna(subset=["mes_num"])[["mes_num", "mes"]]
        .drop_duplicates().set_index("mes_num")["mes"].to_dict()
    )
    meses = [mapa_mes.get(i + 1, MESES[i]) for i in range(12)]

    monthly: dict[str, dict[str, list]] = {}
    prelim_m: dict[str, dict[str, list]] = {}
    for nat, sub in mercados_df.groupby("nacionalidad"):
        serie: dict[str, list] = {}
        prel: dict[str, list] = {}
        for _, r in sub.iterrows():
            y = str(int(r["anio"]))
            arr = serie.setdefault(y, [None] * 12)
            arr[int(r["mes_num"]) - 1] = _num(r["valor"])
            if bool(r["es_preliminar"]):
                prel.setdefault(y, []).append(int(r["mes_num"]))
        monthly[nat] = serie
        prelim_m[nat] = prel

    total_monthly: dict[str, list] = {}
    total_prelim: dict[str, list] = {}
    for _, r in det_ext.iterrows():
        y = str(int(r["anio"]))
        arr = total_monthly.setdefault(y, [None] * 12)
        m = int(r["mes_num"]) - 1
        v = _num(r["valor"])
        if v is not None:
            arr[m] = v if arr[m] is None else arr[m] + v
        if bool(r["es_preliminar"]):
            total_prelim.setdefault(y, []).append(int(r["mes_num"]))

    mercados_a = anual[(anual["nivel"] == "detalle") & (~anual["es_residual"])
                       & (anual["nacionalidad"] != "Bolivia")]
    annual: dict[str, dict[str, float]] = {}
    for _, r in mercados_a.iterrows():
        annual.setdefault(r["nacionalidad"], {})[str(int(r["anio"]))] = _num(r["valor"])

    det_ext_a = anual[(anual["nivel"] == "detalle") & (anual["nacionalidad"] != "Bolivia")]
    total_annual: dict[str, float] = {}
    for _, r in det_ext_a.iterrows():
        y = str(int(r["anio"]))
        v = _num(r["valor"])
        if v is not None:
            total_annual[y] = (total_annual.get(y, 0.0) or 0.0) + v

    # --- Agregados por región (llegadas) ----------------------------------- #
    region_monthly: dict[str, dict[str, list]] = {}
    region_annual: dict[str, dict[str, float]] = {}
    for reg, paises in REGIONES.items():
        rm: dict[str, list] = {}
        ra: dict[str, float] = {}
        for p in paises:
            for y, arr in monthly.get(p, {}).items():
                dest = rm.setdefault(y, [None] * 12)
                for i, v in enumerate(arr):
                    if v is not None:
                        dest[i] = v if dest[i] is None else dest[i] + v
            for y, v in annual.get(p, {}).items():
                if v is not None:
                    ra[y] = ra.get(y, 0.0) + v
        region_monthly[reg] = rm
        region_annual[reg] = ra

    P_DESDE, P_HASTA = 2022, 2025
    regiones = list(REGIONES.keys())

    # --- Chi-cuadrado región x mes (2022-2025) ----------------------------- #
    m4 = mercados_df[mercados_df["anio"].between(P_DESDE, P_HASTA)]
    tabla_reg: dict[str, list] = {r: [0.0] * 12 for r in regiones}
    for _, r in m4.iterrows():
        reg = MAPA_REGION.get(r["nacionalidad"])
        if reg is None:
            continue
        v = _num(r["valor"])
        if v is not None:
            tabla_reg[reg][int(r["mes_num"]) - 1] += v
    matriz_reg = np.array([tabla_reg[r] for r in regiones], dtype=float)
    base_reg = _chi2_tabla(matriz_reg)
    stats_region = {
        "tabla": {r: [float(x) for x in tabla_reg[r]] for r in regiones},
        "regiones": regiones,
        "meses": meses,
        "filaTotales": {r: float(sum(tabla_reg[r])) for r in regiones},
        "colTotales": [float(matriz_reg[:, j].sum()) for j in range(12)],
        "desde": P_DESDE,
        "hasta": P_HASTA,
        **base_reg,
    }

    # --- Prueba europea previa (país x mes, top-8, 2022-2025), recomputada -- #
    # Se reproduce el diseño de `scripts/05_pruebas_inferenciales.py` y se
    # contrasta con el archivo guardado `resultado_prueba_inferencial.txt`.
    # Es un análisis ESPECÍFICO DE EUROPA: no se presenta como de todas las
    # regiones.
    eu = mercados_df[mercados_df["nacionalidad"].isin(REGIONES["Europa"])
                     & mercados_df["anio"].between(P_DESDE, P_HASTA)]
    vol_eu = eu.groupby("nacionalidad")["valor"].sum().sort_values(ascending=False)
    top8 = list(vol_eu.head(8).index)
    cobertura_eu = float(vol_eu.head(8).sum() / vol_eu.sum() * 100) if vol_eu.sum() else 0.0
    sub_eu = eu[eu["nacionalidad"].isin(top8)]
    tabla_eu = (sub_eu.pivot_table(index="nacionalidad", columns="mes_num", values="valor",
                                   aggfunc="sum", fill_value=0)
                .reindex(index=top8, columns=range(1, 13), fill_value=0).fillna(0))
    base_eu = _chi2_tabla(tabla_eu.values.astype(float))
    stats_europa = {
        "paises": top8,
        "cobertura": cobertura_eu,
        "nPaisesEuropa": int(vol_eu.shape[0]),
        "desde": P_DESDE,
        "hasta": P_HASTA,
        **base_eu,
    }
    # Contraste con el archivo de resultados guardado (verificable)
    guardado_chi = guardado_v = None
    coincide = False
    if F_TXT_EU.exists():
        t = F_TXT_EU.read_text(encoding="utf-8")
        mc = re.search(r"chi-cuadrado\s*=\s*([\d.,]+)", t)
        mv = re.search(r"V de Cramér\s*=\s*([\d.]+)", t)
        if mc:
            guardado_chi = float(mc.group(1).replace(",", ""))
        if mv:
            guardado_v = float(mv.group(1))
        if guardado_chi is not None and guardado_v is not None:
            coincide = (abs(guardado_chi - round(stats_europa["chi2"], 3)) < 0.01
                        and abs(guardado_v - round(stats_europa["cramerV"], 4)) < 0.001)
    stats_europa["chi2Guardado"] = guardado_chi
    stats_europa["vGuardado"] = guardado_v
    stats_europa["coincide"] = bool(coincide)
    stats_europa["archivo"] = F_TXT_EU.name

    # --- Gasto por país de residencia -------------------------------------- #
    gp = gasto[(gasto["nivel"] == "detalle") & (~gasto["es_residual"])]
    paises = sorted(gp["pais"].dropna().unique().tolist())
    gasto_pais: dict[str, dict[str, float]] = {}
    prelim_g: dict[str, dict[str, bool]] = {}
    for _, r in gp.iterrows():
        p = r["pais"]
        y = str(int(r["anio"]))
        gasto_pais.setdefault(p, {})[y] = _num(r["valor"])
        if bool(r["es_preliminar"]):
            prelim_g.setdefault(p, {})[y] = True

    resto = gasto[gasto["es_residual"]]
    gasto_resto = {str(int(r["anio"])): _num(r["valor"]) for _, r in resto.iterrows()}
    gtot = gasto[gasto["nivel"] == "total"]
    gasto_total = {str(int(r["anio"])): _num(r["valor"]) for _, r in gtot.iterrows()}
    anios_gasto = [int(a) for a in sorted(gasto["anio"].unique())]
    prelim_g_total = {str(int(r["anio"])): bool(r["es_preliminar"]) for _, r in gtot.iterrows()}

    # Gasto agregado por región: sólo países de detalle asignados a una región.
    # Se EXCLUYEN el TOTAL y la categoría residual 'Resto de Países'.
    gasto_region: dict[str, dict[str, float]] = {}
    gasto_region_paises: dict[str, list[str]] = {}
    for p in paises:
        reg = _region_de_gasto(p)
        if reg is None:
            continue
        gasto_region_paises.setdefault(reg, []).append(p)
        d = gasto_region.setdefault(reg, {})
        for y, v in gasto_pais[p].items():
            if v is not None:
                d[y] = d.get(y, 0.0) + v

    # Tendencias inferenciales calculadas sobre gasto por residencia (no sobre
    # llegadas). Se informan periodos pre-pandemia e histórico por la ruptura
    # observada en 2020–2021; 2023–2025 conservan su marca preliminar.
    tendencias_gasto = construir_tendencias_gasto(gasto_pais)
    gasto_ultimo_vals = [v.get("2025") for v in gasto_pais.values()
                         if v.get("2025") is not None]
    gasto_cuartiles = {
        "anio": 2025,
        "p25": float(np.percentile(gasto_ultimo_vals, 25)) if gasto_ultimo_vals else None,
        "p75": float(np.percentile(gasto_ultimo_vals, 75)) if gasto_ultimo_vals else None,
        "n": len(gasto_ultimo_vals),
    }

    # --- Metadatos de alcance ---------------------------------------------- #
    anios_vis_completos = [y for y in anios_vis if all(
        (total_monthly.get(str(y), [None] * 12)[m] is not None) for m in range(12)
    )]
    ultimo_vis_completo = max(anios_vis_completos) if anios_vis_completos else max(anios_vis)
    # El gasto es una serie ANUAL: cada año presente con dato se considera completo.
    anios_gasto_completos = [y for y in anios_gasto if gasto_total.get(str(y)) is not None]

    return {
        "meses": meses,
        "nacionalidades": nacionalidades,
        "aniosVis": anios_vis,
        "aniosVisCompletos": anios_vis_completos,
        "aniosGastoCompletos": anios_gasto_completos,
        "ultimoVisCompleto": ultimo_vis_completo,
        "monthly": monthly,
        "totalMonthly": total_monthly,
        "totalPrelim": total_prelim,
        "annual": annual,
        "totalAnnual": total_annual,
        "prelimM": prelim_m,
        "regiones": regiones,
        "regionPaises": {r: REGIONES[r] for r in regiones},
        "regionMonthly": region_monthly,
        "regionAnnual": region_annual,
        "statsRegion": stats_region,
        "statsEuropa": stats_europa,
        "paisesGasto": paises,
        "aniosGasto": anios_gasto,
        "gastoPais": gasto_pais,
        "gastoResto": gasto_resto,
        "gastoTotal": gasto_total,
        "gastoRegion": gasto_region,
        "gastoRegionPaises": gasto_region_paises,
        "gastoTendencias": tendencias_gasto,
        "gastoCuartiles": gasto_cuartiles,
        "prelimG": prelim_g,
        "prelimGTotal": prelim_g_total,
        "ultimoGasto": max(anios_gasto),
        "unidadGasto": "millones de USD",
        "infer": construir_inferencial(),
    }


# --------------------------------------------------------------------------- #
# Plantilla HTML (CSS + JS embebidos, sin recursos externos)
# --------------------------------------------------------------------------- #
CSS = r"""
:root{
  --azul:#12345a;--azul2:#2b6cb0;--azul3:#ebf3fb;--verde:#22704a;--verde-bg:#e7f5ee;
  --amber:#9a6b00;--amber-bg:#fdf4e3;--rosa:#b03a5b;--rosa-bg:#fdeaef;
  --gris:#5b6b7b;--linea:#dbe4ee;--fondo:#f5f7fa;--texto:#1c2733;--blanco:#fff;
  --sombra:0 1px 3px rgba(18,52,90,.08),0 8px 24px rgba(18,52,90,.06);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--fondo);color:var(--texto);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  font-size:16px;line-height:1.6;-webkit-text-size-adjust:100%}
h1,h2,h3{line-height:1.25;color:var(--azul);margin:0 0 .5em}
p{margin:0 0 1em}
a{color:var(--azul2)}
header.hero{background:linear-gradient(135deg,#0f2c4d,#1d5488);color:#fff;padding:38px 22px 30px}
header.hero h1{color:#fff;font-size:clamp(1.4rem,3.4vw,2.1rem);margin-bottom:.35em}
header.hero p{color:#d5e4f2;max-width:70ch;margin:0}
.wrap{max-width:1180px;margin:0 auto;padding:0 18px}
nav.topnav{position:sticky;top:0;z-index:20;background:#fff;border-bottom:1px solid var(--linea);
  box-shadow:0 1px 2px rgba(18,52,90,.05)}
nav.topnav .wrap{display:flex;gap:6px;flex-wrap:wrap;padding-top:8px;padding-bottom:8px}
nav.topnav a{font-size:.9rem;text-decoration:none;color:var(--azul);padding:7px 11px;border-radius:8px;font-weight:600}
nav.topnav a:hover,nav.topnav a.act{background:var(--azul3)}
main{padding:26px 0 60px}
section{background:var(--blanco);border:1px solid var(--linea);border-radius:14px;
  padding:22px;margin:0 0 22px;box-shadow:var(--sombra)}
section>h2{font-size:1.3rem;display:flex;align-items:center;gap:.5em;flex-wrap:wrap}
.tag{font-size:.72rem;font-weight:700;letter-spacing:.03em;text-transform:uppercase;
  padding:3px 9px;border-radius:999px;background:var(--azul3);color:var(--azul2)}
.tag.eu{background:var(--verde-bg);color:var(--verde)}
.filters{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:14px 0 18px}
.filters label{display:block;font-size:.78rem;font-weight:700;color:var(--gris);
  text-transform:uppercase;letter-spacing:.03em;margin-bottom:4px}
select,input[type=text]{width:100%;padding:9px 10px;border:1px solid var(--linea);border-radius:9px;
  font-size:.95rem;background:#fff;color:var(--texto)}
select:focus,input:focus{outline:2px solid var(--azul2);outline-offset:1px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:6px 0 18px}
.kpi{border:1px solid var(--linea);border-radius:12px;padding:15px 16px;background:linear-gradient(180deg,#fbfdff,#f3f8fd)}
.kpi .lab{font-size:.78rem;font-weight:700;color:var(--gris);text-transform:uppercase;letter-spacing:.03em}
.kpi .val{font-size:1.75rem;font-weight:800;color:var(--azul);margin:2px 0}
.kpi .sub{font-size:.8rem;color:var(--gris)}
.kpi .sub.up{color:var(--verde);font-weight:700}
.kpi .sub.down{color:var(--rosa);font-weight:700}
.statrow{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:12px 0}
.stat{background:var(--azul3);border:1px solid #cfe1f3;border-radius:10px;padding:10px 12px}
.stat .l{font-size:.7rem;font-weight:700;color:var(--gris);text-transform:uppercase;letter-spacing:.03em}
.stat .v{font-size:1.2rem;font-weight:800;color:var(--azul)}
.stat .v small{font-size:.72rem;font-weight:600;color:var(--gris)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:820px){.grid2{grid-template-columns:1fr}}
.card{border:1px solid var(--linea);border-radius:12px;padding:14px;background:#fff}
.card h3{font-size:1rem;margin-bottom:.15em}
.card .hint{font-size:.78rem;color:var(--gris);margin:0 0 10px}
svg{width:100%;height:auto;display:block;overflow:visible}
.chart-note{font-size:.76rem;color:var(--gris);margin-top:6px}
table{width:100%;border-collapse:collapse;font-size:.88rem}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--linea)}
th{background:#f2f6fb;color:var(--azul);white-space:nowrap;position:sticky;top:0}
th.sortable{cursor:pointer;user-select:none}
th.num,td.num{text-align:right}
tbody tr:hover{background:#f7fafd}
.tbl-wrap{max-height:430px;overflow:auto;border:1px solid var(--linea);border-radius:10px}
.tbl-scroll{overflow:auto;border:1px solid var(--linea);border-radius:10px}
.matrix{font-size:.8rem}
.matrix td.zero,.matrix td.num.zero{color:#c2ccd8}
.badge{display:inline-block;font-size:.68rem;font-weight:700;padding:2px 7px;border-radius:6px;white-space:nowrap}
.badge.p{background:var(--amber-bg);color:var(--amber)}
.badge.n{background:#eef2f6;color:var(--gris)}
.badge.obs{background:var(--verde-bg);color:var(--verde)}
.badge.hip{background:var(--rosa-bg);color:var(--rosa)}
.note{background:var(--amber-bg);border:1px solid #f2ddb0;border-radius:10px;padding:12px 14px;font-size:.88rem;color:#6b4c00}
.note.blue{background:var(--azul3);border-color:#cfe1f3;color:#1c4a7a}
.note.verde{background:var(--verde-bg);border-color:#bfe3d2;color:#1c5a3c}
.muted{color:var(--gris)}
.up{color:var(--verde);font-weight:700}
.down{color:var(--rosa);font-weight:700}
ul.tight{margin:.3em 0 1em;padding-left:1.2em}
ul.tight li{margin:.25em 0}
code{background:#eef2f6;padding:1px 5px;border-radius:5px;font-size:.85em}
footer{color:var(--gris);font-size:.82rem;text-align:center;padding:16px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:.8rem;color:var(--gris);margin-top:8px}
.legend span{display:flex;align-items:center;gap:6px}
.sw{width:12px;height:12px;border-radius:3px;display:inline-block;flex:0 0 auto}
.steps{counter-reset:s;list-style:none;padding-left:0}
.steps li{counter-increment:s;margin:.4em 0;padding-left:2em;position:relative}
.steps li:before{content:counter(s);position:absolute;left:0;top:0;background:var(--azul2);color:#fff;
  width:1.4em;height:1.4em;border-radius:50%;font-size:.8rem;font-weight:700;display:flex;align-items:center;justify-content:center}
.checks{display:flex;flex-wrap:wrap;gap:2px 4px;max-height:170px;overflow:auto;border:1px solid var(--linea);
  border-radius:10px;padding:9px 11px;background:#fbfdff}
.chk{display:inline-flex;align-items:center;gap:6px;margin:2px 12px 2px 0;font-size:.86rem;cursor:pointer}
.chk input{width:auto;margin:0}
.dec .q{font-weight:700;color:var(--azul);margin:0 0 .55em}
.dec .resp{background:var(--verde-bg);border:1px solid #bfe3d2;border-radius:10px;
  padding:11px 13px;font-size:.9rem;color:#1c5a3c;margin:.7em 0 .5em}
.dec .resp strong{color:#12492f}
.dec h3{font-size:1.05rem}
.dec h4.mini{font-size:.8rem;text-transform:uppercase;letter-spacing:.03em;color:var(--gris);margin:0 0 .3em}
.jumps{display:flex;flex-wrap:wrap;gap:8px;margin:.7em 0 0}
a.jump{display:inline-block;font-size:.82rem;font-weight:700;text-decoration:none;
  border:1px solid var(--linea);border-radius:8px;padding:5px 10px;background:#fbfdff;color:var(--azul2)}
a.jump:hover{background:var(--azul3)}
.dec .tbl-scroll{max-height:520px}
/* Pestañas y bloques de decisión */
nav.topnav a.act{background:var(--azul);color:#fff}
.panel{display:none}
.panel.on{display:block}
.lead{font-size:1rem;color:var(--gris);margin:0 0 1em;max-width:88ch}
.resp{background:var(--verde-bg);border:1px solid #bfe3d2;border-radius:10px;
  padding:11px 13px;font-size:.9rem;color:#1c5a3c;margin:.7em 0 .5em}
.resp strong{color:#12492f}
.q{font-weight:700;color:var(--azul);margin:0 0 .55em}
.mini{font-size:.8rem;font-weight:700;text-transform:uppercase;letter-spacing:.03em;color:var(--gris);margin:0 0 .4em}
tr.fila-hl{background:var(--azul3)}
tr.fila-hl td{font-weight:700;color:var(--azul)}
"""

JS = r"""
'use strict';
const D = window.__DATA__;
const $ = (s, r) => (r || document).querySelector(s);
const $$ = (s, r) => Array.prototype.slice.call((r || document).querySelectorAll(s));
const nf = (x, d) => (x === null || x === undefined || isNaN(x)) ? 'n/d'
  : x.toLocaleString('es-BO', {minimumFractionDigits: d || 0, maximumFractionDigits: d || 0});
const fmtInt = x => (x === null || x === undefined || isNaN(x)) ? 'n/d' : nf(Math.round(x), 0);
const fmtDec = x => (x === null || x === undefined || isNaN(x)) ? 'n/d' : nf(x, 1);
const pct = x => (x === null || x === undefined || isNaN(x)) ? 'n/d'
  : (x >= 0 ? '+' : '') + nf(x, 1) + ' %';
const sumNZ = a => a.reduce((s, v) => s + (v === null || v === undefined ? 0 : v), 0);
const countNZ = a => a.filter(v => v !== null && v !== undefined).length;
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

const PALETTE = ['#2b6cb0', '#22704a', '#9a6b00', '#6b3fa0', '#b03a5b', '#0f8b8d',
  '#c2410c', '#4a5568', '#7c3aed', '#0e7490', '#b45309', '#166534', '#9d174d', '#1e40af'];

/* -------------------- State -------------------- */
const state = {
  vis: {year: D.ultimoVisCompleto, month: 0, nat: 'Todas', region: 'Todas', periodo: 'p2225'},
  gas: {year: D.ultimoGasto, pais: 'Todos', region: 'Todas'},
  reg: {region: D.regiones[0]},
  sea: {vista: 'regiones', region: D.regiones[0], year: 'avg', cPaises: [], formato: 'lineas'},
  q1: {periodo: 'p2225', region: 'Todas'},
  q2: {region: 'Todas', pais: 'Todos', periodo: 'avg', cPaises: []},
  q3: {pais: ''},
};

/* Rango [añoIni, añoFin] de los tres periodos del selector de evolución.
   'completos' excluye los años parciales (p. ej. 2026); 'todos' es el histórico. */
function periodoRango(periodo, completos, todos){
  if (periodo === 'p2225') return [2022, 2025];
  if (periodo === 'last5'){ const c = completos.slice(-5); return [c[0], c[c.length - 1]]; }
  return [todos[0], todos[todos.length - 1]];
}
function yearsEnRango(years, rango){ return years.filter(y => y >= rango[0] && y <= rango[1]); }

/* -------------------- Alcance de llegadas -------------------- */
function scopeMarkets(){
  const {nat, region} = state.vis;
  if (nat !== 'Todas') return [nat];
  if (region !== 'Todas') return D.regionPaises[region] || [];
  return D.nacionalidades;
}
function scopeSeries(year){
  const {nat, region} = state.vis;
  const y = String(year);
  if (nat === 'Todas' && region === 'Todas') return (D.totalMonthly[y] || new Array(12).fill(null)).slice();
  if (nat !== 'Todas') return ((D.monthly[nat] || {})[y] || new Array(12).fill(null)).slice();
  return ((D.regionMonthly[region] || {})[y] || new Array(12).fill(null)).slice();
}
function scopeAnnual(year){
  const {nat, region} = state.vis;
  const y = String(year);
  if (nat === 'Todas' && region === 'Todas'){ const v = D.totalAnnual[y]; return v === undefined ? null : v; }
  if (nat !== 'Todas'){ const v = (D.annual[nat] || {})[y]; return v === undefined ? null : v; }
  const v = (D.regionAnnual[region] || {})[y]; return v === undefined ? null : v;
}
function visValue(year, month){
  const s = scopeSeries(year);
  if (month === 0) return countNZ(s) === 0 ? null : sumNZ(s);
  const v = s[month - 1];
  return v === undefined ? null : v;
}
function visHasData(nat, year, month){
  const s = (D.monthly[nat] || {})[String(year)] || new Array(12).fill(null);
  if (month === 0) return countNZ(s) > 0;
  return s[month - 1] !== null && s[month - 1] !== undefined;
}
function mercadosCount(year, month){
  return scopeMarkets().filter(n => visHasData(n, year, month)).length;
}
function scopePrelim(year, month){
  const {nat, region} = state.vis;
  if (nat === 'Todas' && region === 'Todas'){
    const p = (D.totalPrelim || {})[String(year)] || [];
    return month === 0 ? p.length > 0 : p.indexOf(month) >= 0;
  }
  return scopeMarkets().some(n => {
    const p = (D.prelimM[n] || {})[String(year)] || [];
    return month === 0 ? p.length > 0 : p.indexOf(month) >= 0;
  });
}
function annualVal(nat, year){ const v = (D.annual[nat] || {})[String(year)]; return v === undefined ? null : v; }
function visSeriesVal(nat, year, month){
  const s = ((D.monthly[nat] || {})[String(year)] || []);
  const v = s[month - 1];
  return v === undefined ? null : v;
}

/* -------------------- Helpers estacionalidad -------------------- */
function share12(arr){
  if (!arr) return null;
  const tot = sumNZ(arr);
  if (!tot) return null;
  return arr.map(v => (v === null || v === undefined ? 0 : v) / tot * 100);
}
function avgShare(arrs){
  const valid = arrs.map(share12).filter(a => a !== null);
  if (!valid.length) return null;
  return Array.from({length: 12}, (_, i) => valid.reduce((s, a) => s + a[i], 0) / valid.length);
}
function peakMonth(sh){ return sh.indexOf(Math.max.apply(null, sh)); }
function hhiNorm(sh){
  const p = sh.map(v => v / 100);
  const hhi = p.reduce((s, x) => s + x * x, 0);
  return (hhi - 1 / 12) / (1 - 1 / 12);
}
function vMag(v){
  if (v < 0.05) return 'despreciable';
  if (v < 0.10) return 'muy débil';
  if (v < 0.20) return 'débil';
  if (v < 0.30) return 'moderada';
  return 'fuerte';
}

/* ==================== SVG charts ==================== */
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent){
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function svgRoot(host, w, h){
  host.innerHTML = '';
  const s = el('svg', {viewBox: '0 0 ' + w + ' ' + h, role: 'img'});
  host.appendChild(s);
  return s;
}
function vBar(host, rows, opts){
  opts = opts || {};
  const w = 860, h = opts.height || 300, m = {t: 18, r: 12, b: opts.padBottom || 52, l: 62};
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const s = svgRoot(host, w, h);
  const max = Math.max.apply(null, rows.map(r => (r.value || 0)).concat([1]));
  const bw = iw / rows.length;
  for (let i = 0; i <= 4; i++){
    const y = m.t + ih - (ih * i / 4);
    el('line', {x1: m.l, y1: y, x2: m.l + iw, y2: y, stroke: '#e7edf5', 'stroke-width': 1}, s);
    const t = el('text', {x: m.l - 8, y: y + 4, 'text-anchor': 'end', 'font-size': 11, fill: '#7b8a99'}, s);
    t.textContent = fmtInt(max * i / 4);
  }
  rows.forEach((r, i) => {
    const x = m.l + i * bw + bw * 0.16, bwid = bw * 0.68;
    if (r.value !== null && r.value !== undefined){
      const bh = ih * (r.value / max), y = m.t + ih - bh;
      const rect = el('rect', {x: x, y: y, width: bwid, height: Math.max(bh, 1), rx: 3,
        fill: r.hl ? '#b03a5b' : (opts.color || '#2b6cb0')}, s);
      const ti = el('title', {}, rect);
      ti.textContent = r.label + ': ' + fmtInt(r.value) + (r.mark === 'p' ? ' (preliminar)' : '');
    } else {
      const t = el('text', {x: x + bwid / 2, y: m.t + ih - 4, 'text-anchor': 'middle', 'font-size': 10, fill: '#b6c2cf'}, s);
      t.textContent = 'n/d';
    }
    const lbl = el('text', {x: x + bwid / 2, y: m.t + ih + 14, 'text-anchor': 'middle',
      'font-size': 10.5, fill: r.hl ? '#b03a5b' : '#42526b', 'font-weight': r.hl ? 700 : 400}, s);
    lbl.textContent = r.label;
    if (rows.length > 14){ lbl.setAttribute('transform', 'rotate(-30 ' + (x + bwid / 2) + ' ' + (m.t + ih + 14) + ')'); }
  });
}
function hBar(host, rows, opts){
  opts = opts || {};
  const w = 860, rowH = 26, m = {t: 8, r: 70, b: 8, l: opts.labelW || 150};
  const h = m.t + m.b + Math.max(rows.length, 1) * rowH;
  const iw = w - m.l - m.r;
  const s = svgRoot(host, w, h);
  if (!rows.length){
    const t = el('text', {x: w / 2, y: 30, 'text-anchor': 'middle', 'font-size': 12, fill: '#7b8a99'}, s);
    t.textContent = 'Sin datos para el alcance seleccionado.';
    return;
  }
  const max = Math.max.apply(null, rows.map(r => r.value || 0).concat([1]));
  rows.forEach((r, i) => {
    const y = m.t + i * rowH;
    const bw = iw * ((r.value || 0) / max);
    const rect = el('rect', {x: m.l, y: y + 4, width: Math.max(bw, 1), height: rowH - 10, rx: 3,
      fill: opts.color || '#22704a'}, s);
    const ti = el('title', {}, rect); ti.textContent = r.label + ': ' + fmtInt(r.value) + (r.mark === 'p' ? ' (preliminar)' : '');
    const l = el('text', {x: m.l - 8, y: y + rowH / 2 + 3, 'text-anchor': 'end', 'font-size': 11.5, fill: '#42526b'}, s);
    l.textContent = r.label;
    const v = el('text', {x: m.l + bw + 7, y: y + rowH / 2 + 3, 'font-size': 11.5, fill: '#42526b', 'font-weight': 600}, s);
    v.textContent = fmtInt(r.value) + (r.mark === 'p' ? ' (p)' : '');
  });
}
function pieChart(host, rows, opts){
  opts = opts || {};
  const w = 860, h = opts.height || 300, cx = 170, cy = h / 2, r = 116;
  const s = svgRoot(host, w, h);
  const total = rows.reduce(function(a, d){ return a + Math.max(0, d.value || 0); }, 0);
  const colors = opts.colors || ['#12345a', '#2b6cb0', '#4a9b77', '#e3a43b', '#b03a5b', '#8b9bab'];
  if (!(total > 0)){
    const t = el('text', {x: w / 2, y: 32, 'text-anchor': 'middle', 'font-size': 12, fill: '#7b8a99'}, s);
    t.textContent = 'Sin datos para el alcance seleccionado.';
    return;
  }
  let angle = -Math.PI / 2;
  rows.filter(function(d){ return d.value > 0; }).forEach(function(d, i){
    const fraction = d.value / total, next = angle + fraction * Math.PI * 2;
    const x1 = cx + r * Math.cos(angle), y1 = cy + r * Math.sin(angle);
    const x2 = cx + r * Math.cos(next), y2 = cy + r * Math.sin(next);
    const large = fraction > 0.5 ? 1 : 0;
    const path = el('path', {d: 'M ' + cx + ' ' + cy + ' L ' + x1 + ' ' + y1 +
      ' A ' + r + ' ' + r + ' 0 ' + large + ' 1 ' + x2 + ' ' + y2 + ' Z',
      fill: colors[i % colors.length], stroke: '#fff', 'stroke-width': 2}, s);
    const pctVal = d.pct;
    const title = el('title', {}, path);
    title.textContent = d.label + ': ' + fmtInt(d.value) + ' llegadas · ' + pctVal.toFixed(1) + ' %';
    angle = next;
  });
  const legend = el('g', {}, s);
  rows.forEach(function(d, i){
    const y = 48 + i * 34;
    el('rect', {x: 350, y: y - 11, width: 14, height: 14, rx: 3, fill: colors[i % colors.length]}, legend);
    const label = el('text', {x: 374, y: y, 'font-size': 12, fill: '#42526b'}, legend);
    label.textContent = d.label + ' · ' + d.pct.toFixed(1) + ' %';
    const value = el('text', {x: 820, y: y, 'text-anchor': 'end', 'font-size': 11, fill: '#7b8a99'}, legend);
    value.textContent = fmtInt(d.value);
  });
}
function groupedBar(host, cats, series, opts){
  opts = opts || {};
  const w = 860, h = opts.height || 300, m = {t: 18, r: 12, b: 44, l: 62};
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const s = svgRoot(host, w, h);
  const all = series.reduce((a, se) => a.concat(se.values), []);
  const max = Math.max.apply(null, all.filter(v => v !== null).concat([1]));
  const fmt = opts.pct ? (x => x.toFixed(1) + ' %') : fmtInt;
  for (let i = 0; i <= 4; i++){
    const y = m.t + ih - (ih * i / 4);
    el('line', {x1: m.l, y1: y, x2: m.l + iw, y2: y, stroke: '#e7edf5', 'stroke-width': 1}, s);
    const t = el('text', {x: m.l - 8, y: y + 4, 'text-anchor': 'end', 'font-size': 11, fill: '#7b8a99'}, s);
    t.textContent = fmt(max * i / 4);
  }
  const groupW = iw / cats.length, barW = groupW * 0.72 / series.length;
  cats.forEach((c, ci) => {
    const gx = m.l + ci * groupW + groupW * 0.14;
    series.forEach((se, si) => {
      const v = se.values[ci];
      if (v === null || v === undefined) return;
      const bh = ih * (v / max), x = gx + si * barW;
      const rect = el('rect', {x: x, y: m.t + ih - bh, width: Math.max(barW * 0.86, 1),
        height: Math.max(bh, 1), rx: 2, fill: se.color}, s);
      const ti = el('title', {}, rect); ti.textContent = se.name + ' · ' + c + ': ' + fmt(v);
    });
    const lx = m.l + ci * groupW + groupW / 2, ly = m.t + ih + 16;
    const lbl = el('text', {x: lx, y: ly, 'text-anchor': 'middle', 'font-size': 10.5, fill: '#42526b'}, s);
    lbl.textContent = c;
    if (cats.length > 14 || opts.rotate){ lbl.setAttribute('transform', 'rotate(-25 ' + lx + ' ' + ly + ')'); }
  });
}

/* Gráfico de LÍNEAS genérico: labels=[...], series=[{name,values,color}] */
function lineChart(host, labels, series, opts){
  opts = opts || {};
  const w = 860, h = opts.height || 320, m = {t: 18, r: 16, b: opts.padBottom || 44, l: 54};
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const s = svgRoot(host, w, h);
  const allv = series.reduce((a, se) => a.concat(se.values), [])
    .filter(v => v !== null && v !== undefined && !isNaN(v));
  const max = Math.max.apply(null, allv.concat([opts.minTop || 5]));
  const n = labels.length;
  const xAt = i => m.l + (n <= 1 ? iw / 2 : iw * i / (n - 1));
  const yAt = v => m.t + ih - ih * (v / max);
  const fmtv = opts.fmt || (x => x.toFixed(1) + '%');
  for (let i = 0; i <= 4; i++){
    const y = m.t + ih - (ih * i / 4);
    el('line', {x1: m.l, y1: y, x2: m.l + iw, y2: y, stroke: '#e7edf5', 'stroke-width': 1}, s);
    const t = el('text', {x: m.l - 8, y: y + 4, 'text-anchor': 'end', 'font-size': 11, fill: '#7b8a99'}, s);
    t.textContent = fmtv(max * i / 4);
  }
  labels.forEach((lb, i) => {
    const t = el('text', {x: xAt(i), y: m.t + ih + 16, 'text-anchor': 'middle', 'font-size': 10.5, fill: '#42526b'}, s);
    t.textContent = lb;
  });
  series.forEach(se => {
    const pts = [];
    se.values.forEach((v, i) => { if (v !== null && v !== undefined && !isNaN(v)) pts.push([xAt(i), yAt(v), v, i]); });
    if (!pts.length) return;
    el('polyline', {points: pts.map(p => p[0] + ',' + p[1]).join(' '), fill: 'none',
      stroke: se.color, 'stroke-width': 2, 'stroke-linejoin': 'round'}, s);
    pts.forEach(p => {
      const c = el('circle', {cx: p[0], cy: p[1], r: 3, fill: se.color}, s);
      const ti = el('title', {}, c);
      ti.textContent = se.name + ' · ' + labels[p[3]] + ': ' + fmtv(p[2]);
    });
  });
}

/* Barra vertical en % (para la estacionalidad de UNA región) */
function pctBar(host, labels, values, opts){
  opts = opts || {};
  const w = 860, h = opts.height || 320, m = {t: 18, r: 12, b: 44, l: 54};
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const s = svgRoot(host, w, h);
  const vals = (values || []).filter(v => v !== null && v !== undefined && !isNaN(v));
  const max = Math.max.apply(null, vals.concat([1]));
  for (let i = 0; i <= 4; i++){
    const y = m.t + ih - (ih * i / 4);
    el('line', {x1: m.l, y1: y, x2: m.l + iw, y2: y, stroke: '#e7edf5', 'stroke-width': 1}, s);
    const t = el('text', {x: m.l - 8, y: y + 4, 'text-anchor': 'end', 'font-size': 11, fill: '#7b8a99'}, s);
    t.textContent = (max * i / 4).toFixed(1) + ' %';
  }
  const bw = iw / labels.length;
  labels.forEach((lb, i) => {
    const v = (values || [])[i];
    const x = m.l + i * bw + bw * 0.18, bwid = bw * 0.64;
    if (v === null || v === undefined || isNaN(v)){
      const t = el('text', {x: x + bwid / 2, y: m.t + ih - 4, 'text-anchor': 'middle', 'font-size': 10, fill: '#b6c2cf'}, s);
      t.textContent = 'n/d';
    } else {
      const bh = ih * (v / max), y = m.t + ih - bh;
      const isPeak = v === max;
      const rect = el('rect', {x: x, y: y, width: Math.max(bwid, 1), height: Math.max(bh, 1), rx: 3,
        fill: isPeak ? '#b03a5b' : (opts.color || '#2b6cb0')}, s);
      const ti = el('title', {}, rect); ti.textContent = lb + ': ' + v.toFixed(1) + '%';
      const vt = el('text', {x: x + bwid / 2, y: y - 4, 'text-anchor': 'middle', 'font-size': 9.5,
        fill: isPeak ? '#b03a5b' : '#42526b', 'font-weight': isPeak ? 700 : 400}, s);
      vt.textContent = v.toFixed(1) + '%';
    }
    const lbl = el('text', {x: x + bwid / 2, y: m.t + ih + 15, 'text-anchor': 'middle', 'font-size': 10.5, fill: '#42526b'}, s);
    lbl.textContent = lb;
  });
}

/* ==================== Render llegadas ==================== */
function renderVis(){
  const {year, month, nat, region} = state.vis;
  const cur = visValue(year, month);
  const prev = visValue(year - 1, month);
  const varp = (cur !== null && prev !== null && prev > 0) ? ((cur - prev) / prev * 100) : null;
  const nmerc = mercadosCount(year, month);
  const pre = scopePrelim(year, month);
  const alcanceNac = nat === 'Todas' ? (region === 'Todas' ? 'Todos los mercados' : 'mercados de ' + region) : nat;
  const alcance = alcanceNac + ' · ' + (month === 0 ? 'todo ' + year : (D.meses[month - 1] + ' ' + year));

  $('#k1v').textContent = fmtInt(cur);
  $('#k1s').textContent = alcance + (pre ? ' · incluye (p) preliminares' : '');
  $('#k2v').textContent = pct(varp);
  const k2s = $('#k2s');
  k2s.textContent = (prev === null || prev === undefined)
    ? 'Sin periodo comparable anterior (' + (month === 0 ? String(year - 1) : D.meses[month - 1] + ' ' + (year - 1)) + ')'
    : 'vs ' + (month === 0 ? String(year - 1) : D.meses[month - 1] + ' ' + (year - 1)) + ' (' + fmtInt(prev) + ')';
  k2s.className = 'sub ' + (varp === null ? '' : (varp >= 0 ? 'up' : 'down'));
  $('#k3v').textContent = fmtInt(nmerc);
  $('#k3s').textContent = (nat === 'Todas' && region === 'Todas')
    ? 'Mercados extranjeros con dato (excluye Bolivia y residual)'
    : (nat === 'Todas' ? 'Países de la región con dato' : 'Mercado seleccionado');
  $('#visAlcance').textContent = 'Alcance: visitantes extranjeros (excluye nacionales) · ' + alcance + '.';
  $('#visNotaP').style.display = pre ? 'block' : 'none';

  const rangoV = periodoRango(state.vis.periodo, D.aniosVisCompletos, D.aniosVis);
  const aniosEv = yearsEnRango(D.aniosVis, rangoV);
  const rowsE = aniosEv.map(y => {
    const v = scopeAnnual(y);
    return {label: String(y), value: v, mark: scopePrelim(y, 0) ? 'p' : null, hl: y === year};
  });
  vBar($('#chEvolucion'), rowsE, {color: '#2b6cb0', height: 300});
  $('#evolNota').textContent = (nat === 'Todas' && region === 'Todas' ? 'Total de visitantes extranjeros'
      : (nat !== 'Todas' ? nat : region)) +
    ' por año · periodo ' + rangoV[0] + '–' + rangoV[1] +
    ' · por nacionalidad. 2026 es un año parcial (ene–jul) y no cuenta como año completo. El filtro de mes no aplica a este gráfico.';

  const rank = scopeMarkets().map(n => ({
    label: n, value: (month === 0 ? annualVal(n, year) : visSeriesVal(n, year, month)),
    mark: ((D.prelimM[n] || {})[String(year)] || []).length ? 'p' : null,
  })).filter(r => r.value !== null && r.value > 0).sort((a, b) => b.value - a.value).slice(0, 15);
  hBar($('#chRanking'), rank, {color: '#22704a', labelW: 150});
  $('#rankNota').textContent = 'Top 15 nacionalidades extranjeras por llegadas · ' +
    (month === 0 ? 'todo ' + year : D.meses[month - 1] + ' ' + year) +
    (region === 'Todas' ? '' : ' · ' + region) + '. Excluye Bolivia (nacionales) y el residual.';

  // Estacionalidad mensual (SECUNDARIA): año seleccionado vs año anterior
  const curS = scopeSeries(year), prevS = scopeSeries(year - 1);
  groupedBar($('#chEstacional'), D.meses.map(m => m.slice(0, 3)), [
    {name: String(year - 1), values: prevS, color: '#c3d3e3'},
    {name: String(year), values: curS, color: '#2b6cb0'},
  ], {height: 300});
  $('#estNota').textContent = 'Comparación secundaria · llegadas extranjeras por mes: ' + year + ' vs ' + (year - 1) +
    (nat === 'Todas' && region === 'Todas' ? ' (total)' : (nat !== 'Todas' ? ' · ' + nat : ' · ' + region)) +
    '. Para comparar regiones en una misma escala, usa el Módulo 2.';

  renderVisTable();
}

let visSort = {col: 'valor', dir: -1};
function renderVisTable(){
  const {year, month} = state.vis;
  const q = ($('#visSearch').value || '').toLowerCase();
  const markets = scopeMarkets();
  let rows = [];
  if (month === 0){
    markets.forEach(n => {
      const v = annualVal(n, year);
      if (v !== null && v !== undefined && v > 0) rows.push({nat: n, per: String(year), valor: v,
        prelim: ((D.prelimM[n] || {})[String(year)] || []).length > 0});
    });
  } else {
    markets.forEach(n => {
      const v = visSeriesVal(n, year, month);
      if (v !== null && v !== undefined) rows.push({nat: n, per: D.meses[month - 1] + ' ' + year, valor: v,
        prelim: ((D.prelimM[n] || {})[String(year)] || []).indexOf(month) >= 0});
    });
  }
  if (q) rows = rows.filter(r => r.nat.toLowerCase().indexOf(q) >= 0);
  rows.sort((a, b) => {
    const c = visSort.col;
    if (c === 'nat' || c === 'per') return a[c].localeCompare(b[c]) * visSort.dir;
    return (a[c] - b[c]) * visSort.dir;
  });
  $('#visTbody').innerHTML = rows.map(r => '<tr><td>' + esc(r.nat) + '</td><td>' + esc(r.per) +
    '</td><td class="num">' + fmtInt(r.valor) + '</td><td>' +
    (r.prelim ? '<span class="badge p">(p) preliminar</span>' : '<span class="badge n">observado</span>') +
    '</td></tr>').join('');
  $('#visCount').textContent = rows.length + ' fila(s)';
}

/* ==================== Render estacionalidad regional ==================== */
function renderRegiones(){
  const years = D.aniosVis.filter(y => y >= D.statsRegion.desde && y <= D.statsRegion.hasta);
  const series = years.map((y, i) => ({
    name: String(y), color: PALETTE[i % PALETTE.length],
    values: D.regiones.map(r => { const v = (D.regionAnnual[r] || {})[String(y)]; return v === undefined ? null : v; }),
  }));
  groupedBar($('#chRegEvol'), D.regiones, series, {height: 320, rotate: true});
  $('#regEvolLegend').innerHTML = years.map((y, i) =>
    '<span><span class="sw" style="background:' + PALETTE[i % PALETTE.length] + '"></span>' + y + '</span>').join('');

  const reg = state.reg.region;
  const rows = (D.regionPaises[reg] || []).map(p => {
    let tot = 0, any = false;
    years.forEach(y => { const v = (D.annual[p] || {})[String(y)]; if (v !== undefined && v !== null){ tot += v; any = true; } });
    return {label: p, value: any ? tot : null};
  }).filter(r => r.value !== null).sort((a, b) => b.value - a.value);
  hBar($('#chRegRanking'), rows, {color: '#22704a', labelW: 150});
  $('#regRankNota').textContent = 'Llegadas extranjeras acumuladas 2022–2025 por nacionalidad de ' + reg + '.';
}

function regionShare(reg, period, years){
  const get = y => (D.regionMonthly[reg] || {})[String(y)];
  return period === 'avg' ? avgShare(years.map(get)) : share12(get(period));
}
function paisShare(p, period, years){
  const get = y => (D.monthly[p] || {})[String(y)];
  return period === 'avg' ? avgShare(years.map(get)) : share12(get(period));
}

/* Vista C: casillas de países de la región, máximo 5 seleccionados. */
function buildPaisChecks(){
  const reg = state.sea.region;
  const years = D.aniosVis.filter(y => y >= D.statsRegion.desde && y <= D.statsRegion.hasta);
  const list = (D.regionPaises[reg] || []).map(p => {
    let t = 0, any = false;
    years.forEach(y => { const v = (D.annual[p] || {})[String(y)]; if (v !== undefined && v !== null){ t += v; any = true; } });
    return {p: p, t: t, any: any};
  }).filter(o => o.any).sort((a, b) => b.t - a.t);
  const validos = list.map(o => o.p);
  const previos = state.sea.cPaises.length;
  state.sea.cPaises = state.sea.cPaises.filter(p => validos.indexOf(p) >= 0).slice(0, 5);
  if (!state.sea.cPaises.length){
    state.sea.cPaises = validos.slice(0, 5);
    $('#seaPaisHint').textContent = 'Se preseleccionan los 5 países con más llegadas de ' + reg +
      '. Puedes quitar o añadir países (máx. 5) con las casillas.';
  } else {
    $('#seaPaisHint').textContent = 'Selecciona hasta 5 países de ' + reg + '.' + (previos > 5 ? ' Se mantienen los 5 primeros.' : '');
  }
  $('#seaPaisChecks').innerHTML = list.map(o => {
    const on = state.sea.cPaises.indexOf(o.p) >= 0;
    return '<label class="chk"><input type="checkbox" value="' + esc(o.p) + '"' + (on ? ' checked' : '') +
      '> ' + esc(o.p) + ' <span class="muted">' + fmtInt(o.t) + '</span></label>';
  }).join('');
  $('#seaPaisCount').textContent = state.sea.cPaises.length + ' de 5 seleccionados';
  $$('#seaPaisChecks input').forEach(cb => cb.addEventListener('change', onPaisCheck));
}
function onPaisCheck(e){
  const p = e.target.value;
  if (e.target.checked){
    if (state.sea.cPaises.length >= 5){
      e.target.checked = false;
      $('#seaPaisCount').textContent = 'Máximo 5 países. Quita uno para añadir otro.';
      return;
    }
    state.sea.cPaises.push(p);
  } else {
    state.sea.cPaises = state.sea.cPaises.filter(x => x !== p);
  }
  renderSeason();
}

function renderSeason(){
  const vista = state.sea.vista;
  // Mantiene el selector de región sincronizado cuando la región cambia desde
  // los filtros del Módulo 1 o de la sección de Decisiones comerciales.
  if ($('#seaRegion') && $('#seaRegion').value !== state.sea.region){
    $('#seaRegion').value = state.sea.region;
    if ($('#seaRegion').value !== state.sea.region) state.sea.region = $('#seaRegion').value;
  }
  const years = D.aniosVis.filter(y => y >= D.statsRegion.desde && y <= D.statsRegion.hasta);
  const period = state.sea.year;
  const per = period === 'avg' ? ('promedio ' + years[0] + '–' + years[years.length - 1]) : ('año ' + period);

  $('#seaRegionWrap').style.display = (vista === 'regiones') ? 'none' : '';
  $('#seaFormatoWrap').style.display = (vista === 'paises') ? '' : 'none';
  $('#seaPaisBox').style.display = (vista === 'paises') ? '' : 'none';

  const labels = D.meses.map(m => m.slice(0, 3));
  let series = [];
  if (vista === 'regiones'){
    series = D.regiones.map((r, i) => ({name: r, color: PALETTE[i % PALETTE.length],
      values: regionShare(r, period, years)}));
    lineChart($('#chSeason'), labels, series, {height: 340});
    $('#seasonTitle').textContent = 'Vista A · Comparar regiones (una línea por región, máximo 7)';
    $('#seasonNote').textContent = 'Cada línea muestra qué porcentaje aporta cada mes al total anual de su propia región (' +
      per + '). Las 7 regiones comparten la misma escala vertical.';
    $('#seasonPeakNote').textContent = '';
  } else if (vista === 'region'){
    const reg = state.sea.region;
    const sh = regionShare(reg, period, years);
    pctBar($('#chSeason'), labels, sh, {color: '#2b6cb0'});
    series = [{name: reg, color: '#2b6cb0', values: sh}];
    $('#seasonTitle').textContent = 'Vista B · Analizar una región: ' + reg;
    $('#seasonNote').textContent = 'Barras: porcentaje que aporta cada mes al total anual de ' + reg + ' (' + per + '). El mes de mayor afluencia se resalta.';
    const pi = sh ? peakMonth(sh) : null;
    $('#seasonPeakNote').textContent = sh
      ? ('Mes de mayor afluencia en ' + reg + ': ' + D.meses[pi] + ' (' + sh[pi].toFixed(1) + '%). Periodo: ' + per + '.')
      : 'Sin datos para el periodo elegido.';
  } else {
    const reg = state.sea.region;
    buildPaisChecks();
    const sel = state.sea.cPaises;
    series = sel.map((p, i) => ({name: p, color: PALETTE[i % PALETTE.length],
      values: paisShare(p, period, years)}));
    if (state.sea.formato === 'barras'){
      groupedBar($('#chSeason'), labels, series, {height: 360, pct: true});
    } else {
      lineChart($('#chSeason'), labels, series, {height: 340});
    }
    $('#seasonTitle').textContent = 'Vista C · Comparar países de ' + reg + ' (máximo 5)';
    $('#seasonNote').textContent = 'Se dibujan ' + sel.length + ' país(es) de ' + reg + ' (' + per +
      '), con el mismo criterio porcentual: cada mes sobre el total anual del propio país. Nunca más de 5 series simultáneas.';
    $('#seasonPeakNote').textContent = sel.length ? '' : 'Selecciona al menos un país.';
  }

  $('#seasonLegend').innerHTML = series.map(se =>
    '<span><span class="sw" style="background:' + se.color + '"></span>' + esc(se.name) + '</span>').join('');

  // Tabla complementaria: meses (filas) x regiones (columnas), en % (todas las regiones)
  const shares = D.regiones.map(r => regionShare(r, period, years));
  const head = '<tr><th>Mes</th>' + D.regiones.map(r => '<th class="num">' + esc(r) + '</th>').join('') + '</tr>';
  const body = D.meses.map((mes, i) => '<tr><td>' + mes + '</td>' +
    shares.map(sh => sh ? '<td class="num">' + sh[i].toFixed(1) + '</td>' : '<td class="num zero">n/d</td>').join('') +
    '</tr>').join('');
  const peak = '<tr><td><strong>Mes pico</strong></td>' + shares.map(sh => {
    if (!sh) return '<td>n/d</td>';
    const pi = peakMonth(sh);
    return '<td><strong>' + D.meses[pi] + '</strong> (' + sh[pi].toFixed(1) + '%)</td>';
  }).join('') + '</tr>';
  const hhiRow = '<tr><td><strong>HHI norm.</strong></td>' + shares.map(sh =>
    sh ? '<td class="num">' + hhiNorm(sh).toFixed(3) + '</td>' : '<td class="num zero">n/d</td>').join('') + '</tr>';
  $('#seasonTable').innerHTML = '<thead>' + head + '</thead><tbody>' + body + peak + hhiRow + '</tbody>';
  $('#seasonRangeNote').textContent = 'Periodo: ' + per + '.';
}

/* ==================== Render pruebas estadísticas ==================== */
function renderStats(){
  const s = D.statsRegion;
  $('#stN').textContent = fmtInt(s.n);
  $('#stReg').textContent = s.regiones.length;
  $('#stGl').textContent = s.gl;
  $('#stP').textContent = (s.p > 0) ? s.p.toExponential(2) : '< 1e-300';
  $('#stV').innerHTML = s.cramerV.toFixed(4) + ' <small>(' + vMag(s.cramerV) + ')</small>';
  $('#stMinEsp').textContent = fmtDec(s.minEsperado);
  $('#stPctEsp').textContent = s.pctEsperadoLt5.toFixed(1) + ' %';
  const head = '<tr><th>Región</th>' + s.meses.map(m => '<th class="num">' + m.slice(0, 3) + '</th>').join('') +
    '<th class="num">Total</th></tr>';
  const body = s.regiones.map(r => '<tr><td>' + esc(r) + '</td>' +
    s.tabla[r].map(v => '<td class="num">' + fmtInt(v) + '</td>').join('') +
    '<td class="num">' + fmtInt(s.filaTotales[r]) + '</td></tr>').join('');
  const foot = '<tr><td><strong>Total</strong></td>' +
    s.colTotales.map(v => '<td class="num"><strong>' + fmtInt(v) + '</strong></td>').join('') +
    '<td class="num"><strong>' + fmtInt(s.n) + '</strong></td></tr>';
  $('#stTable').innerHTML = '<thead>' + head + '</thead><tbody>' + body + foot + '</tbody>';

  const e = D.statsEuropa;
  $('#euChi').textContent = fmtInt(e.chi2);
  $('#euGl').textContent = e.gl;
  $('#euP').textContent = (e.p > 0) ? e.p.toExponential(2) : '< 1e-300';
  $('#euV').innerHTML = e.cramerV.toFixed(4) + ' <small>(' + vMag(e.cramerV) + ')</small>';
  $('#euN').textContent = fmtInt(e.n);
  $('#euMinEsp').textContent = fmtDec(e.minEsperado);
  $('#euPctEsp').textContent = e.pctEsperadoLt5.toFixed(1) + ' %';
  $('#euCob').textContent = e.cobertura.toFixed(1) + ' %';
  $('#euPaises').textContent = e.paises.join(', ');
  const chk = $('#euCheck');
  chk.textContent = e.coincide
    ? 'Coincide con el archivo guardado (' + e.archivo + '): χ²=' + fmtInt(e.chi2Guardado) +
      ', V=' + e.vGuardado.toFixed(4) + '.'
    : '⚠ DIFIERE del archivo guardado. Valores recomputados aquí: χ²=' + e.chi2.toFixed(3) +
      ', V=' + e.cramerV.toFixed(4) + '.';
  chk.className = e.coincide ? 'chart-note' : 'chart-note down';
}

/* ==================== Render inferencial complementario (Etapa 5) ==================== */
function renderInferencial(){
  const I = D.infer;
  const host = $('#inferContenido');
  if (!I || !I.disponible){
    host.innerHTML = '<div class="note">Sección no disponible todavía. Ejecuta ' +
      '<code>scripts/10_analisis_inferencial_complementario.py</code> y regenera el dashboard.</div>';
    $('#inferArchivos').textContent = '';
    return;
  }
  $('#inferArchivos').textContent = 'Archivos de detalle en resultados/: ' + I.archivos.join(', ') + '.';
  const tabTend = (res, etq) =>
    '<div class="statrow">' +
    '<div class="stat"><div class="l">Mercados evaluados</div><div class="v">' + res.evaluados + '</div></div>' +
    '<div class="stat"><div class="l">Creciente p_BH&lt;0,05 (nominal)</div><div class="v">' + res.nominalArriba + '</div></div>' +
    '<div class="stat"><div class="l">Decreciente p_BH&lt;0,05 (nominal)</div><div class="v">' + res.nominalAbajo + '</div></div>' +
    '<div class="stat"><div class="l">Creciente concluyente (Hamed-Rao)</div><div class="v">' + res.conclArriba + '</div></div>' +
    '<div class="stat"><div class="l">Decreciente concluyente (Hamed-Rao)</div><div class="v">' + res.conclAbajo + '</div></div>' +
    '</div>';
  const topTend = res => res.top.length
    ? '<ul class="tight">' + res.top.map(o => '<li>' + esc(o.pais) + ': Sen ' + fmtDec(o.sen) +
        ' llegadas/año · p_BH=' + (o.pBH === null ? 'n/d' : o.pBH.toExponential(2)) +
        (o.concl ? ' · <span class="badge obs">concluyente</span>' : ' · <span class="badge hip">no concluyente (autocorr.)</span>') +
        '</li>').join('') + '</ul>'
    : '<p class="muted">Sin tendencias crecientes nominalmente significativas.</p>';

  const homRows = I.homogeneidad.map(h => '<tr><td>' + esc(h.ambito) + '</td><td>' + esc(h.paises) + '</td>' +
    '<td class="num">' + (h.chi2 === null ? 'n/d' : fmtDec(h.chi2)) + '</td>' +
    '<td class="num">' + (h.gl === null ? 'n/d' : h.gl) + '</td>' +
    '<td class="num">' + (h.p === null || h.p === 0 ? (h.p === 0 ? '<1e-300' : 'n/d') : h.p.toExponential(2)) + '</td>' +
    '<td class="num">' + (h.V === null ? 'n/d' : fmtDec(h.V)) + '</td>' +
    '<td>' + esc(h.decision) + '</td></tr>').join('');

  const decRows = I.detalle.map(d => '<tr><td>' + esc(d.pais) + '</td><td>' + esc(d.region) + '</td>' +
    '<td class="num">' + fmtInt(d.v2025) + '</td>' +
    '<td class="num">' + (d.var === null ? 'n/d' : d.var.toFixed(1) + ' %') + '</td>' +
    '<td>' + esc(d.tendencia) + '</td><td>' + esc(d.patron) + '</td>' +
    '<td><span class="badge ' + (d.prioridad === 'Alta' ? 'obs' : (d.prioridad === 'Media' ? 'p' : 'n')) + '">' + esc(d.prioridad) + '</span></td></tr>').join('');

  host.innerHTML =
    '<div class="note blue"><strong>Alcance.</strong> Estas pruebas no repiten ni reemplazan el análisis descriptivo, la estacionalidad ni los chi-cuadrado ya validados. La prioridad sugerida es una ' +
    '<em>selección de mercados potenciales a investigar</em>, no una predicción de ventas ni una medida de demanda de SUP.</div>' +
    '<div class="card" style="margin-top:16px"><h3>Tendencias por país · pre-pandemia 2008-2019</h3>' + tabTend(I.pre, 'pre') + topTend(I.pre) + '</div>' +
    '<div class="card" style="margin-top:16px"><h3>Tendencias por país · histórico 2008-2025</h3>' + tabTend(I.hist, 'hist') + topTend(I.hist) + '</div>' +
    '<div class="card" style="margin-top:16px"><h3>Homogeneidad estacional entre países candidatos (2022-2025)</h3>' +
      '<p class="hint">Chi-cuadrado de homogeneidad país × mes por región. Una diferencia de calendario NO implica mayor demanda de SUP.</p>' +
      '<div class="tbl-scroll"><table class="matrix"><thead><tr><th>Ámbito</th><th>Países</th><th class="num">χ²</th><th class="num">gl</th><th class="num">p</th><th class="num">V Cramér</th><th>Decisión</th></tr></thead><tbody>' + homRows + '</tbody></table></div></div>' +
    '<div class="card" style="margin-top:16px"><h3>Tabla de apoyo a la decisión · 15 mercados prioritarios</h3>' +
      '<p class="hint">Prioridad por criterios explícitos V/C/T (volumen ≥ P75 · crecimiento 2024→2025 &gt; 0 · tendencia pre-pandemia concluyente). Ver limitaciones en el TXT.</p>' +
      '<div class="tbl-scroll"><table class="matrix"><thead><tr><th>País</th><th>Región</th><th class="num">Llegadas 2025</th><th class="num">Var. 2024→2025</th><th>Tendencia</th><th>Patrón estacional</th><th>Prioridad</th></tr></thead><tbody>' + decRows + '</tbody></table></div>' +
      '<p class="chart-note">Prioridades: ' + (I.prioridades.Alta || 0) + ' Alta · ' + (I.prioridades.Media || 0) + ' Media · ' + (I.prioridades.Baja || 0) + ' Baja (sobre 51 mercados).</p></div>';
}

/* ==================== Decisiones comerciales (5 módulos) ==================== */
/* ==================== Pestañas (navegación) ==================== */
function showTab(id){
  var target = id || location.hash || '#prioritarios';
  var el = document.querySelector(target);
  if (!el || !el.classList || !el.classList.contains('panel')){ target = '#prioritarios'; el = document.querySelector(target); }
  if (!el) return;
  $$('.panel').forEach(function(p){ p.classList.toggle('on', p === el); });
  $$('nav.topnav a').forEach(function(a){ a.classList.toggle('act', a.getAttribute('href') === target); });
  if (location.hash !== target && history.replaceState) history.replaceState(null, '', target);
  window.scrollTo(0, 0);
}
function wireTabs(){
  $$('a[href^="#"]').forEach(function(a){
    a.addEventListener('click', function(ev){
      var t = document.querySelector(a.getAttribute('href'));
      if (t && t.classList && t.classList.contains('panel')){ ev.preventDefault(); showTab(a.getAttribute('href')); }
    });
  });
  window.addEventListener('hashchange', function(){ showTab(location.hash); });
}

/* ==================== Comercial · indicadores reutilizados ==================== */
/* Reutiliza SOLO datos ya cargados: agregados de llegadas por región y por
   nacionalidad, estacionalidad mensual y las salidas ya calculadas de la Etapa 5
   (D.infer). No ejecuta pruebas estadísticas nuevas. */
const DEC_Y0 = 2022, DEC_Y1 = 2025;
function decNu(v){ return (v === undefined || v === null) ? null : v; }
function decRegAnual(reg, y){ return decNu((D.regionAnnual[reg] || {})[String(y)]); }
function decRegVol(reg, years){ return years.reduce(function(s, y){ return s + (decRegAnual(reg, y) || 0); }, 0); }
function decPaisAnual(p, y){ return decNu((D.annual[p] || {})[String(y)]); }
function decPaisVol(p, years){ return years.reduce(function(s, y){ return s + (decPaisAnual(p, y) || 0); }, 0); }
function decVar(a, b){ return (a !== null && b !== null && b > 0) ? (a - b) / b * 100 : null; }
function decBadge(prio){
  const cls = prio === 'Alta' ? 'obs' : (prio === 'Media' ? 'p' : 'n');
  return '<span class="badge ' + cls + '">' + esc(prio) + '</span>';
}
function decCrit(c){ const m = String(c || '').match(/criterios:\s*([VCT-]{1,3})/); return m ? m[1] : ''; }
function decTend(t){
  if (!t || t.sen === null) return '<span class="muted">no evaluable</span>';
  return esc(t.etq) + ' ' + (t.concl ? '<span class="badge obs">concluyente</span>' : '<span class="badge hip">no concluyente</span>') +
    '<br><small class="muted">Sen ' + fmtDec(t.sen) + '/año · IC95% [' + fmtInt(t.icInf) + ', ' + fmtInt(t.icSup) + '] · p_BH=' +
    (t.pBH === null ? 'n/d' : t.pBH.toExponential(2)) + ' · n=' + (t.n === null ? '—' : t.n) + '</small>';
}
function decVarCls(v){ return v === null ? '' : (v >= 0 ? 'up' : 'down'); }
function regionPaises(reg){ return (D.regionPaises[reg] || []).slice(); }
function fmtCompact(x){
  if (x === null || x === undefined || isNaN(x)) return 'n/d';
  if (Math.abs(x) >= 1e6) return (x / 1e6).toFixed(1) + ' M';
  if (Math.abs(x) >= 1e3) return Math.round(x / 1e3) + ' mil';
  return String(Math.round(x));
}
function pit(p){ return (p === null || p === undefined) ? 'n/d' : (p > 0 ? p.toExponential(2) : '< 1e-300'); }
function perEtiqueta(per){
  if (per === 'avg' || per === 'p2225') return 'promedio 2022–2025';
  if (per === 'hist') return 'promedio histórico ' + D.aniosVisCompletos[0] + '–' + D.aniosVisCompletos[D.aniosVisCompletos.length - 1];
  return 'año ' + per;
}
function perAnios(per){
  if (per === 'avg' || per === 'p2225') return [2022, 2023, 2024, 2025];
  if (per === 'hist') return D.aniosVisCompletos.slice();
  return [+per];
}
function rangoEtiqueta(per){
  const y = perAnios(per);
  return (y.length > 1) ? (y[0] + '–' + y[y.length - 1]) : String(y[0]);
}
/* % mensual del alcance elegido: país > región > total extranjero. */
function shareDe(pais, region, per, years){
  function arr(get){
    if (per === 'avg' || per === 'p2225' || per === 'hist') return avgShare(years.map(get));
    return share12(get(+per));
  }
  if (pais && pais !== 'Todos') return arr(function(y){ return (D.monthly[pais] || {})[String(y)]; });
  if (region && region !== 'Todas') return arr(function(y){ return (D.regionMonthly[region] || {})[String(y)]; });
  return arr(function(y){ return D.totalMonthly[String(y)]; });
}
function inferMercados(){ return (D.infer && D.infer.disponible) ? (D.infer.mercados || {}) : {}; }

/* ==================== Pregunta 1 · Mercados prioritarios ==================== */
function q1Render(){
  if (!$('#q1RegChart')) return;
  const years = perAnios(state.q1.periodo);
  const escala = D.regiones.reduce(function(s, r){ return s + decRegVol(r, years); }, 0);
  const ult = years[years.length - 1], prev = years.length > 1 ? years[years.length - 2] : null;
  let rows = D.regiones.map(function(r){
    const vol = decRegVol(r, years);
    const serie = years.map(function(y){ return decRegAnual(r, y); });
    const a = serie[0], b = serie[serie.length - 1];
    return {reg: r, vol: vol, serie: serie, share: escala ? vol / escala * 100 : null,
      varUlt: prev !== null ? decVar(decRegAnual(r, ult), decRegAnual(r, prev)) : null,
      varPer: (a !== null && b !== null && a > 0) ? (b - a) / a * 100 : null};
  }).filter(function(o){ return o.vol > 0; }).sort(function(a, b){ return b.vol - a.vol; });

  hBar($('#q1RegChart'), rows.map(function(o){ return {label: o.reg, value: o.vol}; }),
    {color: '#2b6cb0', labelW: 172});
  $('#q1RegNota').textContent = 'Llegadas de visitantes extranjeros (nacionalidad) agrupadas por región · ' +
    rangoEtiqueta(state.q1.periodo) + ' · unidad: llegadas registradas. ' +
    '«Var. último año» compara ' + ult + ' con ' + (prev === null ? ult - 1 : prev) +
    '; «Var. del periodo» compara el último con el primer año mostrado. Un solo año no prueba una tendencia.';

  const regSeries = D.regiones.map(function(r, i){
    return {name: r, color: PALETTE[i % PALETTE.length],
      values: years.map(function(y){ return decRegAnual(r, y); })};
  });
  lineChart($('#q1RegEvol'), years.map(String), regSeries,
    {height: 330, minTop: 1, padBottom: 40, fmt: function(x){ return fmtCompact(x); }});
  $('#q1EvolLegend').innerHTML = regSeries.map(function(se){
    return '<span><span class="sw" style="background:' + se.color + '"></span>' + esc(se.name) + '</span>';
  }).join('');
  $('#q1EvolNota').textContent = 'Cada color corresponde a una región, identificada en la leyenda. Eje vertical: llegadas de visitantes extranjeros por año; las 7 regiones comparten la misma escala. ' +
    (state.q1.periodo === 'hist' ? 'Se excluye 2026 porque sólo hay meses parciales.' : 'Vista de referencia 2022–2025.');

  const head = '<tr><th>Región de procedencia</th>' + years.map(function(y){ return '<th class="num">' + y + '</th>'; }).join('') +
    '<th class="num">Var. ' + ult + '</th><th class="num">Var. del periodo</th><th class="num">% del total</th></tr>';
  const body = rows.map(function(o){
    return '<tr><td>' + esc(o.reg) + '</td>' +
      o.serie.map(function(v){ return '<td class="num">' + (v === null ? '<span class="muted">n/d</span>' : fmtInt(v)) + '</td>'; }).join('') +
      '<td class="num ' + decVarCls(o.varUlt) + '">' + pct(o.varUlt) + '</td>' +
      '<td class="num ' + decVarCls(o.varPer) + '">' + pct(o.varPer) + '</td>' +
      '<td class="num">' + (o.share === null ? 'n/d' : o.share.toFixed(1) + ' %') + '</td></tr>';
  }).join('');
  $('#q1RegTable').innerHTML = '<thead>' + head + '</thead><tbody>' + body + '</tbody>';

  let t = '';
  if (rows.length){
    const top2 = rows.slice(0, 2);
    t = 'Por volumen, ' + top2.map(function(o){ return '<strong>' + esc(o.reg) + '</strong> (' + fmtInt(o.vol) + ')'; }).join(' y ') +
      ' concentran el ' + top2.reduce(function(s, o){ return s + (o.share || 0); }, 0).toFixed(1) + ' % de las llegadas de las 7 regiones.';
    const crecen = rows.filter(function(o){ return o.varUlt !== null && o.varUlt > 0; });
    t += crecen.length
      ? ' Crecen en ' + ult + ': ' + crecen.slice(0, 3).map(function(o){ return '<strong>' + esc(o.reg) + '</strong> (' + pct(o.varUlt) + ')'; }).join(', ') + '.'
      : ' Ninguna región creció en ' + ult + ' frente al año anterior.';
    const p25 = rows.length ? rows.slice().sort(function(a, b){ return a.vol - b.vol; })[Math.floor(rows.length * 0.25)].vol : null;
    const bajas = crecen.filter(function(o){ return p25 !== null && o.vol < p25; });
    if (bajas.length){
      t += ' Ojo: ' + bajas.map(function(o){ return esc(o.reg); }).join(' y ') +
        ' crecen sobre un volumen pequeño (' + bajas.map(function(o){ return fmtInt(o.vol); }).join(', ') +
        ' en el periodo), así que su tasa alta no basta para priorizar por sí sola.';
    }
  }
  $('#q1Resp').innerHTML = t || 'No hay datos para el periodo seleccionado.';
  $('#q1Warn').innerHTML = '<strong>Notas.</strong> Las llegadas registradas no equivalen necesariamente a personas únicas ' +
    '(<a href="#metodologia">Fuentes y metodología</a>). La categoría residual «Resto de países» no tiene región y no entra en esta ' +
    'comparación. Un mercado con más llegadas no implica, por sí solo, más interés por el SUP.';
}

function q1PaisesRender(){
  if (!$('#q1PaisBox')) return;
  const reg = state.q1.region;
  const box = $('#q1PaisBox'), prompt = $('#q1PaisPrompt');
  if (reg === 'Todas'){
    box.style.display = 'none';
    prompt.style.display = '';
    $('#q1PaisResp').innerHTML = '';
    return;
  }
  prompt.style.display = 'none';
  box.style.display = '';
  const years = perAnios(state.q1.periodo);
  const ult = years[years.length - 1], prev = years.length > 1 ? years[years.length - 2] : null;
  const tot = decRegVol(reg, years);
  const M = inferMercados();
  const rows = regionPaises(reg).map(function(p){
    const vol = decPaisVol(p, years);
    const shAv = avgShare(years.map(function(y){ return (D.monthly[p] || {})[String(y)]; }));
    const m = M[p] || {};
    return {p: p, vol: vol, share: tot ? vol / tot * 100 : null,
      varUlt: prev !== null ? decVar(decPaisAnual(p, ult), decPaisAnual(p, prev)) : null,
      pico: shAv ? peakMonth(shAv) : null, hhi: shAv ? hhiNorm(shAv) : null,
      pre: ((m.tend || {}).pre) || null, prioridad: m.prioridad || null, criterios: m.criterios || ''};
  }).filter(function(o){ return o.vol > 0; }).sort(function(a, b){ return b.vol - a.vol; });

  const top5 = rows.slice(0, 5);
  hBar($('#q1PaisChart'), top5.map(function(o){ return {label: o.p, value: o.vol}; }),
    {color: '#22704a', labelW: 150});
  const topTotal = top5.reduce(function(s, o){ return s + o.vol; }, 0);
  const resto = Math.max(0, tot - topTotal);
  const reparto = top5.map(function(o){
    return {label: o.p, value: o.vol, pct: tot > 0 ? o.vol / tot * 100 : 0};
  });
  reparto.push({label: 'Resto de la región', value: resto, pct: tot > 0 ? resto / tot * 100 : 0});
  pieChart($('#q1PaisPie'), reparto);
  $('#q1PaisNota').textContent = 'Cinco países con mayor volumen acumulado de ' + reg + ' en ' + rangoEtiqueta(state.q1.periodo) +
    ' (criterio: escala del mercado, no tasa de crecimiento). Unidad: llegadas registradas.';
  $('#q1PaisPieNota').textContent = 'Participación de los cinco principales países y del resto de la región sobre el total de llegadas regionales';

  const head = '<tr><th>País (nacionalidad)</th><th class="num">Volumen ' + rangoEtiqueta(state.q1.periodo) +
    '</th><th class="num">% de ' + esc(reg) + '</th><th class="num">Var. ' + ult +
    '</th><th>Tendencia histórica (2008–2019)</th><th>Estacionalidad</th><th>Prioridad (V/C/T)</th></tr>';
  const body = rows.map(function(o){
    return '<tr><td>' + esc(o.p) + '</td>' +
      '<td class="num">' + fmtInt(o.vol) + '</td>' +
      '<td class="num">' + (o.share === null ? 'n/d' : o.share.toFixed(1) + ' %') + '</td>' +
      '<td class="num ' + decVarCls(o.varUlt) + '">' + pct(o.varUlt) + '</td>' +
      '<td>' + decTend(o.pre) + '</td>' +
      '<td>' + (o.pico === null ? '<span class="muted">n/d</span>'
        : 'pico ' + esc(D.meses[o.pico]) + '<br><small class="muted">HHI ' + o.hhi.toFixed(3) + '</small>') + '</td>' +
      '<td>' + (o.prioridad ? decBadge(o.prioridad) + '<br><small class="muted">criterios ' +
        esc(decCrit(o.criterios)) + '</small>' : '<span class="muted">—</span>') + '</td></tr>';
  }).join('');
  $('#q1PaisTable').innerHTML = '<thead>' + head + '</thead><tbody>' + body + '</tbody>';

  let t = '';
  if (rows.length){
    const top = top5.slice(0, 2);
    t = 'En <strong>' + esc(reg) + '</strong> lideran ' + top.map(function(o){ return '<strong>' + esc(o.p) + '</strong> (' + fmtInt(o.vol) + ')'; }).join(' y ') +
      ', que aportan el ' + top.reduce(function(s, o){ return s + (o.share || 0); }, 0).toFixed(1) + ' % de la región.';
    const altas = rows.filter(function(o){ return o.prioridad === 'Alta'; });
    if (altas.length){
      t += ' Con prioridad Alta quedan ' + altas.length + ': ' + altas.slice(0, 5).map(function(o){ return '<strong>' + esc(o.p) + '</strong>'; }).join(', ') +
        (altas.length > 5 ? '…' : '') + '.';
    }
    const soloTasa = rows.filter(function(o){ return o.varUlt !== null && o.varUlt > 0 && o.vol < (top5[4] ? top5[4].vol : 0); });
    if (soloTasa.length){
      t += ' ' + soloTasa.length + ' mercado(s) crecieron en ' + ult + ' pero con volumen menor al de los cinco primeros: ' +
        'conviene mirarlos como oportunidad secundaria, no como prioridad.';
    }
  }
  $('#q1PaisResp').innerHTML = t || 'Sin países con volumen en el periodo seleccionado.';
}

/* ==================== Pregunta 2 · Calendario de campañas ==================== */
function q2BuildChecks(reg, years){
  const list = regionPaises(reg).map(function(p){ return {p: p, v: decPaisVol(p, years)}; })
    .filter(function(o){ return o.v > 0; }).sort(function(a, b){ return b.v - a.v; }).slice(0, 20);
  const validos = list.map(function(o){ return o.p; });
  state.q2.cPaises = state.q2.cPaises.filter(function(p){ return validos.indexOf(p) >= 0; }).slice(0, 5);
  if (!state.q2.cPaises.length) state.q2.cPaises = validos.slice(0, 5);
  $('#q2PaisChecks').innerHTML = list.map(function(o){
    const on = state.q2.cPaises.indexOf(o.p) >= 0;
    return '<label class="chk"><input type="checkbox" value="' + esc(o.p) + '"' + (on ? ' checked' : '') +
      '> ' + esc(o.p) + ' <span class="muted">' + fmtInt(o.v) + '</span></label>';
  }).join('');
  $('#q2PaisCount').textContent = state.q2.cPaises.length + ' de 5 seleccionados para comparar.';
  $$('#q2PaisChecks input').forEach(function(cb){ cb.addEventListener('change', onQ2PaisCheck); });
}
function onQ2PaisCheck(e){
  const p = e.target.value;
  if (e.target.checked){
    if (state.q2.cPaises.length >= 5){
      e.target.checked = false;
      $('#q2PaisCount').textContent = 'Máximo 5 países. Quita uno para añadir otro.';
      return;
    }
    state.q2.cPaises.push(p);
  } else {
    state.q2.cPaises = state.q2.cPaises.filter(function(x){ return x !== p; });
  }
  q2Render();
}
function q2Render(){
  if (!$('#q2SeasonChart')) return;
  const per = state.q2.periodo;
  const years = perAnios(per);
  const reg = state.q2.region, pais = state.q2.pais;
  const labels = D.meses.map(function(m){ return m.slice(0, 3); });
  const comparando = (reg !== 'Todas' && pais === 'Todos');

  if (comparando){ $('#q2PaisBox').style.display = ''; q2BuildChecks(reg, years); }
  else { $('#q2PaisBox').style.display = 'none'; state.q2.cPaises = []; }

  let primArr, primLabel;
  if (pais !== 'Todos'){ primArr = shareDe(pais, reg, per, years); primLabel = pais; }
  else if (reg !== 'Todas'){ primArr = shareDe('Todos', reg, per, years); primLabel = reg; }
  else { primArr = shareDe('Todos', 'Todas', per, years); primLabel = 'todos los mercados extranjeros'; }

  let series = [];
  if (comparando && state.q2.cPaises.length && primArr){
    series = state.q2.cPaises.map(function(p, i){
      return {name: p, color: PALETTE[i % PALETTE.length], values: shareDe(p, reg, per, years)};
    });
    lineChart($('#q2SeasonChart'), labels, series, {height: 350});
  } else if (primArr){
    pctBar($('#q2SeasonChart'), labels, primArr, {color: '#2b6cb0'});
    series = [{name: primLabel, color: '#2b6cb0', values: primArr}];
  } else {
    $('#q2SeasonChart').innerHTML = '<p class="muted">Sin datos para el alcance seleccionado.</p>';
  }
  $('#q2Legend').innerHTML = series.map(function(se){
    return '<span><span class="sw" style="background:' + se.color + '"></span>' + esc(se.name) + '</span>';
  }).join('');
  $('#q2SeasonNota').textContent = (comparando && state.q2.cPaises.length ? 'Las líneas comparan el patrón mensual de los países seleccionados. ' : '') +
    'Porcentaje que aporta cada mes al total anual de ' + primLabel + ' · ' + perEtiqueta(per) +
    '. Cada serie se calcula sobre su propio total anual (no sobre el total nacional). Eje vertical: % del año.';

  const meses = primArr ? primArr.map(function(v, i){ return [i, v]; }).sort(function(a, b){ return b[1] - a[1]; }) : [];
  if (!meses.length){
    $('#q2Fuertes').innerHTML = '<p class="muted">Sin datos.</p>';
    $('#q2Flojos').innerHTML = '';
    $('#q2Resp').innerHTML = 'No hay datos suficientes para el alcance seleccionado.';
    $('#q2Warn').innerHTML = '';
    q2StatsRender();
    return;
  }
  $('#q2Fuertes').innerHTML = '<ul class="tight">' + meses.slice(0, 3).map(function(o){
    return '<li><strong>' + D.meses[o[0]] + '</strong> — ' + o[1].toFixed(1) + ' % del año</li>';
  }).join('') + '</ul>';
  $('#q2Flojos').innerHTML = '<ul class="tight">' + meses.slice(-3).reverse().map(function(o){
    return '<li><strong>' + D.meses[o[0]] + '</strong> — ' + o[1].toFixed(1) + ' %</li>';
  }).join('') + '</ul>';

  const yearsRef = [2022, 2023, 2024, 2025];
  const refArrs = yearsRef.map(function(y){
    if (pais !== 'Todos') return (D.monthly[pais] || {})[String(y)];
    if (reg !== 'Todas') return (D.regionMonthly[reg] || {})[String(y)];
    return D.totalMonthly[String(y)];
  });
  const refProfile = avgShare(refArrs);
  const picoRef = refProfile ? peakMonth(refProfile) : meses[0][0];
  const picos = refArrs.map(function(a){ const sh = share12(a); return sh ? peakMonth(sh) : null; })
    .filter(function(v){ return v !== null; });
  const coinciden = picos.filter(function(p){ return p === picoRef; }).length;
  const hhi = hhiNorm(primArr);

  let t = 'Para <strong>' + esc(primLabel) + '</strong> (' + perEtiqueta(per) + '), conviene intensificar la promoción durante los meses de mayor afluencia observada: <strong>' +
    meses.slice(0, 3).map(function(o){ return D.meses[o[0]] + ' (' + o[1].toFixed(1) + ' %)'; }).join(', ') +
    '</strong>. La concentración mensual descriptiva es ' + (hhi < 0.15 ? 'baja' : (hhi < 0.35 ? 'moderada' : 'alta')) +
    ' (HHI norm. = ' + hhi.toFixed(3) + '); el pico del perfil de referencia 2022–2025 (' + D.meses[picoRef] +
    ') aparece en ' + coinciden + ' de ' + picos.length + ' años de ese periodo.';
  if (comparando && state.q2.cPaises.length){
    const pks = state.q2.cPaises.map(function(p){
      const sh = shareDe(p, reg, per, years);
      return sh ? (p + ': ' + D.meses[peakMonth(sh)]) : null;
    }).filter(function(x){ return x; });
    t += ' Mes pico por país comparado: ' + pks.join(' · ') + '. Úsalo para revisar posibles matices; el contraste estadístico no demuestra por sí solo que deban separarse campañas.';
  }
  const vRegion = D.statsRegion.cramerV;
  if (vRegion < 0.05){
    t += ' Recomendación inicial: usar un calendario general; ' + (D.statsRegion.p < 0.05
      ? 'el contraste detecta asociación estadística, pero su magnitud es despreciable'
      : 'el contraste no detecta diferencias regionales relevantes y su magnitud observada es despreciable') +
      ' (V = ' + vRegion.toFixed(4) + ').';
  } else if (vRegion >= 0.10){
    t += ' Recomendación inicial: valorar calendarios por región, porque el tamaño de las diferencias regionales no es trivial (V = ' + vRegion.toFixed(4) + ').';
  } else {
    t += ' Recomendación inicial: partir de un calendario general y revisar matices regionales (V = ' + vRegion.toFixed(4) + '); la magnitud observada es pequeña.';
  }
  const hCountries = (D.infer && D.infer.disponible) ? (D.infer.homogeneidad || []).filter(function(h){ return h.tipo === 'region'; }) : [];
  const countryDiff = hCountries.some(function(h){ return h.pBH !== null && h.pBH < 0.05; });
  if (countryDiff){
    t += ' La prueba entre países dentro de algunas regiones encuentra diferencias en los perfiles agregados, pero agrupa 2022–2025 y no establece que se repitan año a año; por ahora no justifica segmentar campañas por país sin validar esa estabilidad.';
  } else if (hCountries.length){
    t += ' Las pruebas entre países no detectan diferencias tras el ajuste BH; esto no demuestra igualdad absoluta ni justifica descartar matices descriptivos.';
  } else {
    t += ' No están disponibles las pruebas existentes entre países por región; no se puede concluir si se justifican ajustes por país.';
  }
  t += ' La promoción puede intensificarse en los meses pico observados; el plazo para iniciar campañas es una decisión de planificación, no un resultado estadístico, porque no hay datos de búsquedas, consultas ni reservas.';
  $('#q2Resp').innerHTML = t;
  $('#q2Warn').innerHTML = '<strong>Limitaciones.</strong> Que las llegadas a Bolivia se concentren en ciertos meses no ' +
    'demuestra que la demanda de SUP en Copacabana siga ese calendario ni cuál es el momento óptimo de reserva: para eso harían ' +
    'falta datos de búsquedas, consultas, reservas o campañas reales. Significación estadística ≠ importancia comercial ' +
    '(<a href="#metodologia">Fuentes y metodología</a>).';

  q2StatsRender();
}

function q2StatsRender(){
  const H = (D.infer && D.infer.disponible) ? (D.infer.homogeneidad || []) : [];
  const sr = D.statsRegion;
  const years = [2022, 2023, 2024, 2025];
  const profiles = D.regiones.map(function(r){
    const arrs = years.map(function(y){ return (D.regionMonthly[r] || {})[String(y)]; });
    const avg = avgShare(arrs);
    const peaks = arrs.map(function(a){ const sh = share12(a); return sh ? peakMonth(sh) : null; }).filter(function(x){ return x !== null; });
    const peak = peakMonth(avg);
    const repeats = peaks.filter(function(x){ return x === peak; }).length;
    const top3 = avg.map(function(v, i){ return {i: i, v: v}; }).sort(function(a, b){ return b.v - a.v; }).slice(0, 3);
    return {region: r, avg: avg, peak: peak, repeats: repeats, nYears: peaks.length, hhi: hhiNorm(avg), top3: top3};
  });
  const rowsA = profiles.map(function(o){
    return '<tr><td>' + esc(o.region) + '</td><td>' + D.meses[o.peak] + ' (' + o.avg[o.peak].toFixed(1) + ' %)</td>' +
      '<td>' + o.top3.map(function(m){ return D.meses[m.i] + ' ' + m.v.toFixed(1) + '%'; }).join(' · ') + '</td>' +
      '<td class="num">' + o.hhi.toFixed(3) + '</td><td class="num">' + o.repeats + ' de ' + o.nYears + '</td></tr>';
  }).join('');
  const evaluables = H.filter(function(h){ return h.tipo === 'region' && h.pBH !== null; });
  const rowsC = H.filter(function(h){ return h.tipo === 'region'; }).map(function(h){
    const sig = h.pBH !== null && h.pBH < 0.05;
    return '<tr><td>' + esc(h.ambito) + '</td><td>' + esc(h.paises) + ' (' + h.nPaises + ')</td>' +
      '<td class="num">' + (h.pBH === null ? 'n/d' : pit(h.pBH)) + '</td><td class="num">' +
      (h.V === null ? 'n/d' : fmtDec(h.V) + ' (' + vMag(h.V) + ')') + '</td><td>' +
      (h.pBH === null ? 'No evaluable: menos de 2 países con volumen suficiente.' :
        (sig ? 'Diferencias detectadas; revisar magnitud y estabilidad.' : 'No se detectan diferencias; no prueba igualdad absoluta.')) + '</td></tr>';
  }).join('');
  const sigBH = evaluables.filter(function(h){ return h.pBH < 0.05; }).length;
  const regionTestText = sr.p < 0.05
    ? 'Asociación estadísticamente significativa'
    : 'No se detectan diferencias; esto no demuestra igualdad absoluta';
  const global = '<div class="stat"><div class="l">B · Diferencias entre regiones · 2022–2025</div>' +
    '<div class="v" style="font-size:.95rem">χ² = ' + fmtInt(sr.chi2) + ' · gl = ' + sr.gl + ' · p = ' + pit(sr.p) +
    '</div><div class="sub muted">' + regionTestText + '. V de Cramér = ' + sr.cramerV.toFixed(4) + ' (' + vMag(sr.cramerV) +
    '); ' + (sr.cramerV < 0.05 ? 'magnitud despreciable, no basta para justificar calendarios regionales separados.' : 'valorar la magnitud antes de separar calendarios.') + '</div></div>';
  $('#q2Stats').innerHTML =
    '<div class="card"><h4 class="mini">A · Estacionalidad dentro de cada región (descriptiva)</h4>' +
    '<p class="hint">Se muestran los perfiles mensuales promedio y cuántos años (de 2022–2025) repiten el mes pico. No hay una prueba inferencial existente contra un reparto mensual uniforme; estos datos describen concentración, no demuestran estacionalidad estadística.</p>' +
    '<div class="tbl-scroll"><table class="matrix"><thead><tr><th>Región</th><th>Mes pico promedio</th><th>Tres meses con mayor concentración</th><th class="num">HHI norm.</th><th class="num">Repite el pico</th></tr></thead><tbody>' + rowsA + '</tbody></table></div></div>' +
    '<div class="card" style="margin-top:12px">' + global + '</div>' +
    '<div class="card" style="margin-top:12px"><h4 class="mini">C · Diferencias entre países de una misma región</h4>' +
    '<p class="hint">Chi-cuadrado de homogeneidad en los tres principales países por región con al menos 5.000 llegadas acumuladas en 2022–2025; p ajustado Benjamini–Hochberg. ' +
    sigBH + ' de ' + evaluables.length + ' regiones evaluables muestran diferencias. La prueba agrupa el periodo completo y no verifica estabilidad año a año.</p>' +
    '<div class="tbl-scroll"><table class="matrix"><thead><tr><th>Región</th><th>Países incluidos</th><th class="num">p ajustado</th><th class="num">V de Cramér</th><th>Lectura</th></tr></thead><tbody>' +
    (rowsC || '<tr><td colspan="5">Etapa 5 no disponible.</td></tr>') + '</tbody></table></div></div>';
}

/* ==================== Pregunta 3 · Evolución del gasto por país ==================== */
function q3Render(){
  if (!$('#q3GastoChart')) return;
  const pais = state.q3.pais;
  const years = D.aniosGasto;
  const serie = D.gastoPais[pais] || {};
  const tendencias = (D.gastoTendencias || {})[pais] || {};
  const pre = tendencias.pre || {}, hist = tendencias.hist || {};
  const ultimo = String(D.ultimoGasto), anterior = String(D.ultimoGasto - 1);
  const valorUlt = serie[ultimo], valorAnt = serie[anterior];
  const varUlt = valorUlt !== null && valorUlt !== undefined && valorAnt > 0
    ? (valorUlt - valorAnt) / valorAnt * 100 : null;
  const cuartiles = D.gastoCuartiles || {};
  const nivel = valorUlt >= cuartiles.p75 ? 'alto (cuartil superior)' :
    (valorUlt <= cuartiles.p25 ? 'bajo (cuartil inferior)' : 'intermedio');
  const qTxt = 'En ' + ultimo + ', el gasto agregado de este país se ubica en el nivel ' + nivel +
    ' frente a los ' + cuartiles.n + ' países con dato; P25 = ' + fmtDec(cuartiles.p25) +
    ' y P75 = ' + fmtDec(cuartiles.p75) + ' M USD.';

  lineChart($('#q3GastoTotalChart'), years.map(String), [{name: 'Total receptor Bolivia', color: '#12345a',
    values: years.map(function(y){ const v = D.gastoTotal[String(y)]; return v === undefined ? null : v; })}],
    {height: 290, minTop: 5, fmt: function(x){ return fmtDec(x) + ' M'; }});
  $('#q3GastoTotalNota').textContent = 'Gasto turístico receptor total anual · ' + years[0] + '–' + years[years.length - 1] +
    ' · millones de USD. El total incluye las categorías necesarias de la fuente; no se suma con sus componentes.';
  lineChart($('#q3GastoChart'), years.map(String), [{name: pais, color: '#2b6cb0',
    values: years.map(function(y){ const v = serie[String(y)]; return v === undefined ? null : v; })}],
    {height: 340, minTop: 5, fmt: function(x){ return fmtDec(x) + ' M'; }});
  $('#q3GastoNota').textContent = pais + ' · gasto agregado por país de residencia · ' + years[0] + '–' + years[years.length - 1] +
    ' · millones de USD. Los datos 2023–2025 son preliminares (p).';

  const rows = years.map(function(y, i){
    const v = serie[String(y)], pv = i ? serie[String(years[i - 1])] : null;
    const yoy = v !== null && v !== undefined && pv !== null && pv !== undefined && pv > 0 ? (v - pv) / pv * 100 : null;
    const prelim = !!((D.prelimG[pais] || {})[String(y)]);
    return '<tr><td>' + y + '</td><td class="num">' + (v === undefined || v === null ? 'n/d' : fmtDec(v)) + '</td>' +
      '<td class="num ' + decVarCls(yoy) + '">' + pct(yoy) + '</td><td>' +
      (prelim ? '<span class="badge p">(p) preliminar</span>' : '<span class="badge n">observado</span>') + '</td></tr>';
  }).join('');
  $('#q3GastoTable').innerHTML = '<thead><tr><th>Año</th><th class="num">Gasto (M USD)</th><th class="num">Variación anual</th><th>Dato</th></tr></thead><tbody>' + rows + '</tbody>';

  const llegadaMercado = (D.infer && D.infer.mercados && D.infer.mercados[pais]) || null;
  const llegadaTend = llegadaMercado && llegadaMercado.tend ? llegadaMercado.tend : {};
  const llegadaPre = llegadaTend.pre || {}, llegadaHist = llegadaTend.hist || {};
  const trendCard = function(r, label, unidad, autocorrLabel){
    const evaluable = r && r.n >= 8 && r.sen !== null && r.sen !== undefined;
    if (!evaluable) return '<div class="stat"><div class="l">' + label + '</div><div class="v">No evaluable</div><div class="sub muted">No hay serie o resultado estadístico disponible para este país.</div></div>';
    const concl = r.concl === true;
    const direccion = concl ? (r.sen > 0 ? 'Creciente concluyente' : (r.sen < 0 ? 'Decreciente concluyente' : 'Sin dirección concluyente')) : 'Tendencia no concluyente';
    const ci = r.icInf === null || r.icInf === undefined || r.icSup === null || r.icSup === undefined
      ? 'IC95% no disponible' : 'IC95% Sen [' + fmtDec(r.icInf) + ', ' + fmtDec(r.icSup) + '] ' + unidad;
    const p = r.pBHHR !== undefined ? r.pBHHR : r.pBH;
    const pLabel = r.pBHHR !== undefined ? 'p BH-Hamed–Rao' : 'p BH';
    const rho = r.rho1 !== undefined ? r.rho1 : r.autocorr;
    return '<div class="stat"><div class="l">' + label + '</div><div class="v" style="font-size:1rem">' + direccion +
      '</div><div class="sub muted">Pendiente de Sen = ' + fmtDec(r.sen) + ' ' + unidad + ' · ' + ci + '<br>' + pLabel + ' = ' + pit(p) +
      ' · n = ' + r.n + (rho === undefined ? '' : ' · autocorrelación = ' + fmtDec(rho)) +
      '<br>' + (concl ? 'Evidencia estadística concluyente según el resultado ajustado disponible.' : 'No asignar crecimiento ni contracción confirmados.') + '</div></div>';
  };
  $('#q3Tendencias').innerHTML = '<div class="statrow">' +
    trendCard(llegadaPre, 'Llegadas por nacionalidad · pre-pandemia 2008–2019', 'llegadas/año') +
    trendCard(llegadaHist, 'Llegadas por nacionalidad · histórico 2008–2025', 'llegadas/año') +
    trendCard(pre, 'Gasto por residencia · pre-pandemia 2008–2019', 'M USD/año') +
    trendCard(hist, 'Gasto por residencia · histórico 2008–2025', 'M USD/año') + '</div>';

  const llegadaDisponible = llegadaHist && llegadaHist.n >= 8 && llegadaHist.sen !== undefined;
  const gastoDisponible = hist && hist.n >= 8 && hist.sen !== undefined;
  const dirLlegada = llegadaDisponible && llegadaHist.concl === true && llegadaHist.sen !== 0
    ? (llegadaHist.sen > 0 ? 'creciente' : 'decreciente') : null;
  const dirGasto = gastoDisponible && hist.concl === true && hist.sen !== 0
    ? (hist.sen > 0 ? 'creciente' : 'decreciente') : null;
  const ambasConcl = dirLlegada !== null && dirGasto !== null;
  const scenarioRows = [
    {l: 'Llegadas crecientes + gasto creciente', i: 'Investigar el mercado.'},
    {l: 'Llegadas crecientes + gasto decreciente', i: 'Investigar la divergencia.'},
    {l: 'Llegadas decrecientes + gasto decreciente', i: 'Señal de cautela.'},
    {l: 'Llegadas decrecientes + gasto creciente', i: 'Investigar la divergencia.'},
  ];
  const activeLabel = ambasConcl ? 'Llegadas ' + dirLlegada + ' + gasto ' + dirGasto : '';
  const scenarioTable = '<thead><tr><th>Escenario</th><th>Lectura orientativa</th><th>Resultado para este país</th></tr></thead><tbody>' +
    scenarioRows.map(function(row){
      const active = ambasConcl && row.l === activeLabel;
      const result = ambasConcl ? (active ? '<strong>Señales concluyentes</strong>' : 'No corresponde a las tendencias observadas') :
        'Tendencia no concluyente';
      return '<tr' + (active ? ' style="background:#ebf3fb"' : '') + '><td>' + row.l + '</td><td>' + row.i + '</td><td>' + result + '</td></tr>';
    }).join('') + '</tbody>';
  $('#q3Escenarios').innerHTML = scenarioTable;
  const faltaLlegadas = !llegadaDisponible ? 'No hay una serie de llegadas por nacionalidad con el mismo nombre de país.' :
    (dirLlegada ? 'Llegadas: ' + dirLlegada + ' concluyente.' : 'Llegadas: Tendencia no concluyente.');
  const faltaGasto = !gastoDisponible ? 'No hay resultado de tendencia de gasto disponible.' :
    (dirGasto ? 'Gasto: ' + dirGasto + ' concluyente.' : 'Gasto: Tendencia no concluyente.');
  $('#q3ComparacionNota').innerHTML = '<strong>Resultado histórico 2008–2025.</strong> ' + faltaLlegadas + ' ' + faltaGasto +
    ' La comparación pre-pandemia (2008–2019) se muestra arriba como sensibilidad. La tendencia histórica incluye la ruptura de 2020–2021; el gasto de 2023–2025 es preliminar (p).';

  const resumenLlegadas = !llegadaDisponible ? 'Llegadas por nacionalidad: sin resultado comparable disponible.' :
    (dirLlegada ? 'Llegadas por nacionalidad: tendencia ' + dirLlegada + ' concluyente.' : 'Llegadas por nacionalidad: Tendencia no concluyente.');
  const resumenGasto = !gastoDisponible ? 'Gasto por residencia: sin resultado disponible.' :
    (dirGasto ? 'Gasto por residencia: tendencia ' + dirGasto + ' concluyente.' : 'Gasto por residencia: Tendencia no concluyente.');
  let conclusion = resumenLlegadas + ' ' + resumenGasto + ' ';
  if (ambasConcl){
    if (dirLlegada === 'creciente' && dirGasto === 'creciente') conclusion = 'Ambas señales son crecientes y concluyentes: investigar este mercado con validación comercial adicional.';
    else if (dirLlegada === 'decreciente' && dirGasto === 'decreciente') conclusion = 'Ambas señales son decrecientes y concluyentes: tratar como señal de cautela y revisar el contexto antes de priorizar investigación.';
    else conclusion = 'Las tendencias concluyentes divergen: investigar por qué difieren la señal de llegadas y la del gasto antes de priorizar el mercado.';
  } else {
    if (dirLlegada && !dirGasto){
      conclusion = 'Las llegadas muestran una tendencia ' + dirLlegada + ' concluyente, pero el gasto no; considerar investigar esta señal de forma exploratoria sin clasificar el escenario como confirmado.';
    } else if (!dirLlegada && dirGasto){
      conclusion = 'El gasto muestra una tendencia ' + dirGasto + ' concluyente, pero las llegadas no; investigar esta divergencia de forma exploratoria sin clasificar el escenario como confirmado.';
    } else {
      conclusion = 'Al menos una tendencia no es concluyente; no se asigna un escenario confirmado de crecimiento o contracción. Considerar el mercado para investigación exploratoria y mejorar la evidencia antes de priorizarlo.';
    }
  }
  $('#q3Conclusion').innerHTML = '<strong>Recomendación:</strong> ' + conclusion +
    ' Esta lectura sirve para priorizar investigación comercial; no demuestra rentabilidad ni demanda de SUP.';
  $('#q3Limit').innerHTML = '<strong>Alcance.</strong> Llegadas por nacionalidad y gasto agregado por residencia son señales complementarias y no registros vinculados de una misma población. No se calcula gasto por visitante ni se infiere gasto individual, disposición a pagar o demanda de SUP. La relación observada no demuestra causalidad; 2020–2021 puede afectar la tendencia de largo plazo.';
}

/* ==================== Render lectura comercial ==================== */
function renderInterpretacion(){
  const years = D.aniosVis.filter(y => y >= D.statsRegion.desde && y <= D.statsRegion.hasta);
  const vol = D.regiones.map(r => {
    let t = 0; years.forEach(y => { const v = (D.regionAnnual[r] || {})[String(y)]; if (v) t += v; });
    return {r: r, t: t};
  }).filter(o => o.t > 0).sort((a, b) => b.t - a.t);
  $('#interpVol').innerHTML = '<p class="muted"><span class="badge obs">observado</span> 2022–2025.</p><ul class="tight">' +
    vol.map((o, i) => '<li>' + (i + 1) + '. <strong>' + esc(o.r) + '</strong>: ' + fmtInt(o.t) + ' llegadas acumuladas.</li>').join('') +
    '</ul>';

  const conc = D.regiones.map(r => {
    const arr = Array.from({length: 12}, (_, i) => years.reduce((s, y) => {
      const a = (D.regionMonthly[r] || {})[String(y)];
      return s + ((a && a[i] != null) ? a[i] : 0);
    }, 0));
    const sh = share12(arr);
    return {r: r, hhi: sh ? hhiNorm(sh) : null, pico: sh ? peakMonth(sh) : null, sh: sh};
  }).filter(o => o.hhi !== null).sort((a, b) => b.hhi - a.hhi);
  $('#interpConc').innerHTML = '<p class="muted"><span class="badge obs">observado</span> HHI normalizado (0 = uniforme, 1 = un solo mes), 2022–2025.</p><ul class="tight">' +
    conc.slice(0, 4).map(o => '<li><strong>' + esc(o.r) + '</strong>: HHI ' + o.hhi.toFixed(3) +
      ' · mes pico ' + D.meses[o.pico] + ' (' + o.sh[o.pico].toFixed(1) + '%).</li>').join('') + '</ul>';

  const dest = D.regiones.map(r => {
    let best = null;
    (D.regionPaises[r] || []).forEach(p => {
      let t = 0; years.forEach(y => { const v = (D.annual[p] || {})[String(y)]; if (v) t += v; });
      if (t > 0 && (!best || t > best.t)) best = {p: p, t: t};
    });
    return {r: r, best: best};
  }).filter(o => o.best);
  $('#interpDest').innerHTML = '<p class="muted"><span class="badge obs">observado</span> país de mayor volumen dentro de cada región (2022–2025).</p><ul class="tight">' +
    dest.map(o => '<li><strong>' + esc(o.r) + '</strong>: ' + esc(o.best.p) + ' (' + fmtInt(o.best.t) + ').</li>').join('') + '</ul>';

  $('#interpMeses').innerHTML = '<p class="muted"><span class="badge hip">hipótesis</span> a partir del mes de mayor afluencia observado por región.</p><ul class="tight">' +
    conc.map(o => '<li>' + esc(o.r) + ': probar campañas en torno a <strong>' + D.meses[o.pico] + '</strong>.</li>').join('') +
    '</ul><div class="note" style="margin-top:10px">Estas son <strong>hipótesis comerciales</strong>, no resultados. Las llegadas registradas a Bolivia no demuestran demanda de SUP en el lago Titicaca ni disposición a pagar: requieren validación con datos de la experiencia.</div>';
}

/* ==================== Render gasto ==================== */
function gasIsRegion(){ return state.gas.pais === 'Todos' && state.gas.region !== 'Todas'; }
function gasSerie(){
  const {pais, region} = state.gas;
  if (pais !== 'Todos') return {serie: D.gastoPais[pais] || {}, label: pais, tipo: 'pais'};
  if (region !== 'Todas') return {serie: D.gastoRegion[region] || {}, label: 'Región ' + region, tipo: 'region'};
  return {serie: D.gastoTotal, label: 'Gasto receptor total', tipo: 'total'};
}
function gasPrelim(year){
  const {pais, region} = state.gas;
  if (pais !== 'Todos') return !!((D.prelimG[pais] || {})[String(year)]);
  if (region !== 'Todas') return (D.gastoRegionPaises[region] || []).some(p => (D.prelimG[p] || {})[String(year)]);
  return !!D.prelimGTotal[String(year)];
}
function gastoCount(year){
  const {pais, region} = state.gas;
  const list = region === 'Todas' ? D.paisesGasto : (D.gastoRegionPaises[region] || []);
  if (pais !== 'Todos') return (D.gastoPais[pais] || {})[String(year)] !== undefined ? 1 : 0;
  return list.filter(p => (D.gastoPais[p] || {})[String(year)] !== undefined).length;
}

function renderGas(){
  const {year} = state.gas;
  const sc = gasSerie();
  const cur = (sc.serie[String(year)] !== undefined) ? sc.serie[String(year)] : null;
  const prev = (sc.serie[String(year - 1)] !== undefined) ? sc.serie[String(year - 1)] : null;
  const varp = (cur !== null && prev !== null && prev > 0) ? ((cur - prev) / prev * 100) : null;
  const pre = gasPrelim(year);

  $('#g1v').textContent = fmtDec(cur) + ' M';
  $('#g1s').textContent = sc.label + ' · ' + year + ' · millones de USD' + (pre ? ' · incluye (p) preliminares' : '');
  $('#g2v').textContent = pct(varp);
  const g2s = $('#g2s');
  g2s.textContent = (prev === null || prev === undefined) ? 'Sin año anterior comparable (' + (year - 1) + ')'
    : 'vs ' + (year - 1) + ' (' + fmtDec(prev) + ' M USD)';
  g2s.className = 'sub ' + (varp === null ? '' : (varp >= 0 ? 'up' : 'down'));
  $('#g3v').textContent = fmtInt(gastoCount(year));
  $('#g3s').textContent = state.gas.pais === 'Todos'
    ? (state.gas.region === 'Todas' ? 'Países de residencia con dato en ' + year
        : 'Países de ' + state.gas.region + ' con dato en ' + year)
    : 'País de residencia seleccionado';
  $('#gasAlcance').textContent = 'Alcance: ' + sc.label + ' · ' + year +
    ' · por país de residencia (no es nacionalidad).' +
    (sc.tipo === 'region' ? ' Suma sólo de los países asignados a la región; excluye el total general y el residual.' : '');
  $('#gasNotaP').style.display = pre ? 'block' : 'none';

  const list = state.gas.region === 'Todas' ? D.paisesGasto : (D.gastoRegionPaises[state.gas.region] || []);
  const rank = list.map(p => ({
    label: p, value: (D.gastoPais[p] || {})[String(year)] !== undefined ? D.gastoPais[p][String(year)] : null,
    mark: (D.prelimG[p] || {})[String(year)] ? 'p' : null,
  })).filter(r => r.value !== null).sort((a, b) => b.value - a.value);
  hBar($('#chGastoRanking'), rank, {color: '#9a6b00', labelW: 150});
  $('#gasRankNota').textContent = 'Ranking de países de residencia por gasto · ' + year + ' · millones de USD.' +
    (state.gas.region === 'Todas' ? '' : ' Región: ' + state.gas.region + '.');

  renderGasTable();
}

function renderGastoRegion(){
  const years = D.aniosGasto;
  const series = D.regiones.map((r, i) => ({
    name: r, color: PALETTE[i % PALETTE.length],
    values: years.map(y => { const v = (D.gastoRegion[r] || {})[String(y)]; return v === undefined ? null : v; }),
  }));
  lineChart($('#chGastoRegion'), years.map(String), series,
    {height: 330, minTop: 50, padBottom: 40, fmt: function(x){ return fmtCompact(x) + ' M'; }});
  $('#gastoRegionLegend').innerHTML = series.map(se =>
    '<span><span class="sw" style="background:' + se.color + '"></span>' + esc(se.name) + '</span>').join('');
  const head = '<tr><th>Región</th>' + years.map(y => '<th class="num">' + y + '</th>').join('') + '</tr>';
  const body = D.regiones.map(r => '<tr><td>' + esc(r) + '</td>' +
    years.map(y => {
      const v = (D.gastoRegion[r] || {})[String(y)];
      return v === undefined ? '<td class="num zero">n/d</td>' : '<td class="num">' + fmtDec(v) + '</td>';
    }).join('') + '</tr>').join('');
  $('#gastoRegionTable').innerHTML = '<thead>' + head + '</thead><tbody>' + body + '</tbody>';
}

let gasSort = {col: 'valor', dir: -1};
function renderGasTable(){
  const {year, pais} = state.gas;
  const q = ($('#gasSearch').value || '').toLowerCase();
  const list = state.gas.region === 'Todas' ? D.paisesGasto : (D.gastoRegionPaises[state.gas.region] || []);
  let rows = [];
  if (pais === 'Todos'){
    list.forEach(p => {
      const v = (D.gastoPais[p] || {})[String(year)];
      if (v === undefined) return;
      const pv = (D.gastoPais[p] || {})[String(year - 1)];
      const yoy = (pv !== undefined && pv > 0) ? ((v - pv) / pv * 100) : null;
      rows.push({pais: p, per: String(year), valor: v, yoy: yoy, prelim: !!((D.prelimG[p] || {})[String(year)])});
    });
  } else {
    const serie = D.gastoPais[pais] || {};
    D.aniosGasto.forEach(y => {
      const v = serie[String(y)];
      if (v === undefined) return;
      const pv = serie[String(y - 1)];
      const yoy = (pv !== undefined && pv > 0) ? ((v - pv) / pv * 100) : null;
      rows.push({pais: pais, per: String(y), valor: v, yoy: yoy, prelim: !!((D.prelimG[pais] || {})[String(y)])});
    });
  }
  if (q) rows = rows.filter(r => r.pais.toLowerCase().indexOf(q) >= 0);
  rows.sort((a, b) => {
    const c = gasSort.col;
    if (c === 'pais' || c === 'per') return String(a[c]).localeCompare(String(b[c])) * gasSort.dir;
    const av = a[c] === null ? -Infinity : a[c], bv = b[c] === null ? -Infinity : b[c];
    return (av - bv) * gasSort.dir;
  });
  $('#gasTbody').innerHTML = rows.map(r => '<tr><td>' + esc(r.pais) + '</td><td>' + esc(r.per) +
    '</td><td class="num">' + fmtDec(r.valor) + '</td><td class="num">' + pct(r.yoy) + '</td><td>' +
    (r.prelim ? '<span class="badge p">(p) preliminar</span>' : '<span class="badge n">observado</span>') +
    '</td></tr>').join('');
  $('#gasCount').textContent = rows.length + ' fila(s)';
}

/* ==================== Inicialización ==================== */
function fillSelect(sel, values, allLabel){
  sel.innerHTML = (allLabel ? '<option value="">' + allLabel + '</option>' : '') +
    values.map(v => '<option value="' + esc(v.value) + '">' + esc(v.label) + '</option>').join('');
}
function fillNatOptions(){
  const reg = state.vis.region;
  const list = reg === 'Todas' ? D.nacionalidades : (D.regionPaises[reg] || []);
  fillSelect($('#fNat'), list.map(n => ({value: n, label: n})), 'Todas las nacionalidades');
  $('#fNat').value = '';
  state.vis.nat = 'Todas';
}
function fillGPaisOptions(){
  const reg = state.gas.region;
  const list = reg === 'Todas' ? D.paisesGasto : (D.gastoRegionPaises[reg] || []);
  fillSelect($('#gPais'), list.map(p => ({value: p, label: p})),
    reg === 'Todas' ? 'Todos los países' : 'Todos los países de la región');
  $('#gPais').value = '';
  state.gas.pais = 'Todos';
}

function fillQ2Pais(){
  const reg = state.q2.region;
  const list = (reg === 'Todas') ? D.nacionalidades.slice() : regionPaises(reg);
  fillSelect($('#q2Pais'), list.map(function(n){ return {value: n, label: n}; }), 'Todos los países');
  $('#q2Pais').value = '';
  state.q2.pais = 'Todos';
}

function init(){
  /* ---------- Datos y detalle · llegadas ---------- */
  fillSelect($('#fYear'), D.aniosVis.map(function(y){ return {value: String(y), label: String(y)}; }));
  $('#fYear').value = String(state.vis.year);
  const last5Vis = D.aniosVisCompletos.slice(-5);
  fillSelect($('#fPeriodo'), [
    {value: 'p2225', label: '2022–2025'},
    {value: 'last5', label: 'Últimos 5 años completos (' + last5Vis[0] + '–' + last5Vis[last5Vis.length - 1] + ')'},
    {value: 'hist', label: 'Histórico completo (' + D.aniosVis[0] + '–' + D.aniosVis[D.aniosVis.length - 1] + ')'},
  ]);
  $('#fPeriodo').value = 'p2225';
  fillSelect($('#fMonth'), D.meses.map(function(m, i){ return {value: String(i + 1), label: m}; }), 'Todos los meses');
  fillSelect($('#fRegion'), D.regiones.map(function(r){ return {value: r, label: r}; }), 'Todas las regiones');
  fillNatOptions();

  /* ---------- Estacionalidad regional ---------- */
  const regYears = D.aniosVis.filter(function(y){ return y >= D.statsRegion.desde && y <= D.statsRegion.hasta; });
  fillSelect($('#seaVista'), [
    {value: 'regiones', label: 'A · Comparar regiones'},
    {value: 'region', label: 'B · Analizar una región'},
    {value: 'paises', label: 'C · Comparar países (máx. 5)'},
  ]);
  fillSelect($('#seaYear'), [{value: 'avg', label: 'Promedio ' + regYears[0] + '–' + regYears[regYears.length - 1]}]
    .concat(regYears.map(function(y){ return {value: String(y), label: 'Año ' + y}; })));
  $('#seaYear').value = 'avg';
  fillSelect($('#seaRegion'), D.regiones.map(function(r){ return {value: r, label: r}; }));
  $('#seaRegion').value = state.sea.region = D.regiones[0];
  fillSelect($('#seaFormato'), [{value: 'lineas', label: 'Líneas'}, {value: 'barras', label: 'Barras agrupadas'}]);
  $('#seaFormato').value = 'lineas';
  fillSelect($('#regSel'), D.regiones.map(function(r){ return {value: r, label: r}; }));
  $('#regSel').value = state.reg.region = D.regiones[0];

  /* ---------- Gasto por país de residencia ---------- */
  fillSelect($('#gYear'), D.aniosGasto.map(function(y){ return {value: String(y), label: String(y)}; }));
  $('#gYear').value = String(state.gas.year);
  fillSelect($('#gRegion'), D.regiones.map(function(r){ return {value: r, label: r}; }), 'Todas las regiones');
  $('#gRegion').value = 'Todas';
  fillGPaisOptions();

  /* ---------- Pregunta 1 · filtros propios ---------- */
  fillSelect($('#q1Periodo'), [
    {value: 'p2225', label: '2022–2025 (referencia)'},
    {value: 'hist', label: 'Histórico completo (' + D.aniosVisCompletos[0] + '–' + D.aniosVisCompletos[D.aniosVisCompletos.length - 1] + ')'},
  ]);
  $('#q1Periodo').value = state.q1.periodo;
  fillSelect($('#q1Region'), D.regiones.map(function(r){ return {value: r, label: r}; }), 'Todas las regiones');
  $('#q1Region').value = '';

  /* ---------- Pregunta 2 · filtros propios ---------- */
  fillSelect($('#q2Region'), D.regiones.map(function(r){ return {value: r, label: r}; }), 'Todas las regiones');
  $('#q2Region').value = '';
  fillSelect($('#q2Periodo'), [{value: 'avg', label: 'Promedio 2022–2025'}]
    .concat([2022, 2023, 2024, 2025].map(function(y){ return {value: String(y), label: 'Año ' + y}; }))
    .concat([{value: 'hist', label: 'Promedio histórico ' + D.aniosVisCompletos[0] + '–' + D.aniosVisCompletos[D.aniosVisCompletos.length - 1]}]));
  $('#q2Periodo').value = 'avg';
  fillQ2Pais();

  /* ---------- Pregunta 3 · filtros propios ---------- */
  const q3Paises = D.paisesGasto.slice().sort(function(a, b){
    return ((D.gastoPais[b] || {})[String(D.ultimoGasto)] || 0) - ((D.gastoPais[a] || {})[String(D.ultimoGasto)] || 0);
  });
  fillSelect($('#q3Pais'), q3Paises.map(function(p){ return {value: p, label: p}; }));
  state.q3.pais = q3Paises[0] || '';
  $('#q3Pais').value = state.q3.pais;

  /* ---------- Eventos · llegadas ---------- */
  $('#fYear').addEventListener('change', function(e){ state.vis.year = +e.target.value; renderVis(); });
  $('#fPeriodo').addEventListener('change', function(e){ state.vis.periodo = e.target.value; renderVis(); });
  $('#fMonth').addEventListener('change', function(e){ state.vis.month = +e.target.value; renderVis(); });
  $('#fRegion').addEventListener('change', function(e){
    state.vis.region = e.target.value || 'Todas';
    fillNatOptions(); renderVis();
  });
  $('#fNat').addEventListener('change', function(e){ state.vis.nat = e.target.value || 'Todas'; renderVis(); });
  $('#visSearch').addEventListener('input', renderVisTable);

  /* ---------- Eventos · estacionalidad regional ---------- */
  $('#seaVista').addEventListener('change', function(e){
    state.sea.vista = e.target.value;
    if (state.sea.vista === 'paises') state.sea.cPaises = [];
    renderSeason();
  });
  $('#seaRegion').addEventListener('change', function(e){
    state.sea.region = e.target.value || D.regiones[0]; state.sea.cPaises = []; renderSeason();
  });
  $('#seaYear').addEventListener('change', function(e){
    state.sea.year = (e.target.value === 'avg') ? 'avg' : +e.target.value; renderSeason();
  });
  $('#seaFormato').addEventListener('change', function(e){ state.sea.formato = e.target.value; renderSeason(); });
  $('#regSel').addEventListener('change', function(e){ state.reg.region = e.target.value; renderRegiones(); });

  /* ---------- Eventos · gasto ---------- */
  $('#gYear').addEventListener('change', function(e){ state.gas.year = +e.target.value; renderGas(); });
  $('#gRegion').addEventListener('change', function(e){
    state.gas.region = e.target.value || 'Todas'; fillGPaisOptions(); renderGas();
  });
  $('#gPais').addEventListener('change', function(e){ state.gas.pais = e.target.value || 'Todos'; renderGas(); });
  $('#gasSearch').addEventListener('input', renderGasTable);

  /* ---------- Eventos · Pregunta 1 ---------- */
  $('#q1Periodo').addEventListener('change', function(e){ state.q1.periodo = e.target.value; q1Render(); q1PaisesRender(); });
  $('#q1Region').addEventListener('change', function(e){ state.q1.region = e.target.value || 'Todas'; q1Render(); q1PaisesRender(); });

  /* ---------- Eventos · Pregunta 2 ---------- */
  $('#q2Region').addEventListener('change', function(e){
    state.q2.region = e.target.value || 'Todas'; state.q2.cPaises = []; fillQ2Pais(); q2Render();
  });
  $('#q2Pais').addEventListener('change', function(e){ state.q2.pais = e.target.value || 'Todos'; state.q2.cPaises = []; q2Render(); });
  $('#q2Periodo').addEventListener('change', function(e){ state.q2.periodo = e.target.value; q2Render(); });

  /* ---------- Eventos · Pregunta 3 ---------- */
  $('#q3Pais').addEventListener('change', function(e){ state.q3.pais = e.target.value; q3Render(); });

  /* ---------- Ordenación de tablas ---------- */
  $$('#visTable th').forEach(function(th){
    th.addEventListener('click', function(){
      const c = th.dataset.col;
      visSort = {col: c, dir: visSort.col === c ? -visSort.dir : -1};
      renderVisTable();
    });
  });
  $$('#gasTable th').forEach(function(th){
    th.addEventListener('click', function(){
      const c = th.dataset.col;
      gasSort = {col: c, dir: gasSort.col === c ? -gasSort.dir : -1};
      renderGasTable();
    });
  });

  /* ---------- Pestañas ---------- */
  wireTabs();

  /* ---------- Primer render ---------- */
  renderVis();
  renderRegiones();
  renderSeason();
  renderGastoRegion();
  renderStats();
  renderInterpretacion();
  renderInferencial();
  renderGas();
  q1Render();
  q1PaisesRender();
  q2Render();
  q3Render();
  showTab(location.hash || '#prioritarios');
}

document.addEventListener('DOMContentLoaded', init);
"""


def construir_html(datos: dict) -> str:
    data_json = json.dumps(datos, ensure_ascii=False, separators=(",", ":"))
    return HTML_TEMPLATE.replace("/*__DATA__*/", data_json)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard de mercados turísticos internacionales — Bolivia / SUP Lago Titicaca</title>
<meta name="author" content="Andrea Irahola Ramirez">
<meta name="description" content="Dashboard interactivo para explorar llegadas de visitantes extranjeros y gasto turístico por país de residencia en Bolivia, orientado a la comercialización de experiencias SUP en el lago Titicaca.">
<style>__CSS__</style>
</head>
<body>
<header class="hero">
  <div class="wrap">
    <h1>Dashboard de mercados turísticos internacionales</h1>
    <p>Exploración interactiva de las llegadas de <strong>visitantes extranjeros</strong> a Bolivia (por nacionalidad) y del <strong>gasto turístico por país de residencia</strong>, para orientar la comercialización de experiencias de Stand Up Paddle (SUP) en el lago Titicaca. Los módulos de llegadas (nacionalidad) y gasto (residencia) se exploran por separado y nunca se cruzan.</p>
  </div>
</header>
<nav class="topnav" aria-label="Pestañas del dashboard">
  <div class="wrap">
    <a href="#prioritarios">1 · Mercados prioritarios</a>
    <a href="#calendario">2 · Calendario de campañas</a>
    <a href="#investigar">3 · Mercados para investigar</a>
    <a href="#datos">Datos y detalle</a>
    <a href="#pruebas">Pruebas estadísticas</a>
    <a href="#metodologia">Fuentes y metodología</a>
  </div>
</nav>
<main class="wrap">
  <section class="panel on" id="prioritarios">
    <h2>Pregunta 1 · ¿Qué mercados conviene priorizar? <span class="tag">llegadas por nacionalidad</span></h2>
    <p class="lead">Primero se comparan las <strong>regiones de procedencia</strong> y, al elegir una, los <strong>países</strong> que la componen. Cifras = llegadas registradas a Bolivia: miden flujo turístico, no demanda de SUP. La conclusión se deriva de los datos y de los filtros activos.</p>
    <div class="filters">
      <div><label for="q1Periodo">Periodo de análisis</label><select id="q1Periodo"></select></div>
      <div><label for="q1Region">Región de procedencia</label><select id="q1Region"></select></div>
    </div>

    <div class="card">
      <h3>Nivel 1 · Regiones de procedencia</h3>
      <p class="hint">Volumen de llegadas del periodo, participación sobre el total de las 7 regiones y variación interanual. Unidad: llegadas registradas.</p>
      <div id="q1RegChart"></div>
      <p class="chart-note" id="q1RegNota"></p>
      <div class="tbl-scroll" style="margin-top:12px"><table class="matrix" id="q1RegTable"></table></div>
      <p class="resp" id="q1Resp"></p>
      <div class="note" id="q1Warn"></div>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Evolución anual de las llegadas por región</h3>
      <p class="hint">Eje vertical: llegadas de visitantes extranjeros por año. Las 7 regiones comparten la misma escala.</p>
      <div id="q1RegEvol"></div>
      <div class="legend" id="q1EvolLegend" aria-label="Leyenda de regiones"></div>
      <p class="chart-note" id="q1EvolNota"></p>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Nivel 2 · Países dentro de la región seleccionada</h3>
      <p class="muted" id="q1PaisPrompt">Elige una <strong>región</strong> en el filtro superior para ver sus países.</p>
      <div id="q1PaisBox" style="display:none">
        <p class="hint">El gráfico muestra los <strong>cinco países con mayor volumen acumulado</strong> del periodo (criterio de escala, no de tasa de crecimiento). La tabla añade participación dentro de la región, variación interanual, tendencia histórica disponible y estacionalidad.</p>
        <div id="q1PaisChart"></div>
        <p class="chart-note" id="q1PaisNota"></p>
        <h4 class="mini">Participación sobre el total regional</h4>
        <div id="q1PaisPie"></div>
        <p class="chart-note" id="q1PaisPieNota"></p>
        <div class="tbl-scroll" style="margin-top:12px"><table class="matrix" id="q1PaisTable"></table></div>
        <p class="resp" id="q1PaisResp"></p>
      </div>
    </div>
  </section>

  <section class="panel" id="calendario">
    <h2>Pregunta 2 · ¿Cuándo conviene promocionar las experiencias SUP? <span class="tag">estacionalidad de llegadas</span></h2>
    <p class="lead">Compara los meses de mayor concentración de llegadas y consulta si las diferencias entre regiones y países justifican adaptar el calendario. Los datos describen llegadas, no reservas ni intención de compra.</p>
    <div class="filters">
      <div><label for="q2Region">Región de procedencia</label><select id="q2Region"></select></div>
      <div><label for="q2Pais">País (dentro de la región)</label><select id="q2Pais"></select></div>
      <div><label for="q2Periodo">Periodo de análisis</label><select id="q2Periodo"></select></div>
    </div>

    <div class="card">
      <h3>Distribución mensual de las llegadas</h3>
      <p class="hint">Cada valor es el porcentaje que aporta el mes al total anual del propio mercado (no al total nacional).</p>
      <div id="q2SeasonChart"></div>
      <div class="legend" id="q2Legend"></div>
      <p class="chart-note" id="q2SeasonNota"></p>
      <div id="q2PaisBox" style="display:none;margin-top:14px">
        <h4 class="mini">Comparar países de la región (máximo 5)</h4>
        <div class="checks" id="q2PaisChecks"></div>
        <p class="chart-note" id="q2PaisCount"></p>
      </div>
      <div class="grid2" style="margin-top:14px">
        <div><h4 class="mini">Meses de mayor afluencia</h4><div id="q2Fuertes"></div></div>
        <div><h4 class="mini">Meses de menor afluencia</h4><div id="q2Flojos"></div></div>
      </div>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Conclusión y recomendación comercial</h3>
      <p class="hint">La recomendación separa patrón descriptivo, significación estadística, tamaño del efecto y estabilidad observada entre años.</p>
      <div class="resp" id="q2Resp"></div>
      <div class="note" id="q2Warn"></div>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Qué permiten concluir los datos</h3>
      <p class="hint">A describe la concentración mensual dentro de cada región; B compara la distribución entre regiones; C compara países seleccionados dentro de una región. Un valor p pequeño no implica una diferencia comercial importante. La falta de significación no demuestra igualdad.</p>
      <div id="q2Stats"></div>
    </div>
    <p class="jumps"><a class="jump" href="#metodologia">→ Fuentes y metodología</a><a class="jump" href="#pruebas">→ Detalle de las pruebas</a></p>
  </section>

  <section class="panel" id="investigar">
    <h2>Pregunta 3 · ¿Cómo evoluciona el gasto turístico de cada país y qué señales ofrece para investigar ese mercado? <span class="tag">gasto agregado anual</span></h2>
    <p class="lead">Selecciona un país de residencia para consultar su gasto turístico agregado anual, su variación y la evidencia de tendencia. Las series de gasto se analizan por separado de las llegadas por nacionalidad.</p>
    <div class="filters">
      <div><label for="q3Pais">País de residencia</label><select id="q3Pais"></select></div>
    </div>

    <div class="card">
      <h3>Evolución del gasto turístico receptor total en Bolivia</h3>
      <p class="hint">Contexto nacional; gasto agregado anual en millones de USD.</p>
      <div id="q3GastoTotalChart"></div>
      <p class="chart-note" id="q3GastoTotalNota"></p>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Evolución del gasto agregado del país seleccionado</h3>
      <p class="hint">Gasto anual agregado por país de residencia, en millones de USD. La marca (p) identifica observaciones preliminares.</p>
      <div id="q3GastoChart"></div>
      <p class="chart-note" id="q3GastoNota"></p>
      <div class="tbl-scroll"><table id="q3GastoTable" class="matrix"></table></div>
      <div id="q3Tendencias" style="margin-top:12px"></div>
      <div class="card" style="margin-top:12px">
        <h3>Lectura gerencial: señales complementarias</h3>
        <p class="hint">Llegadas por nacionalidad y gasto por país de residencia se muestran como indicadores distintos. No representan necesariamente registros de una misma población.</p>
        <div id="q3ComparacionNota" class="note" style="margin:10px 0"></div>
        <div class="tbl-scroll"><table id="q3Escenarios" class="matrix"></table></div>
      </div>
      <div class="resp" id="q3Conclusion" style="margin-top:12px"></div>
      <div class="note" id="q3Limit" style="margin-top:12px"></div>
    </div>
  </section>

  <section class="panel" id="datos">
    <h2>Datos y detalle <span class="tag">consulta técnica</span></h2>
    <p class="lead">Todos los datos que alimentan las tres preguntas, con sus filtros completos y sus tablas. Las llegadas (nacionalidad) y el gasto (país de residencia) se exploran por separado y nunca se cruzan.</p>

    <div class="card">
      <h3>Llegadas de visitantes extranjeros <span class="tag">por nacionalidad</span></h3>
      <p class="hint">Fuente: registros de llegadas de visitantes extranjeros a Bolivia (ANDA / INE). Extranjeros = nivel de detalle excluyendo «Bolivia» (nacionales); el total no usa la fila <code>nivel="total"</code> porque incluye nacionales.</p>
      <div class="filters">
        <div><label for="fPeriodo">Periodo del gráfico de evolución</label><select id="fPeriodo"></select></div>
        <div><label for="fYear">Año</label><select id="fYear"></select></div>
        <div><label for="fMonth">Mes</label><select id="fMonth"></select></div>
        <div><label for="fRegion">Región</label><select id="fRegion"></select></div>
        <div><label for="fNat">Nacionalidad</label><select id="fNat"></select></div>
      </div>
      <div class="kpis">
        <div class="kpi"><div class="lab">Llegadas extranjeras del periodo</div><div class="val" id="k1v">—</div><div class="sub" id="k1s"></div></div>
        <div class="kpi"><div class="lab">Variación vs periodo anterior comparable</div><div class="val" id="k2v">—</div><div class="sub" id="k2s"></div></div>
        <div class="kpi"><div class="lab">Mercados extranjeros incluidos</div><div class="val" id="k3v">—</div><div class="sub" id="k3s"></div></div>
      </div>
      <p class="muted" id="visAlcance"></p>
      <div class="note" id="visNotaP" style="display:none">El periodo seleccionado incluye datos <strong>(p) preliminares</strong>. Las cifras preliminares pueden ser revisadas por la fuente.</div>

      <div class="grid2" style="margin-top:16px">
        <div><h4 class="mini">Evolución anual de llegadas</h4><div id="chEvolucion"></div><p class="chart-note" id="evolNota"></p></div>
        <div><h4 class="mini">Ranking de mercados</h4><div id="chRanking"></div><p class="chart-note" id="rankNota"></p></div>
      </div>
      <div style="margin-top:16px">
        <h4 class="mini">Estacionalidad mensual · año seleccionado vs año anterior</h4>
        <div id="chEstacional"></div><p class="chart-note" id="estNota"></p>
      </div>
      <div style="margin-top:16px">
        <h4 class="mini">Tabla exploratoria de llegadas</h4>
        <p class="hint">Búsqueda y ordenación por columna (haz clic en el encabezado). <span id="visCount" class="muted"></span></p>
        <div class="filters" style="grid-template-columns:1fr;max-width:420px;margin:8px 0 12px">
          <div><label for="visSearch">Buscar nacionalidad</label><input type="text" id="visSearch" placeholder="p. ej. Alemania"></div>
        </div>
        <div class="tbl-wrap">
          <table id="visTable">
            <thead><tr>
              <th data-col="nat" class="sortable">Nacionalidad</th>
              <th data-col="per" class="sortable">Periodo</th>
              <th data-col="valor" class="num sortable">Llegadas</th>
              <th data-col="prelim">Dato</th>
            </tr></thead>
            <tbody id="visTbody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Estacionalidad regional <span class="tag eu">2022–2025</span></h3>
      <p class="hint">Llegadas de visitantes extranjeros por nacionalidad agrupadas en regiones. Cifras de llegadas registradas a Bolivia: no equivalen a demanda de SUP ni a disposición a pagar.</p>
      <div style="margin-top:8px"><h4 class="mini">Evolución anual de las llegadas por región</h4><div id="chRegEvol"></div><div class="legend" id="regEvolLegend"></div></div>
      <div class="filters" style="margin-top:16px">
        <div><label for="seaVista">Vista</label><select id="seaVista"></select></div>
        <div id="seaRegionWrap"><label for="seaRegion">Región (vistas B y C)</label><select id="seaRegion"></select></div>
        <div><label for="seaYear">Periodo</label><select id="seaYear"></select></div>
        <div id="seaFormatoWrap"><label for="seaFormato">Formato (vista C)</label><select id="seaFormato"></select></div>
      </div>
      <div style="margin-top:8px">
        <h4 class="mini" id="seasonTitle">Estacionalidad · comparación de regiones</h4>
        <div id="chSeason"></div>
        <div class="legend" id="seasonLegend"></div>
        <p class="chart-note" id="seasonNote"></p>
        <p class="chart-note" id="seasonPeakNote"></p>
      </div>
      <div id="seaPaisBox" style="margin-top:16px;display:none">
        <h4 class="mini">Países a comparar</h4>
        <p class="hint" id="seaPaisHint"></p>
        <div class="checks" id="seaPaisChecks"></div>
        <p class="chart-note" id="seaPaisCount"></p>
      </div>
      <div style="margin-top:16px">
        <h4 class="mini">Tabla mensual complementaria</h4>
        <p class="hint">Meses en filas y regiones en columnas; porcentajes del total anual de cada región. <span class="muted" id="seasonRangeNote"></span></p>
        <div class="tbl-scroll"><table id="seasonTable" class="matrix"></table></div>
        <p class="chart-note">La fila «Mes pico» señala el mes de mayor afluencia de cada región y su participación. La fila «HHI norm.» resume la concentración mensual (0 = reparto uniforme entre 12 meses, 1 = todo en un mes).</p>
      </div>
      <div class="filters" style="margin-top:16px">
        <div><label for="regSel">Región (ranking de países)</label><select id="regSel"></select></div>
      </div>
      <div>
        <h4 class="mini">Ranking de países dentro de la región</h4>
        <p class="hint">Nacionalidades con más llegadas acumuladas 2022–2025 en la región seleccionada.</p>
        <div id="chRegRanking"></div>
        <p class="chart-note" id="regRankNota"></p>
      </div>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Gasto turístico por país de residencia <span class="tag">millones de USD</span></h3>
      <p class="hint">Fuente: serie de gasto turístico receptor por país de residencia (estimada a partir de las encuestas de gasto 2014–2016 y el flujo de visitantes). Es gasto <strong>agregado anual</strong>: no existe gasto mensual ni gasto por visitante. Este bloque es independiente del de llegadas por nacionalidad.</p>
      <div class="filters">
        <div><label for="gYear">Año</label><select id="gYear"></select></div>
        <div><label for="gRegion">Región</label><select id="gRegion"></select></div>
        <div><label for="gPais">País de residencia</label><select id="gPais"></select></div>
      </div>
      <div class="kpis">
        <div class="kpi"><div class="lab">Gasto del año seleccionado</div><div class="val" id="g1v">—</div><div class="sub" id="g1s"></div></div>
        <div class="kpi"><div class="lab">Variación interanual</div><div class="val" id="g2v">—</div><div class="sub" id="g2s"></div></div>
        <div class="kpi"><div class="lab">Países de residencia con dato</div><div class="val" id="g3v">—</div><div class="sub" id="g3s"></div></div>
      </div>
      <p class="muted" id="gasAlcance"></p>
      <div class="note" id="gasNotaP" style="display:none">El año seleccionado incluye datos <strong>(p) preliminares</strong>.</div>
      <div class="grid2" style="margin-top:16px">
        <div><h4 class="mini">Ranking de países de residencia</h4><div id="chGastoRanking"></div><p class="chart-note" id="gasRankNota"></p></div>
      </div>
      <div style="margin-top:16px">
        <h4 class="mini">Gasto anual agregado por región</h4>
        <div id="chGastoRegion"></div>
        <div class="legend" id="gastoRegionLegend"></div>
        <div class="tbl-scroll" style="margin-top:12px"><table id="gastoRegionTable" class="matrix"></table></div>
      </div>
      <div style="margin-top:16px">
        <h4 class="mini">Tabla exploratoria de gasto</h4>
        <p class="hint">Incluye la variación interanual calculada cuando el año anterior tiene dato. <span id="gasCount" class="muted"></span></p>
        <div class="filters" style="grid-template-columns:1fr;max-width:420px;margin:8px 0 12px">
          <div><label for="gasSearch">Buscar país</label><input type="text" id="gasSearch" placeholder="p. ej. España"></div>
        </div>
        <div class="tbl-wrap">
          <table id="gasTable">
            <thead><tr>
              <th data-col="pais" class="sortable">País de residencia</th>
              <th data-col="per" class="sortable">Año</th>
              <th data-col="valor" class="num sortable">Gasto (M USD)</th>
              <th data-col="yoy" class="num sortable">Var. interanual</th>
              <th data-col="prelim">Dato</th>
            </tr></thead>
            <tbody id="gasTbody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Resumen descriptivo</h3>
      <p class="hint">Síntesis de apoyo. Cada bloque distingue lo <span class="badge obs">observado</span> de lo <span class="badge hip">hipótesis</span>.</p>
      <div class="grid2" style="margin-top:8px">
        <div><h4 class="mini">Regiones con mayor volumen</h4><div id="interpVol"></div></div>
        <div><h4 class="mini">Regiones con mayor concentración mensual</h4><div id="interpConc"></div></div>
      </div>
      <div class="grid2" style="margin-top:16px">
        <div><h4 class="mini">País que destaca en cada región</h4><div id="interpDest"></div></div>
        <div><h4 class="mini">Meses candidatos para probar campañas</h4><div id="interpMeses"></div></div>
      </div>
    </div>
  </section>

  <section class="panel" id="pruebas">
    <h2>Pruebas estadísticas <span class="tag">consulta técnica</span></h2>
    <div class="note blue">
      <strong>¿Qué se contrasta?</strong> La <em>hipótesis nula (H0)</em> dice que el <strong>mercado de procedencia</strong> y el <strong>mes del calendario</strong> son independientes: el reparto mensual de llegadas sería el mismo en todos los mercados. Se rechaza H0 cuando el valor p es muy pequeño. Como el número de registros es enorme, el valor p se vuelve diminuto incluso para diferencias pequeñas; por eso se informa la <strong>V de Cramér</strong>, que mide el <em>tamaño del efecto</em> (de «despreciable» a «fuerte»).
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Chi-cuadrado región × mes (2022–2025) <span class="tag eu">todas las regiones</span></h3>
      <p class="hint">Tabla de contingencia región × mes sobre <strong>llegadas mensuales agregadas</strong> (registros administrativos, no personas independientes).</p>
      <div class="statrow">
        <div class="stat"><div class="l">Observaciones (Σ)</div><div class="v" id="stN">—</div></div>
        <div class="stat"><div class="l">Regiones</div><div class="v" id="stReg">—</div></div>
        <div class="stat"><div class="l">Grados de libertad</div><div class="v" id="stGl">—</div></div>
        <div class="stat"><div class="l">Valor p</div><div class="v" id="stP">—</div></div>
        <div class="stat"><div class="l">V de Cramér</div><div class="v" id="stV">—</div></div>
        <div class="stat"><div class="l">Frec. esperada mínima</div><div class="v" id="stMinEsp">—</div></div>
        <div class="stat"><div class="l">Celdas con esperado &lt; 5</div><div class="v" id="stPctEsp">—</div></div>
      </div>
      <div class="tbl-scroll"><table id="stTable" class="matrix"></table></div>
      <div class="note" style="margin-top:12px">
        <strong>Advertencias.</strong> Los datos son <strong>administrativos y agregados</strong>; existe posible <strong>dependencia temporal</strong> (estacionalidad, tendencia, pandemia) y no hay muestreo probabilístico. Por eso la prueba se ofrece como <em>indicio</em>, no como prueba concluyente: una significación estadística <strong>no implica relevancia comercial</strong>.
      </div>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Prueba europea previa (país × mes) <span class="tag">2022–2025 · sólo Europa</span></h3>
      <p class="hint">Reproduce la prueba de <code>scripts/05_pruebas_inferenciales.py</code>: los <strong>8 principales mercados europeos</strong> por volumen. <strong>No</strong> incluye otras regiones y no debe presentarse como válida para todas ellas.</p>
      <div class="statrow">
        <div class="stat"><div class="l">χ² (Europa)</div><div class="v" id="euChi">—</div></div>
        <div class="stat"><div class="l">Grados de libertad</div><div class="v" id="euGl">—</div></div>
        <div class="stat"><div class="l">Valor p</div><div class="v" id="euP">—</div></div>
        <div class="stat"><div class="l">V de Cramér</div><div class="v" id="euV">—</div></div>
        <div class="stat"><div class="l">Observaciones (Σ)</div><div class="v" id="euN">—</div></div>
        <div class="stat"><div class="l">Cobertura de la selección</div><div class="v" id="euCob">—</div></div>
        <div class="stat"><div class="l">Frec. esperada mínima</div><div class="v" id="euMinEsp">—</div></div>
        <div class="stat"><div class="l">Celdas con esperado &lt; 5</div><div class="v" id="euPctEsp">—</div></div>
      </div>
      <p class="muted"><strong>Países usados:</strong> <span id="euPaises"></span>.</p>
      <p class="chart-note" id="euCheck"></p>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Análisis inferencial complementario (Etapa 5) <span class="tag">selección de mercados</span></h3>
      <p class="hint">Pruebas <strong>complementarias</strong> a los análisis ya validados: tendencias por país (Mann-Kendall + pendiente de Sen, con corrección de comparaciones múltiples Benjamini-Hochberg y de autocorrelación de Hamed-Rao) y homogeneidad estacional entre países candidatos (2022-2025). Se distingue siempre significación estadística de relevancia comercial. El detalle completo está en:</p>
      <p class="muted" id="inferArchivos"></p>
      <div id="inferContenido"></div>
    </div>
  </section>

  <section class="panel" id="metodologia">
    <h2>Fuentes y metodología</h2>
    <p class="lead">Qué datos usa el dashboard, cómo se calculan los indicadores y qué <strong>no</strong> permiten afirmar. Se conservan las referencias del proyecto; no se añaden fuentes, enlaces ni fechas nuevas.</p>

    <div class="card">
      <h3>Fuentes de datos</h3>
      <ul class="tight">
        <li><strong>INE · Estadísticas de Flujo de Viajeros y Visitantes Internacionales.</strong> Archivo <code>datos/Bolivia_Datos_Visitantes_Extranjeros.xlsx</code> (no modificado), con las hojas <code>Visitantes_Nacionalidad</code>, <code>Ocupación Hotelera</code>, <code>Gasto_Turístico_Receptor</code> y <code>Gasto_Receptor_País</code>.</li>
        <li><strong>Documentación de la fuente:</strong> <code>datos/ddi-documentation-spanish-200.pdf</code> (DDI del INE).</li>
        <li><strong>Documentos de contexto del proyecto:</strong> <code>datos/demanda-turistica GMLP.pdf</code> y <code>datos/PowerPoint Presentation - eca-sport-program-2025-2028_sup-6786.pdf</code>.</li>
        <li><strong>Bases tidy derivadas</strong> (etapas 2–7 del proyecto, en <code>resultados/</code>): <code>visitantes_mensual_largo.csv</code> y <code>visitantes_anual_largo.csv</code> (por nacionalidad), <code>gasto_pais_anual_largo.csv</code> (por país de residencia), <code>ocupacion_hotelera_mensual_largo.csv</code> (por ciudad de hospedaje).</li>
        <li><strong>Resultados estadísticos reutilizados:</strong> <code>resultado_prueba_inferencial.txt</code> y <code>tabla_contingencia_pais_mes_europa.csv</code> (prueba europea previa), <code>desc_visitantes_anual_historico.csv</code> (contraste del total extranjero) y las salidas de la Etapa 5: <code>tendencias_mann_kendall_por_pais.csv</code>, <code>homogeneidad_estacional_paises.csv</code>, <code>homogeneidad_estacional_residuos.csv</code>, <code>tabla_decision_mercados_potenciales.csv</code>, <code>mercados_evolucion_reciente_2022_2025.csv</code> e <code>interpretacion_inferencial_complementaria.txt</code>.</li>
        <li><strong>Notas de las figuras:</strong> <code>graficas/notas_graficas.txt</code> y <code>graficas/notas_graficas_addendum.txt</code>.</li>
      </ul>
      <p class="chart-note">Periodos cubiertos: llegadas 2008–2026 (2026 sólo enero–julio: año parcial); gasto 2008–2025; ocupación hotelera 2020–2026. El análisis principal, la estacionalidad y los contrastes usan 2022–2025.</p>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Definición de los principales indicadores</h3>
      <ul class="tight">
        <li><strong>Llegadas de visitantes extranjeros (nacionalidad).</strong> Registros de la hoja <code>Visitantes_Nacionalidad</code> con <code>nivel="detalle"</code>, excluyendo la nacionalidad «Bolivia» (nacionales) y las filas TOTAL. La categoría residual «Resto de países» sólo se cuenta en el total extranjero.</li>
        <li><strong>Participación (%).</strong> Cuota del mercado o región sobre el total que corresponda al gráfico: total de las 7 regiones, total de la región o total extranjero.</li>
        <li><strong>Variación interanual (%).</strong> <code>(valor_año − valor_año_anterior)/valor_año_anterior × 100</code>, sólo si existe el año anterior y es mayor que 0; en otro caso <code>n/d</code>.</li>
        <li><strong>Estacionalidad.</strong> Porcentaje que aporta cada mes al total anual del propio mercado o región (no al total nacional). El <em>promedio</em> promedia los porcentajes de cada año del periodo.</li>
        <li><strong>HHI normalizado.</strong> <code>(HHI − 1/12)/(1 − 1/12)</code> con <code>HHI = Σ pᵢ²</code>: 0 = reparto uniforme entre 12 meses, 1 = todo en un mes.</li>
        <li><strong>Gasto turístico receptor (M USD).</strong> Agregado anual por país de residencia; serie estimada. Se muestran el total, el agregado por región de residencia y el del país homónimo, siempre en columnas separadas.</li>
        <li><strong>Tendencia histórica.</strong> Mann-Kendall con pendiente de Sen, p ajustado por Benjamini-Hochberg y corrección de autocorrelación de Hamed-Rao; series 2008–2019 (pre-pandemia) y 2008–2025.</li>
        <li><strong>V de Cramér.</strong> Tamaño del efecto de una prueba chi-cuadrado: &lt; 0,05 despreciable; 0,05–0,10 muy débil; 0,10–0,20 débil; 0,20–0,30 moderada; ≥ 0,30 fuerte.</li>
        <li><strong>Prioridad de mercado (V/C/T).</strong> Criterios explícitos, sin índice compuesto: <em>V</em> = llegadas 2025 ≥ percentil 75 de los 51 mercados; <em>C</em> = crecimiento 2024→2025 &gt; 0; <em>T</em> = tendencia pre-pandemia creciente concluyente. Es una <em>selección de mercados a investigar</em>, no una predicción de ventas.</li>
      </ul>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Nacionalidad, residencia y categorías agregadas</h3>
      <ul class="tight">
        <li><strong>Nacionalidad ≠ país de residencia.</strong> Las llegadas se registran por nacionalidad del visitante; el gasto, por país de residencia de quien gasta. No son necesariamente la misma población.</li>
        <li><strong>Categorías residuales.</strong> «Resto de países» / «Resto de Países» no representa un país individual y no debe interpretarse como tal. Se incluye en el total, pero no entra en rankings ni en los agregados regionales de llegadas.</li>
        <li><strong>Sin doble conteo.</strong> No se suma un total agregado con sus componentes.</li>
        <li><strong>Datos preliminares (p).</strong> El sello <span class="badge p">(p) preliminar</span> indica que el dato es preliminar y puede ser revisado por la fuente. 2023–2025 son preliminares.</li>
        <li><strong>2026.</strong> Sólo hay meses parciales (enero–julio): no se compara como si fuera un año completo.</li>
      </ul>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Advertencias metodológicas</h3>
      <ol class="steps">
        <li>Los datos de llegadas por nacionalidad no identifican por sí solos a los turistas interesados en experiencias SUP ni a quienes visitan específicamente Copacabana o el lago Titicaca.</li>
        <li>La nacionalidad y el país de residencia son variables diferentes.</li>
        <li>Las categorías residuales, como «Resto de países», no representan un país individual y no deben interpretarse como tal.</li>
        <li>No debe sumarse un total agregado con sus componentes: produciría doble conteo.</li>
        <li>Los datos preliminares marcados con (p) pueden ser revisados.</li>
        <li>Los datos de 2026 abarcan únicamente los meses disponibles; no deben compararse como si fueran un año completo.</li>
        <li>La pandemia y las posibles estimaciones afectan la interpretación de determinadas series (2020–2021 rompen la continuidad).</li>
        <li>Las llegadas a establecimientos de alojamiento no equivalen necesariamente a personas únicas, noches de estancia o duración total del viaje.</li>
        <li>El gasto turístico agregado no permite deducir automáticamente gasto individual, gasto diario, disposición a pagar ni gasto en SUP.</li>
        <li>Las pruebas de asociación estadística no demuestran causalidad ni garantizan que una estrategia publicitaria sea rentable.</li>
        <li>Para fines exploratorios, se supone que la distribución por nacionalidad de las llegadas es una aproximación de la distribución por país de residencia del gasto turístico. Esta equivalencia no puede verificarse con los datos agregados disponibles.</li>
      </ol>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Tendencia del gasto por país de residencia (Pregunta 3)</h3>
      <ul class="tight">
        <li>Se aplica Mann–Kendall y la pendiente de Sen a las series anuales de gasto agregado por país (2008–2025), con periodo pre-pandemia (2008–2019) como análisis de sensibilidad. Los resultados existentes de <code>tendencias_mann_kendall_por_pais.csv</code> corresponden a llegadas por nacionalidad y no se reutilizan como resultados de gasto.</li>
        <li>La autocorrelación de rangos se ajusta con Hamed–Rao; los p-valores se corrigen entre países con Benjamini–Hochberg. La incertidumbre de la magnitud se resume con IC95% de Sen. La clasificación de gasto alto/bajo compara cada país con los cuartiles de gasto agregado de 2025 (alto ≥ P75; bajo ≤ P25); no mide gasto individual.</li>
        <li>El gasto tiene 18 años por país sin faltantes en esta extracción, pero 2023–2025 son preliminares y 2020–2021 muestran una ruptura asociada a la pandemia. El periodo histórico mezcla caída y recuperación; el corte pre-pandemia sirve de sensibilidad. No se realiza una prueba formal de cambio estructural y las conclusiones son exploratorias.</li>
      </ul>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Pruebas estadísticas y criterios de interpretación</h3>
      <ul class="tight">
        <li><strong>Chi-cuadrado región × mes (2022–2025)</strong> sobre llegadas mensuales agregadas, con V de Cramér como tamaño de efecto y comprobación de frecuencias esperadas.</li>
        <li><strong>Prueba europea previa (país × mes, 8 mercados, 2022–2025).</strong> Se recalcula y se contrasta con <code>resultado_prueba_inferencial.txt</code>.</li>
        <li><strong>Homogeneidad estacional país × mes por región (2022–2025)</strong> de la Etapa 5, con residuos ajustados y corrección Benjamini-Hochberg.</li>
        <li><strong>Criterio general.</strong> Los datos son administrativos y agregados: no hay muestreo aleatorio y puede existir dependencia temporal. Con N muy grande el valor p es diminuto incluso para diferencias triviales, así que la decisión se apoya en la <em>magnitud</em> del efecto (V de Cramér) y en la importancia comercial del mercado, no sólo en la significación.</li>
        <li>No se ejecutan pruebas adicionales para cada ranking descriptivo: para eso se usan las salidas ya validadas de las etapas previas.</li>
      </ul>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Supuesto al comparar nacionalidad y residencia</h3>
      <ul class="tight">
        <li>Para fines exploratorios, se supone que la distribución por nacionalidad de las llegadas es una aproximación de la distribución por país de residencia del gasto turístico. Esta equivalencia no puede verificarse con los datos agregados disponibles.</li>
        <li>Este supuesto no constituye una equivalencia demostrada. La Pregunta 3 presenta las tendencias de llegadas por nacionalidad y gasto por residencia como señales separadas; no vincula registros ni calcula gasto por visitante.</li>
        <li>El gasto agregado anual no demuestra rentabilidad, demanda de SUP ni disposición a pagar; la conclusión automática recomienda investigar el mercado, no invertir automáticamente en publicidad.</li>
        <li>Para evaluar el potencial real harían falta datos primarios sobre interés por SUP, precio aceptable, intención de reserva y canales de compra.</li>
      </ul>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Limitaciones de los datos de hotelería y de gasto</h3>
      <ul class="tight">
        <li>El gasto es un <em>agregado anual estimado</em> por país de residencia: no permite gasto mensual ni gasto por visitante.</li>
        <li>La suma de regiones de gasto no incluye el residual «Resto de Países» ni los países no listados; por tanto no iguala al total receptor.</li>
        <li>La serie de gasto no es comparable término a término con la de llegadas por nacionalidad (variables geográficas distintas).</li>
        <li>Los datos de ocupación hotelera corresponden a la ciudad de hospedaje (ámbito La Paz / Nacional) y a categorías por tipo; no son intercambiables con las llegadas por nacionalidad.</li>
        <li>Regiones con poco volumen (p. ej. África) tienen series muy pequeñas; su HHI y su perfil mensual son poco estables.</li>
      </ul>
    </div>
  </section>
</main>
<footer>Dashboard generado por <code>scripts/09_dashboard_html.py</code> · HTML autocontenido y sin conexión externa.</footer>

<script>window.__DATA__=/*__DATA__*/;</script>
<script>__JS__</script>
</body>
</html>
"""


def main() -> int:
    datos = construir_datos()
    html_out = construir_html(datos)
    html_out = html_out.replace("__CSS__", CSS).replace("__JS__", JS)
    OUT.write_text(html_out, encoding="utf-8")

    checks = []
    # Estructura, autocontención y pestañas
    checks.append(("Datos embebidos presentes", "window.__DATA__=" in html_out and '"nacionalidades"' in html_out))
    externos = ['src="http', 'href="http', '@import', 'url(http']
    checks.append(("Sin recursos externos", not any(s in html_out for s in externos)))
    checks.append(("Sin librerías/CDN externos", not any(s in html_out for s in ("cdn.", "unpkg", "jsdelivr"))))
    checks.append(("Idioma español", '<html lang="es">' in html_out))

    paneles = ["prioritarios", "calendario", "investigar", "datos", "pruebas", "metodologia"]
    checks.append(("6 pestañas declaradas",
                   all((f'<section class="panel on" id="{p}">' in html_out
                        or f'<section class="panel" id="{p}">' in html_out) for p in paneles)))
    checks.append(("Navegación enlaza las 6 pestañas",
                   all(f'href="#{p}"' in html_out for p in paneles)))
    checks.append(("Navegación por pestañas cableada",
                   "function showTab" in html_out and "wireTabs()" in html_out))
    checks.append(("Los filtros comerciales NO se acoplan entre preguntas",
                   "state.q1.region" in html_out and "state.q2.region" in html_out
                   and "renderVis(); renderDecisiones();" not in html_out))

    # Pregunta 1 · Mercados prioritarios (dos niveles)
    checks.append(("P1 · regiones y países con tabla, gráficos y conclusión",
                   all(f'id="{i}"' in html_out for i in (
                       "q1Periodo", "q1Region", "q1RegChart", "q1RegNota", "q1RegEvol", "q1EvolLegend", "q1EvolNota",
                       "q1RegTable", "q1Resp", "q1Warn", "q1PaisPrompt", "q1PaisBox", "q1PaisChart",
                       "q1PaisPie", "q1PaisPieNota", "q1PaisTable", "q1PaisResp"))))
    checks.append(("P1 · leyenda de regiones y nota del gráfico de evolución",
                   "$('#q1EvolLegend').innerHTML = regSeries.map" in html_out
                   and "Cada color corresponde a una región, identificada en la leyenda." in html_out))
    checks.append(("P1 · torta usa total regional y se renderiza al cambiar filtros",
                   "function pieChart" in html_out and "o.vol / tot * 100" in html_out
                   and "resto / tot * 100" in html_out
                   and "q1PaisesRender();" in html_out
                   and "state.q1.periodo = e.target.value; q1Render(); q1PaisesRender();" in html_out
                   and "state.q1.region = e.target.value || 'Todas'; q1Render(); q1PaisesRender();" in html_out))
    # Pregunta 2 · Calendario de campañas
    checks.append(("P2 · filtros, estacionalidad, recomendación y pruebas",
                   all(f'id="{i}"' in html_out for i in (
                       "q2Region", "q2Pais", "q2Periodo", "q2SeasonChart", "q2Legend",
                       "q2PaisBox", "q2PaisChecks", "q2Fuertes", "q2Flojos",
                       "q2Stats", "q2Resp", "q2Warn"))))
    checks.append(("P2 · eliminados filtro y calendario de anticipación",
                   all(x not in html_out for x in ("q2Lead", "q2LeadChart", "q2LeadTable",
                                                     "meses de anticipación", "Inicio de campaña"))))
    checks.append(("P2 · distinción A/B/C, efecto y recomendación automática",
                   all(x in html_out for x in ("Estacionalidad dentro de cada región (descriptiva)",
                                                "Diferencias entre regiones", "Diferencias entre países",
                                                "p ajustado", "V de Cramér", "calendario general"))))
    checks.append(("P2 · filtros vuelven a renderizar la distribución mensual",
                   all(x in html_out for x in (
                       "state.q2.region = e.target.value || 'Todas'; state.q2.cPaises = []; fillQ2Pais(); q2Render();",
                       "state.q2.pais = e.target.value || 'Todos'; state.q2.cPaises = []; q2Render();",
                       "state.q2.periodo = e.target.value; q2Render();"))))
    checks.append(("P2 · líneas comparan los países seleccionados y la nota lo explica",
                   "series = state.q2.cPaises.map(function(p, i){" in html_out
                   and "Las líneas comparan el patrón mensual de los países seleccionados." in html_out))
    checks.append(("P2 · reutiliza p ajustado BH existente en homogeneidad regional",
                   '"pBH":' in html_out and "h.pBH < 0.05" in html_out))
    # Pregunta 3 · Mercados para investigar
    checks.append(("P3 · selector de país, gráficos, tabla, tendencia y conclusión",
                   all(f'id="{i}"' in html_out for i in (
                       "q3Pais", "q3GastoTotalChart", "q3GastoChart", "q3GastoTable",
                       "q3Tendencias", "q3ComparacionNota", "q3Escenarios", "q3Conclusion", "q3Limit"))))
    checks.append(("P3 · selector cambia la serie y la tendencia del país",
                   "state.q3.pais = e.target.value; q3Render();" in html_out
                   and "D.gastoPais[pais]" in html_out and "D.gastoTendencias || {})[pais]" in html_out))
    checks.append(("P3 · matriz anterior eliminada",
                   all(x not in html_out for x in ("q3Matriz", "q3MatrizNota", "function scatterChart"))))
    checks.append(("P3 · tendencias separadas de llegadas y gasto, escenarios y cautelas",
                   all(x in html_out for x in (
                       "Llegadas por nacionalidad · histórico 2008–2025",
                       "Gasto por residencia · histórico 2008–2025",
                       "D.infer.mercados[pais]", "Llegadas crecientes + gasto creciente",
                       "La relación observada no demuestra causalidad",
                       "Esta lectura sirve para priorizar investigación comercial"))))

    # Pestaña 4 · Datos y detalle
    checks.append(("KPIs de llegadas", all(f'id="k{i}v"' in html_out for i in (1, 2, 3))))
    checks.append(("KPIs de gasto", all(f'id="g{i}v"' in html_out for i in (1, 2, 3))))
    checks.append(("Filtros de llegadas", all(f'id="{i}"' in html_out for i in ("fYear", "fMonth", "fRegion", "fNat"))))
    checks.append(("Filtros de gasto (año/región/país)", all(f'id="{i}"' in html_out for i in ("gYear", "gRegion", "gPais"))))
    checks.append(("Estacionalidad: gráfico + selectores A/B/C",
                   all(f'id="{i}"' in html_out for i in ("chSeason", "seaVista", "seaRegion",
                                                          "seaRegionWrap", "seaYear", "seaFormato",
                                                          "seaPaisChecks", "seasonTable"))))
    checks.append(("Selector de periodo de llegadas; sin selector redundante de gasto",
                   'id="fPeriodo"' in html_out and 'id="gPeriodo"' not in html_out))
    checks.append(("2026 no se trata como año completo",
                   2026 not in datos["aniosVisCompletos"]))
    checks.append(("Gasto por región: gráfico + tabla",
                   all(f'id="{i}"' in html_out for i in ("chGastoRegion", "gastoRegionTable"))))
    checks.append(("Tablas exploratorias", all(f'id="{i}"' in html_out for i in ("visTable", "gasTable"))))

    # Pestaña 5 · Pruebas
    checks.append(("Chi-cuadrado región x mes en la pestaña de pruebas",
                   all(f'id="{i}"' in html_out for i in ("pruebas", "stTable", "euChi", "euV"))))
    checks.append(("Etapa 5 embebida", 'id="inferContenido"' in html_out))
    checks.append(("Etapa 5 disponible si hay salidas",
                   bool(datos["infer"]["disponible"])))
    if datos["infer"]["disponible"]:
        inf = datos["infer"]
        _m = inf.get("mercados", {}).get("Perú", {})
        _pre = (_m.get("tend") or {}).get("pre") or {}
        checks.append(("P1/P3: detalle por mercado (51)", len(inf.get("mercados", {})) == 51))
        checks.append(("P2: homogeneidad por región disponible", len(inf.get("homogeneidad", [])) >= 7))
        checks.append(("P1/P3: prioridad y tendencia por mercado",
                       bool(_m.get("prioridad")) and _pre.get("sen") is not None))

    # Pestaña 6 · Fuentes y metodología
    checks.append(("Pestaña «Fuentes y metodología» completa",
                   'id="metodologia"' in html_out
                   and "Definición de los principales indicadores" in html_out
                   and "Advertencias metodológicas" in html_out
                   and "Esta equivalencia no puede verificarse con los datos agregados disponibles." in html_out
                   and "datos/Bolivia_Datos_Visitantes_Extranjeros.xlsx" in html_out))

    # Integridad de datos
    nac = datos["nacionalidades"]
    checks.append(("Bolivia excluida de los mercados", "Bolivia" not in nac))
    checks.append(("51 mercados extranjeros reales", len(nac) == 51))
    checks.append(("7 regiones", len(datos["regiones"]) == 7))
    # Integridad de la torta: top 5 + países restantes cubren el total regional.
    shares_pie = []
    for reg, paises_reg in datos["regionPaises"].items():
        vol_reg = sum((datos["annual"].get(p, {}).get("2022", 0) or 0) +
                      (datos["annual"].get(p, {}).get("2023", 0) or 0) +
                      (datos["annual"].get(p, {}).get("2024", 0) or 0) +
                      (datos["annual"].get(p, {}).get("2025", 0) or 0)
                      for p in paises_reg)
        vol_paises = sorted([
            sum((datos["annual"].get(p, {}).get(str(y), 0) or 0) for y in (2022, 2023, 2024, 2025))
            for p in paises_reg
        ], reverse=True)
        top = vol_paises[:5]
        resto = max(0, vol_reg - sum(top))
        shares_pie.append((reg, (sum(top) + resto) / vol_reg * 100 if vol_reg else 0))
    checks.append(("P1 · porcentajes de torta sobre total regional suman 100% (2022–2025)",
                   all(abs(pct_total - 100) < 1e-9 for _, pct_total in shares_pie)))

    desc = pd.read_csv(RES / "desc_visitantes_anual_historico.csv").set_index("anio")["visitantes_extranjeros"]
    for y in (2022, 2023, 2024, 2025):
        v = datos["totalAnnual"].get(str(y))
        checks.append((f"Extranjeros {y} = desc ({desc[y]:,.0f})", v is not None and abs(v - float(desc[y])) < 1))
    checks.append(("Total extranjero 2024 ≠ total internacional", abs(datos["totalAnnual"].get("2024", 0) - 2373596.0) > 1))
    checks.append(("Gasto total 2025 = CSV", abs((datos["gastoTotal"].get("2025") or 0) - 818.5512) < 0.01))
    checks.append(("P3 · series de gasto completas 2008–2025 (29 países)",
                   len(datos["paisesGasto"]) == 29 and all(
                       all(datos["gastoPais"][p].get(str(y)) is not None for y in range(2008, 2026))
                       for p in datos["paisesGasto"])))
    checks.append(("P3 · tendencia calculada sobre gasto del país, n=18",
                   all(datos["gastoTendencias"][p]["hist"].get("n") == 18
                       and datos["gastoTendencias"][p]["hist"].get("evaluable")
                       for p in datos["paisesGasto"])))
    checks.append(("P3 · clasificación de escala transparente por cuartiles 2025",
                   datos["gastoCuartiles"]["n"] == 29
                   and datos["gastoCuartiles"]["p25"] < datos["gastoCuartiles"]["p75"]))

    sr = datos["statsRegion"]
    checks.append(("χ² región×mes: gl=66", sr["gl"] == 66))
    checks.append(("V región×mes en [0,1]", 0.0 <= sr["cramerV"] <= 1.0))

    # Gasto por región: cobertura y exclusión de totales/residuales
    gano = [p for reg, ps in datos["gastoRegionPaises"].items() for p in ps]
    checks.append(("Todos los países de gasto con región", len(gano) == len(datos["paisesGasto"])))
    checks.append(("Regiones de gasto sin TOTAL ni residual",
                   all(r in REGIONES for r in datos["gastoRegion"])))
    sum_reg_2025 = sum((datos["gastoRegion"][r].get("2025") or 0) for r in datos["gastoRegion"])
    checks.append(("Σ regiones gasto 2025 < total receptor (residual excluido)",
                   sum_reg_2025 < (datos["gastoTotal"].get("2025") or 0)))

    # Prueba europea: recomputada y cotejada con el archivo guardado
    se = datos["statsEuropa"]
    checks.append(("Prueba Europa: 8 países", len(se["paises"]) == 8))
    checks.append(("Prueba Europa coincide con archivo guardado", bool(se["coincide"])))

    print("Dashboard generado:", OUT)
    print("Tamaño:", f"{OUT.stat().st_size/1024:.1f} KiB")
    print("Mercados:", len(nac), "| Regiones:", len(datos["regiones"]),
          "| Países gasto:", len(datos["paisesGasto"]), "| Regiones gasto:", len(datos["gastoRegion"]))
    print(f"Extranjeros 2024 = {datos['totalAnnual'].get('2024'):,.0f} (Bolivia excluida)")
    print(f"χ² región×mes = {sr['chi2']:,.3f} · gl={sr['gl']} · V={sr['cramerV']:.4f} "
          f"· minEsp={sr['minEsperado']:.1f} · celdas<5={sr['pctEsperadoLt5']:.1f}%")
    print(f"Europa: χ²={se['chi2']:,.3f} · gl={se['gl']} · V={se['cramerV']:.4f} "
          f"· coincide con TXT: {se['coincide']} (guardado χ²={se['chi2Guardado']}, V={se['vGuardado']})")
    print(f"Σ regiones gasto 2025 = {sum_reg_2025:,.2f} M < total {datos['gastoTotal'].get('2025'):,.2f} M")
    print()
    fallos = 0
    for label, ok in checks:
        print(f"  [{'OK' if ok else 'REVISAR'}] {label}")
        if not ok:
            fallos += 1
    print()
    if fallos:
        print(f"EXIT=1 · {fallos} verificación(es) con discrepancia")
        return 1
    print("EXIT=0 · verificaciones de integridad superadas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
