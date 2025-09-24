# pages/04_Manobras_NA_Percursos.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import re
from collections import defaultdict
from io import BytesIO

st.set_page_config(page_title="Manobras (NA) + Percursos NF", layout="wide")

# ================= Helpers =================
def _norm(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()

def _find_col(df, aliases):
    low = {c.strip().lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in low:
            return low[a.lower()]
    return None

OPEN_PATTERNS = [
    r"\bna\b",            # 'NA'
    r"\bn\.a\.\b",        # 'N.A.'
    r"\bnormally\s*open\b",
    r"\bna\s*open\b",
    r"\bopen\b",
    r"\babert",           # 'aberta/aberto'
]

def is_NA(status: str) -> bool:
    t = _norm(status).lower()
    if not t:
        return False
    return any(re.search(p, t) for p in OPEN_PATTERNS)

def split_feeders(s: str):
    s = _norm(s)
    return [p.strip() for p in s.replace(";", ",").split(",") if p.strip()] or ["(sem alimentador)"]

def to_xlsx_bytes(sheets: dict) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as wr:
        for name, df in sheets.items():
            (df if df is not None else pd.DataFrame()).to_excel(
                wr, index=False, sheet_name=(name or "Sheet")[:31]
            )
    return bio.getvalue()

# ================= Core =================
def identify_na_and_paths(df_raw: pd.DataFrame):
    """Retorna (df_all, df_mano, na_paths, res_na, res_feed)"""
    # detectar colunas
    name_col   = _find_col(df_raw, ["Name","Chave","Equipamento"])
    parent_col = _find_col(df_raw, ["Name Bloco","Name_Bloc","NameBloco","Chave Pai","Pai"])
    estado_col = _find_col(df_raw, ["Estado Normal","Estado_Normal","EstadoNormal","Estado","Status"])
    feed_col   = _find_col(df_raw, ["Alimentador","Feeder"])

    if not name_col or not parent_col or not estado_col or not feed_col:
        raise ValueError("Planilha precisa conter: 'Name', 'Name Bloco', 'Estado Normal' e 'Alimentador' (nomes equivalentes aceitos).")

    df = df_raw[[name_col, parent_col, estado_col, feed_col]].copy()
    for c in df.columns:
        df[c] = df[c].astype(str).map(_norm)

    # flag NA
    df["Is_NA"] = df[estado_col].map(is_NA)

    # resumo simples de NA por alimentador
    def _split(s):
        s = _norm(s)
        return [p.strip() for p in s.replace(";", ",").split(",") if p.strip()] or ["(sem alimentador)"]

    rows = []
    for _, r in df.iterrows():
        for f in _split(r[feed_col]):
            rows.append({"Alimentador": f, "Is_NA": bool(r["Is_NA"])})
    long = pd.DataFrame(rows)
    resumo_simple = (long.groupby("Alimentador", as_index=False)
                          .agg(Qtd_NA=("Is_NA","sum"), Qtd_Total=("Is_NA","count")))
    resumo_simple["Pct_NA_%"] = (resumo_simple["Qtd_NA"]/resumo_simple["Qtd_Total"]*100).round(2)

    # construir hierarquia
    tree = defaultdict(list)
    reverse = {}
    for _, r in df.iterrows():
        ch = _norm(r[name_col]); pa = _norm(r[parent_col])
        if ch and pa and pa.lower() not in {"nan","none","null"}:
            if ch not in tree[pa]:
                tree[pa].append(ch)
            reverse[ch] = pa

    # feeders por chave
    k2f = defaultdict(set)
    for _, r in df.iterrows():
        k = _norm(r[name_col]); raw = _norm(r[feed_col])
        for f in _split(raw):
            k2f[k].add(f)

    # lookup de estado
    state = {_norm(r[name_col]): _norm(r[estado_col]) for _, r in df.iterrows()}

    def upstream_within_nf(node: str, feeder: str):
        up_nf = []
        cur = node
        seen = set()
        while cur in reverse and cur not in seen:
            seen.add(cur)
            p = reverse[cur]
            if feeder not in k2f.get(p, set()):
                break
            if is_NA(state.get(p, "")):
                break
            up_nf.append(p)
            cur = p
        return up_nf

    def downstream_nf(node: str, feeder: str):
        acc = []
        seen = set()
        def dfs(n):
            for c in tree.get(n, []):
                if feeder not in k2f.get(c, set()):
                    continue
                if is_NA(state.get(c, "")):
                    continue
                if c in seen: 
                    continue
                seen.add(c)
                acc.append(c)
                dfs(c)
        dfs(node)
        return acc

    # tabela por NA x alimentador
    rows = []
    na_df = df[df["Is_NA"]].copy()
    for _, r in na_df.iterrows():
        name = _norm(r[name_col])
        feeders = _split(r[feed_col])
        for f in feeders:
            up = upstream_within_nf(name, f)
            down = downstream_nf(name, f)
            rows.append({
                "Chave_NA": name,
                "Alimentador": f,
                "Upstream_NF": " \\ ".join(up) if up else "",
                "Downstream_NF": " \\ ".join(down) if down else "",
                "Len_Up_NF": len(up),
                "Len_Down_NF": len(down)
            })
    na_paths = pd.DataFrame(rows, columns=["Chave_NA","Alimentador","Upstream_NF","Downstream_NF","Len_Up_NF","Len_Down_NF"])

    if not na_paths.empty:
        res_na = na_paths.groupby("Chave_NA", as_index=False).agg(
            Alimentadores=("Alimentador","nunique"),
            NF_Up_Total=("Len_Up_NF","sum"),
            NF_Down_Total=("Len_Down_NF","sum")
        )
        res_feed = (na_paths.groupby("Alimentador", as_index=False)
                    .agg(Chaves_NA=("Chave_NA","nunique"),
                         NF_Up=("Len_Up_NF","sum"),
                         NF_Down=("Len_Down_NF","sum")))
    else:
        res_na = pd.DataFrame(columns=["Chave_NA","Alimentadores","NF_Up_Total","NF_Down_Total"])
        res_feed = pd.DataFrame(columns=["Alimentador","Chaves_NA","NF_Up","NF_Down"])

    # df_all com flag
    df_all = df.rename(columns={
        name_col:"Name", parent_col:"Name Bloco", estado_col:"Estado Normal", feed_col:"Alimentador"
    })
    df_mano = df_all[df_all["Is_NA"]].copy()

    return df_all, df_mano, na_paths, res_na, res_feed, resumo_simple


# ================= UI =================
st.title("⚡ Manobras (NA) — Percursos NF e Exportações")

up = st.file_uploader("Envie a planilha **Chaves.xlsx/CSV** (deve conter: Name, Name Bloco, Estado Normal, Alimentador)", type=["xlsx","xls","csv"])
if up:
    # ler
    if up.name.lower().endswith(".csv"):
        df_raw = pd.read_csv(up, dtype=str).fillna("")
    else:
        df_raw = pd.read_excel(up, dtype=str).fillna("")
    try:
        df_all, df_mano, na_paths, res_na, res_feed, resumo_simple = identify_na_and_paths(df_raw)
    except Exception as e:
        st.error(f"Erro ao processar: {e}")
        st.stop()

    st.success(f"Arquivo carregado. Registros: {len(df_all)} | Manobras (NA): {len(df_mano)}")

    # ----------------- Filtros -----------------
    st.subheader("Filtros")
    col_f1, col_f2, col_f3 = st.columns([1,1,1])

    # options de alimentador
    alimentadores = sorted({f for s in df_all["Alimentador"].apply(lambda x: [p.strip() for p in str(x).replace(';',',').split(',') if p.strip()]) for f in s})
    sel_feeders = col_f1.multiselect("Filtrar por Alimentador", options=alimentadores, default=[])

    filtro_na = col_f2.text_input("Filtrar por Chave NA (contém)")
    filtro_nf = col_f3.text_input("Filtrar por Chave NF (contém em Upstream/Downstream)")

    # aplicar filtros na tabela de percursos
    filtered = na_paths.copy()
    if sel_feeders:
        filtered = filtered[filtered["Alimentador"].isin(sel_feeders)]
    if filtro_na:
        t = filtro_na.strip().upper()
        filtered = filtered[filtered["Chave_NA"].str.upper().str.contains(t, na=False)]
    if filtro_nf:
        t = filtro_nf.strip().upper()
        mask = (filtered["Upstream_NF"].str.upper().str.contains(t, na=False)) | \
               (filtered["Downstream_NF"].str.upper().str.contains(t, na=False))
        filtered = filtered[mask]

    st.markdown("### Percursos por Chave **NA** (com filtros)")
    st.dataframe(filtered, use_container_width=True, height=360)

    st.markdown("### Resumo por **NA**")
    st.dataframe(res_na, use_container_width=True, height=260)

    st.markdown("### Resumo por **Alimentador**")
    st.dataframe(res_feed, use_container_width=True, height=260)

    # ----------------- Downloads -----------------
    st.subheader("Exportar")

    # XLSX completo
    all_xlsx = to_xlsx_bytes({
        "NA_Percursos": na_paths,
        "Resumo_por_NA": res_na,
        "Resumo_por_Alimentador": res_feed,
        "Todas_com_Flag": df_all,
        "Manobras": df_mano,
        "Resumo_Simples": resumo_simple,
    })
    st.download_button("⬇️ Baixar XLSX (Completo)", data=all_xlsx,
                       file_name="NA_Percursos_Completo.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # XLSX somente o filtrado
    filt_xlsx = to_xlsx_bytes({
        "NA_Percursos(Filtrado)": filtered,
        "Resumo_por_NA": res_na if sel_feeders or filtro_na or filtro_nf else res_na,  # mantém resumos
        "Resumo_por_Alimentador": res_feed,
    })
    st.download_button("⬇️ Baixar XLSX (Apenas Filtro)", data=filt_xlsx,
                       file_name="NA_Percursos_Filtrado.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("Envie a planilha para começar. Dica: pode ser **.xlsx** ou **.csv** (com colunas: Name, Name Bloco, Estado Normal, Alimentador).")
