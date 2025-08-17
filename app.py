# EHF Nordeste 

import os, json, re
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --------- Config (por padrão lê arquivos no mesmo diretório) ---------
ARQ_PREV = os.environ.get("PREV_XLSX", "previsao_ne_5dias.xlsx")
ARQ_ATTR = os.environ.get("ATTR_XLSX", "arquivo_ne_completo_preenchido.xlsx")
GEOJSON_PATH = os.environ.get("GEOJSON_PATH", "municipios_nordeste.geojson")

CLASS_ORDER = ["Normal","Baixa intensidade","Severa","Extrema"]
COLOR_MAP = {
    "Normal": "#2E7D32",
    "Baixa intensidade": "#F1C40F",
    "Severa": "#E67E22",
    "Extrema": "#C0392B",
}
BAR_COLOR_MAP = {
    "Tmín":  "#BFDBFE",
    "Tméd":  "#60A5FA",
    "Tmáx":  "#1E3A8A",
}
RISK_ORDER  = ["Baixo","Moderado","Alto","Muito alto"]
RISK_COLORS = {"Baixo":"#65A30D","Moderado":"#FACC15","Alto":"#FB923C","Muito alto":"#DC2626"}

# --------- Helpers ---------
def z7(s):
    return pd.Series(s, dtype=str).str.extract(r"(\d+)")[0].str.zfill(7)

def calc_ehf(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["CD_MUN","data"]).reset_index(drop=True)
    def _t3_centered_keep_index(s: pd.Series) -> pd.Series:
        m = s.rolling(3, center=True, min_periods=3).mean()
        if len(s) >= 3:
            m.iloc[0]  = s.iloc[0:3].mean()
            m.iloc[-1] = s.iloc[-3:].mean()
        return m
    t3 = df.groupby("CD_MUN")["Tmean"].apply(_t3_centered_keep_index).reset_index(level=0, drop=True)
    df["T3d_prev"] = t3
    df["EHI_sig"]  = df["T3d_prev"] - df["Tmean_p95"]
    df["EHI_accl"] = df["T3d_prev"] - df["Tmean_30d"]
    df["EHF"]      = df["EHI_sig"].clip(lower=0) * df["EHI_accl"].apply(lambda x: x if pd.notna(x) and x > 1 else 1)
    return df

def classify_by_tmean_percentis(df):
    has_p90 = "Tmean_p90" in df.columns
    has_p99 = "Tmean_p99" in df.columns
    if has_p90 and has_p99:
        gate = (df["EHF"] > 0)
        def _cls(row):
            if not gate.loc[row.name]: return "Normal"
            tm = row["T3d_prev"]; p90=row["Tmean_p90"]; p95=row["Tmean_p95"]; p99=row["Tmean_p99"]
            if pd.isna(tm) or pd.isna(p90) or pd.isna(p95) or pd.isna(p99): return "Normal"
            if tm >= p99: return "Extrema"
            if tm >= p95: return "Severa"
            if tm >= p90: return "Baixa intensidade"
            return "Normal"
        df["classification"] = df.apply(_cls, axis=1)
        df["ratio"] = (df["T3d_prev"] / df["Tmean_p99"]).where(df["EHF"] > 0)
        return df
    if "EHF99" in df.columns:
        def _ratio(row):
            e, t = row["EHF"], row["EHF99"]
            if pd.isna(e) or e <= 0 or pd.isna(t): return np.nan
            if t <= 0: return float("inf")
            return e / t
        df["ratio"] = df.apply(_ratio, axis=1)
        def _cls2(row):
            if pd.isna(row["EHF99"]) or row["EHF"] <= 0: return "Normal"
            if row["ratio"] < 1: return "Baixa intensidade"
            if row["ratio"] < 3: return "Severa"
            return "Extrema"
        df["classification"] = df.apply(_cls2, axis=1)
    else:
        df["classification"] = np.where(df["EHF"]>0, "Baixa intensidade", "Normal")
        df["ratio"] = np.nan
    return df

def load_geojson():
    """Carrega GeoJSON, normaliza CD_MUN, recorta NE se SIGLA_UF existir, e loga o que achou."""
    here = Path(__file__).resolve().parent
    candidates = []
    if GEOJSON_PATH:
        p = Path(GEOJSON_PATH)
        candidates += [str(p), str(here / p), str(here / p.name)]
    candidates += [
        str(here / "municipios_nordeste.geojson"),
        str(here / "data" / "municipios_nordeste.geojson"),
    ]
    tried = []
    for p in candidates:
        if not p: continue
        tried.append(p)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8-sig") as f:
                    gj = json.load(f)
            except Exception as e:
                print(f"[geojson] Erro lendo {p}: {e}")
                continue

            feats = []
            has_uf = False
            for ft in gj.get("features", []):
                props = (ft.get("properties") or {})
                cd = props.get("CD_MUN")
                if not cd:
                    for k in ("CD_GEOCMU","CD_GEOCODI","CD_MUNIC","CD_IBGE","GEOCODIGO","GEOCODI","id"):
                        if props.get(k):
                            cd = props[k]; break
                cd = re.sub(r"\D", "", str(cd or "")).zfill(7)
                props["CD_MUN"] = cd
                if props.get("SIGLA_UF"):
                    has_uf = True
                ft["properties"] = props
                feats.append(ft)

            if has_uf:
                NE = {"BA","SE","AL","PE","PB","RN","CE","PI","MA"}
                feats = [ft for ft in feats if ft["properties"].get("SIGLA_UF") in NE]

            gj["features"] = feats
            print(f"[geojson] OK: {p} — {len(feats)} features")
            if feats:
                print("[geojson] props exemplo:", list(feats[0]["properties"].keys())[:8],
                      "CD_MUN exemplo:", feats[0]["properties"].get("CD_MUN"))
            return gj
    print(f"[geojson] NÃO ENCONTRADO. Tentativas: {tried}")
    return None

# --------- GeoSES + risco combinado ---------
def geoses_from_attr(attr_df: pd.DataFrame):
    if "GeoSES" not in attr_df.columns: return None
    out = attr_df[["CD_MUN","GeoSES"]].copy()
    out["CD_MUN"] = z7(out["CD_MUN"])
    out["V"] = ((1 - pd.to_numeric(out["GeoSES"], errors="coerce")) / 2).clip(0, 1)
    return out[["CD_MUN","GeoSES","V"]]

def add_combined_risk(df, geoses):
    if geoses is None:
        df["H_norm"]=np.nan; df["V"]=np.nan; df["risk_index"]=np.nan; df["risk_class"]=np.nan
        return df, False
    d = df.merge(geoses, on="CD_MUN", how="left")
    ratio = (d["EHF"].clip(lower=0)) / d["EHF99"].replace(0, np.nan)
    H = ratio.clip(0, 1).fillna(0)
    V = d["V"]
    risk = 0.5*H + 0.5*V
    d["H_norm"]     = H
    d["risk_index"] = risk
    d["risk_class"] = pd.cut(risk, [-0.001,0.25,0.5,0.75,1.0], labels=RISK_ORDER, include_lowest=True)
    has_risk = d["V"].notna().any()
    return d, has_risk

# --------- Carrega dados ---------
if not os.path.exists(ARQ_PREV):
    raise SystemExit(f"❌ PREV_XLSX não encontrado: {ARQ_PREV}")
if not os.path.exists(ARQ_ATTR):
    raise SystemExit(f"❌ ATTR_XLSX não encontrado: {ARQ_ATTR}")

prev = pd.read_excel(ARQ_PREV, engine="openpyxl")
attr = pd.read_excel(ARQ_ATTR, engine="openpyxl")
prev["CD_MUN"]=z7(prev["CD_MUN"]); attr["CD_MUN"]=z7(attr["CD_MUN"])
prev["data"]=pd.to_datetime(prev["data"], errors="coerce")

keep_attr = [
    "CD_MUN","NM_MUN","SIGLA_UF","lat","lon","NM_REGIAO","GeoSES",
    "Tmean_p90","Tmean_p95","Tmean_p99","EHF85","EHF90","EHF95","EHF99","Tmean_30d","Tmean_3d"
]
keep_attr = [c for c in keep_attr if c in attr.columns]

base = prev.merge(attr[keep_attr], on=["CD_MUN","NM_MUN","SIGLA_UF"], how="left")
base = calc_ehf(base)
base = classify_by_tmean_percentis(base)

geoses = geoses_from_attr(attr)
base, HAS_RISK = add_combined_risk(base, geoses)

# listas
ufs   = sorted(base["SIGLA_UF"].dropna().unique().tolist())
dates = sorted(base["data"].dropna().dt.date.unique().tolist())
gj = load_geojson()

# --------- App ---------
app = Dash(__name__)
server = app.server
app.title = "Fator de Excesso de Calor (EHF) – Nordeste"

layer_options = [{"label":"Classificação EHF","value":"ehf"}]
if HAS_RISK:
    layer_options.append({"label":"Risco combinado (EHF+GeoSES)","value":"risk"})

controls = html.Div([
    html.Div([
        html.Label("Filtrar por estado (SIGLA_UF)"),
        dcc.Dropdown(id="uf-filter",
                     options=[{"label": u, "value": u} for u in ufs],
                     value=[], multi=True, placeholder="SIGLA_UF"),
    ], style={"minWidth":"220px","flex":"1","marginRight":"8px"}),
    html.Div([
        html.Label("Filtrar por município"),
        dcc.Dropdown(id="muni-filter", options=[], value=[], multi=True, placeholder="Município"),
    ], style={"minWidth":"320px","flex":"2","marginRight":"8px"}),
    html.Div([
        html.Label("Data"),
        dcc.Slider(id="date-slider",
                   min=0, max=max(len(dates)-1,0), step=1,
                   value=max(len(dates)-1,0),
                   marks={i: d.strftime("%d/%m") for i, d in enumerate(dates)} if dates else {})
    ], style={"minWidth":"320px","flex":"3","marginRight":"8px"}),
    html.Div([
        html.Label("Camada"),
        dcc.RadioItems(id="layer", options=layer_options, value=layer_options[0]["value"], inline=True)
    ], style={"minWidth":"320px","flex":"2"})
], style={"display":"flex","gap":"12px","alignItems":"center","marginBottom":"10px","flexWrap":"wrap"})

two_cols = html.Div([
    html.Div([dcc.Graph(id="mapa", style={"height":"68vh"})], style={"flex":"1","paddingRight":"8px"}),
    html.Div([
        dcc.Graph(id="barras", style={"height":"55vh","marginBottom":"8px"}),
        html.Div(id="cards-ehf", style={"display":"grid","gridTemplateColumns":"repeat(5, 1fr)","gap":"8px"})
    ], style={"flex":"1","paddingLeft":"8px"})
], style={"display":"flex","gap":"8px"})

# painel por classificação
class_panel = html.Div([
    html.H4("Consulta por classificação (EHF no dia)"),
    html.Div([
        html.Div([
            html.Label("Classificação"),
            dcc.Dropdown(
                id="class-filter",
                options=[{"label": c, "value": c} for c in CLASS_ORDER],
                value="Extrema",
                clearable=False
            )
        ], style={"minWidth":"300px","maxWidth":"360px","marginRight":"12px"}),
        html.Div(id="class-count", style={"alignSelf":"center","fontWeight":"800","fontSize":"16px"})
    ], style={"display":"flex","alignItems":"center","gap":"12px","flexWrap":"wrap","marginBottom":"8px"}),
    html.Div(id="class-list", style={
        "maxHeight":"28vh","overflowY":"auto","backgroundColor":"#f8fafc",
        "padding":"10px","borderRadius":"8px","border":"1px solid #e5e7eb"
    })
], style={"marginTop":"14px"})

store_sel = dcc.Store(id="muni-sel")

app.layout = html.Div(style={"fontFamily":"Inter, system-ui, Arial","padding":"12px"}, children=[
    html.H3("Fator de Excesso de Calor (EHF) – Nordeste"),
    controls, two_cols, class_panel, store_sel
])

# --------- Callbacks ---------
@app.callback(
    Output("muni-filter", "options"),
    Output("muni-filter", "value"),
    Input("uf-filter", "value"),
    State("muni-filter", "value"),
)
def update_muni_dropdown(ufs_sel, current_vals):
    df = base.copy()
    if ufs_sel:
        df = df[df["SIGLA_UF"].isin(ufs_sel)]
    muni = (df[["CD_MUN","NM_MUN","SIGLA_UF"]]
            .drop_duplicates()
            .sort_values(["SIGLA_UF","NM_MUN"]))
    options = [{"label": f"{r.NM_MUN} / {r.SIGLA_UF}", "value": r.CD_MUN}
               for r in muni.itertuples()]
    valid = set(muni["CD_MUN"])
    value = [v for v in (current_vals or []) if v in valid]
    return options, value

@app.callback(
    Output("mapa","figure"),
    Input("uf-filter","value"),
    Input("muni-filter","value"),
    Input("date-slider","value"),
    Input("layer","value"),
    Input("muni-sel","data")
)
def update_map(ufs_sel, munis_sel, date_idx, layer, sel_cd):
    dff = base.copy()
    if ufs_sel:
        dff = dff[dff["SIGLA_UF"].isin(ufs_sel)]
    if munis_sel:
        dff = dff[dff["CD_MUN"].isin(munis_sel)]
    if dates:
        d_sel = dates[int(date_idx)]
        dff = dff[dff["data"].dt.date == d_sel]

    use_risk = (layer == "risk") and HAS_RISK
    cat_col  = "risk_class" if use_risk else "classification"
    order    = RISK_ORDER if use_risk else CLASS_ORDER
    cmap     = RISK_COLORS if use_risk else COLOR_MAP
    legend   = "Risco combinado" if use_risk else "Classificação"
    if cat_col not in dff.columns or dff[cat_col].notna().sum() == 0:
        cat_col, order, cmap, legend = "classification", CLASS_ORDER, COLOR_MAP, "Classificação"

    dff["cat"] = pd.Categorical(dff[cat_col], categories=order, ordered=True)
    dff["cat_EHF"] = dff[cat_col].astype(str)

    # --- Escolhe featureidkey dinamicamente e calcula taxa de match (debug em logs)
    feature_key = "properties.CD_MUN"
    gj_ok = (gj is not None) and bool(gj.get("features"))
    if gj_ok:
        f0 = gj["features"][0]
        if "properties" in f0 and "CD_MUN" in f0["properties"]:
            feature_key = "properties.CD_MUN"
        elif "id" in f0:
            feature_key = "id"
        # taxa de match
        gj_ids = set()
        if feature_key == "properties.CD_MUN":
            gj_ids = { (ft["properties"].get("CD_MUN") or "").zfill(7) for ft in gj["features"] }
        else:
            gj_ids = { str(ft.get("id") or "").zfill(7) for ft in gj["features"] }
        df_ids = set(dff["CD_MUN"])
        match = len(df_ids & gj_ids)
        print(f"[map] match CD_MUN: {match}/{len(df_ids)} (df) com {len(gj_ids)} (geojson) | featureidkey={feature_key}")

    if gj_ok:
        fig = px.choropleth_mapbox(
            dff, geojson=gj, locations="CD_MUN", featureidkey=feature_key,
            color="cat", color_discrete_map=cmap,
            hover_data={"NM_MUN": True, "SIGLA_UF": True, "cat_EHF": True, "cat": False, "CD_MUN": False},
            center={"lat": -8.9, "lon": -38.5}, zoom=4.1, height=680
        )
        fig.update_traces(marker_line_width=1.6, marker_line_color="#1f2937")
        fig.update_layout(mapbox_style="carto-positron", margin=dict(l=0,r=0,t=0,b=0),
                          legend_title_text=legend)
        if sel_cd:
            fig.add_trace(go.Choroplethmapbox(
                geojson=gj, locations=[sel_cd], featureidkey=feature_key,
                z=[1], colorscale=[[0,'rgba(0,0,0,0)'], [1,'rgba(0,0,0,0)']],
                marker_line_width=3.0, marker_line_color="#111111",
                showscale=False, hoverinfo="skip"
            ))
        return fig

    # ---- Fallback: pontos (para não ficar “nada”)
    print("[map] GeoJSON ausente — fallback para pontos (lat/lon)")
    fig = px.scatter_mapbox(
        dff, lat="lat", lon="lon",
        color="cat", color_discrete_map=cmap,
        hover_name="NM_MUN",
        hover_data={"SIGLA_UF": True, "cat_EHF": True, "cat": False, "lat": False, "lon": False},
        zoom=4.2, height=680
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=0, b=0),
                      legend_title_text=legend)
    return fig

@app.callback(
    Output("muni-sel","data"),
    Input("mapa","clickData"),
    State("muni-sel","data"),
    prevent_initial_call=True
)
def keep_selection(clickData, sel):
    if clickData and clickData.get("points"):
        p = clickData["points"][0]
        cd = str(p.get("location") or p.get("customdata") or "").zfill(7)
        if cd.strip("0"): return cd
    return sel

@app.callback(
    Output("barras","figure"),
    Output("cards-ehf","children"),
    Input("muni-sel","data"),
    Input("uf-filter","value"),
    Input("muni-filter","value"),
    Input("date-slider","value"),
)
def update_side(sel_cd, ufs_sel, munis_sel, date_idx):
    b = base.copy()
    if ufs_sel:
        b = b[b["SIGLA_UF"].isin(ufs_sel)]
    if munis_sel:
        b = b[b["CD_MUN"].isin(munis_sel)]

    valid_cds = set(b["CD_MUN"])
    if sel_cd not in valid_cds:
        if munis_sel:
            sel_cd = munis_sel[0]
        else:
            if dates:
                d_sel = dates[int(date_idx)]
                day = b[b["data"].dt.date == d_sel]
                if len(day) > 0:
                    sel_cd = day.sort_values("T3d_prev", ascending=False)["CD_MUN"].iloc[0]
            if sel_cd is None and len(b) > 0:
                sel_cd = b["CD_MUN"].iloc[0]

    muni = b[b["CD_MUN"]==sel_cd].sort_values("data")
    if len(muni)==0:
        return px.bar(pd.DataFrame({"Dia":[],"Variável":[],"Valor":[]}), x="Dia", y="Valor"), []

    tmp = muni[["data","Tmin","Tmean","Tmax"]].copy()
    tmp.rename(columns={"Tmin":"Tmín","Tmean":"Tméd","Tmax":"Tmáx"}, inplace=True)
    tmp["Dia"] = tmp["data"].dt.strftime("%d/%m")
    long = tmp.melt(id_vars=["data","Dia"], var_name="Variável", value_name="Valor")

    ymax = float(long["Valor"].max()) if len(long) else 0.0
    headroom = ymax * 0.12 if ymax > 0 else 5

    fig_bar = px.bar(
        long, x="Dia", y="Valor",
        color="Variável", barmode="group",
        color_discrete_map=BAR_COLOR_MAP, text="Valor",
        labels={"Valor":"Temperatura (°C)"},
        title=f"Temperaturas – {muni['NM_MUN'].iloc[0]} / {muni['SIGLA_UF'].iloc[0]}"
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
    fig_bar.update_yaxes(range=[0, ymax + headroom], ticks="outside")
    fig_bar.update_layout(margin=dict(l=8, r=8, t=48, b=8), legend_title_text="")

    cards=[]
    for _, r in muni.iterrows():
        dt=r["data"].date(); cl=r["classification"]; cor=COLOR_MAP.get(cl, "#9aa5b1")
        cards.append(
            html.Div([
                html.Div(dt.strftime("%d/%m"), style={"fontSize":"12px","fontWeight":"600","marginBottom":"4px"}),
                html.Div(cl, style={"fontSize":"14px","fontWeight":"700"})
            ], style={
                "backgroundColor": cor, "color":"white", "borderRadius":"10px",
                "padding":"10px", "textAlign":"center", "minHeight":"60px",
                "boxShadow":"0 1px 2px rgba(0,0,0,.08)"
            })
        )
    return fig_bar, cards

@app.callback(
    Output("class-count","children"),
    Output("class-list","children"),
    Input("class-filter","value"),
    Input("uf-filter","value"),
    Input("muni-filter","value"),
    Input("date-slider","value"),
)
def list_by_class(sel_class, ufs_sel, munis_sel, date_idx):
    dff = base.copy()
    if ufs_sel:
        dff = dff[dff["SIGLA_UF"].isin(ufs_sel)]
    if munis_sel:
        dff = dff[dff["CD_MUN"].isin(munis_sel)]
    d_sel = dates[int(date_idx)] if dates else None
    if d_sel is not None:
        dff = dff[dff["data"].dt.date == d_sel]

    dff = dff[dff["classification"] == sel_class]
    if dff.empty:
        return html.Div("0 município(s) na categoria selecionada."), html.Div("Nenhum município com os filtros atuais.")

    dff = dff[["CD_MUN","NM_MUN","SIGLA_UF","EHF"]].drop_duplicates("CD_MUN").sort_values(["SIGLA_UF","NM_MUN"])
    count = len(dff)
    header = html.Div(f"{count} município(s) na categoria: {sel_class} – {d_sel.strftime('%d/%m') if d_sel else ''}")
    lines = "\n".join(f"- {row.NM_MUN} / {row.SIGLA_UF} — EHF {row.EHF:.2f}" for row in dff.itertuples())
    return header, dcc.Markdown(lines)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)






