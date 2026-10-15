import sys
import subprocess

# --- 0. AUTO-INSTALACIÓN AUTOMÁTICA DE DEPENDENCIAS ---
REQUERIDOS = ["pandas", "numpy", "plotly", "dash"]

def verificar_e_instalar(paquetes):
    for paquete in paquetes:
        try:
            __import__(paquete)
        except ImportError:
            print(f"📦 La librería '{paquete}' no está instalada. Instalando automáticamente...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", paquete])

verificar_e_instalar(REQUERIDOS)

# --- 1. IMPORTACIONES PRINCIPALES ---
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table

# --- 2. CARGA Y PREPARACIÓN DE DATOS ---
pelis_df = pd.read_csv("./netflix_movies_cleaned.csv")
series_df = pd.read_csv("./netflix_tv_shows_cleaned.csv")

pelis_df["tipo"] = "Película"
series_df["tipo"] = "Serie"

if 'presupuesto' not in series_df.columns:
    series_df['presupuesto'] = np.nan

df = pd.concat([pelis_df, series_df], ignore_index=True)
df = df.drop_duplicates(subset=['id_contenido'])

# Escala de Presupuesto ($M USD)
df['presupuesto_m'] = (df['presupuesto'] / 1_000_000).round(2)

# Eficiencia de Tracción = Popularidad / Presupuesto Estimado ($M)
df['eficiencia_traccion'] = np.where(
    df['presupuesto_m'] > 0,
    (df['popularidad'] / df['presupuesto_m']).round(2),
    np.nan
)

# Tiers de Presupuesto para Adquisición
def categorizar_tier_costo(p_m):
    if pd.isna(p_m) or p_m == 0: return "Indefinido / Licencia Estándar"
    elif p_m < 15: return "Bajo Costo (<$15M)"
    elif p_m <= 60: return "Costo Medio ($15M-$60M)"
    else: return "Blockbuster (> $60M)"

df['tier_costo'] = df['presupuesto_m'].apply(categorizar_tier_costo)

# Categorías de Rating
def categorizar_rating(rating):
    if pd.isna(rating): return "Sin Rating"
    elif rating >= 8.0: return "Excelente / Premium (8.0-10)"
    elif rating >= 6.5: return "Buena / Aceptable (6.5-7.9)"
    elif rating >= 5.0: return "Regular (5.0-6.4)"
    else: return "Baja / Riesgo (<5.0)"

df['tier_calificacion'] = df['calificacion_promedio'].apply(categorizar_rating)

# Extraer género principal
df['genero_principal'] = df['generos'].astype(str).apply(
    lambda x: x.split(',')[0].strip() if pd.notna(x) and x != 'nan' else "Desconocido"
)

# Mapeo de idiomas
IDIOMA_MAP = {
    'en': 'Inglés', 'zh': 'Chino', 'ja': 'Japonés', 'ko': 'Coreano',
    'es': 'Español', 'fr': 'Francés', 'de': 'Alemán', 'hi': 'Hindi',
    'tl': 'Tagalo', 'pt': 'Portugués', 'ru': 'Ruso', 'it': 'Italiano',
    'tr': 'Turco', 'ar': 'Árabe', 'nl': 'Holandés', 'pl': 'Polaco'
}
df['idioma_nombre'] = df['idioma'].map(IDIOMA_MAP).fillna(df['idioma'].astype(str).str.upper())

GENEROS_OPCIONES = sorted(df['genero_principal'].dropna().unique().tolist())
TIPOS_OPCIONES = sorted(df['tipo'].dropna().unique().tolist())
TIERS_COSTO_OPCIONES = ["Bajo Costo (<$15M)", "Costo Medio ($15M-$60M)", "Blockbuster (> $60M)", "Indefinido / Licencia Estándar"]
ANIOS_OPCIONES = sorted([int(y) for y in df['anio_estreno'].dropna().unique()], reverse=True)

# --- 3. PALETA Y ESTILOS DE DISEÑO ---
COLOR_MAP = {"Película": "#38BDF8", "Serie": "#A855F7"}

BACKGROUND = "#0F172A"
CARD_BG    = "#1E293B"
TEXT       = "#F8FAFC"
SUBTEXT    = "#94A3B8"
ACCENT     = "#38BDF8"
BORDER     = "#334155"

LAYOUT_BASE = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT, family="Segoe UI, Roboto, sans-serif"),
    margin=dict(l=35, r=35, t=45, b=35),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER)
)

def build_kpi_card(titulo, valor, subtexto, accent_color="#38BDF8"):
    return html.Div([
        html.Div(style={
            "height": "4px", "backgroundColor": accent_color,
            "borderRadius": "8px 8px 0 0", "margin": "-12px -14px 8px -14px"
        }),
        html.H4(titulo, style={"margin": "0", "fontSize": "11px", "color": SUBTEXT, "textTransform": "uppercase", "fontWeight": "600"}),
        html.H2(valor, style={"margin": "4px 0 2px 0", "color": TEXT, "fontSize": "20px", "fontWeight": "700"}),
        html.Span(subtexto, style={"fontSize": "10px", "color": SUBTEXT, "display": "block"})
    ], style={"backgroundColor": CARD_BG, "padding": "12px 14px", "borderRadius": "8px", "border": f"1px solid {BORDER}"})

def build_info_badge(label, value, color="#38BDF8"):
    return html.Div([
        html.Span(label, style={"color": SUBTEXT, "fontSize": "11px", "fontWeight": "500"}),
        html.Span(value, style={"color": color, "fontSize": "12px", "fontWeight": "700", "float": "right"})
    ], style={"padding": "6px 0", "borderBottom": f"1px solid {BORDER}"})

def create_section_header(title, subtitle=""):
    return html.Div([
        html.H3(title, style={"fontSize": "16px", "color": ACCENT, "margin": "0 0 4px 0", "fontWeight": "700"}),
        html.P(subtitle, style={"fontSize": "12px", "color": SUBTEXT, "margin": "0 0 14px 0"}) if subtitle else None
    ])

# --- 4. INICIALIZACIÓN DASH ---
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "StreamView - Gerencia de Adquisición de Contenidos"

app.layout = html.Div(
    style={"backgroundColor": BACKGROUND, "minHeight": "100vh", "fontFamily": "Segoe UI, Roboto, sans-serif", "color": TEXT, "padding": "20px 28px"},
    children=[
        # Encabezado
        html.Div([
            html.H1("🎬 STREAMVIEW ANALYTICS | ADQUISICIÓN Y GESTIÓN DE LICENCIAS", style={"margin": "0", "color": TEXT, "fontSize": "22px", "fontWeight": "800"}),
            html.P("Plataforma de inteligencia para evaluar paquetes de licencias, optimizar la retención de suscriptores y balancear la oferta audiovisual", style={"margin": "4px 0 0 0", "color": SUBTEXT, "fontSize": "13px"}),
        ], style={"borderBottom": f"1px solid {BORDER}", "paddingBottom": "12px", "marginBottom": "16px"}),

        # Panel Global de Controles
        html.Div([
            html.Div([
                html.Label("Formato de Contenido:", style={"fontSize": "11px", "fontWeight": "600", "color": SUBTEXT, "marginBottom": "4px", "display": "block"}),
                dcc.Checklist(id="filter-type", options=[{"label": f" {t}", "value": t} for t in TIPOS_OPCIONES], value=TIPOS_OPCIONES, inline=True, style={"color": TEXT, "fontSize": "12px"})
            ], style={"width": "180px"}),

            html.Div([
                html.Label("Géneros Principales:", style={"fontSize": "11px", "fontWeight": "600", "color": SUBTEXT, "marginBottom": "4px", "display": "block"}),
                dcc.Dropdown(id="filter-genre", options=[{"label": g, "value": g} for g in GENEROS_OPCIONES], value=GENEROS_OPCIONES, multi=True, placeholder="Todos los géneros", style={"backgroundColor": CARD_BG, "color": "#000", "fontSize": "12px"})
            ], style={"flex": "1", "minWidth": "200px"}),

            html.Div([
                html.Label("Tier de Costo / Escala:", style={"fontSize": "11px", "fontWeight": "600", "color": SUBTEXT, "marginBottom": "4px", "display": "block"}),
                dcc.Dropdown(id="filter-tier-costo", options=[{"label": c, "value": c} for c in TIERS_COSTO_OPCIONES], value=[], multi=True, placeholder="Todos los tiers", style={"backgroundColor": CARD_BG, "color": "#000", "fontSize": "12px"})
            ], style={"flex": "1", "minWidth": "200px"}),

            html.Div([
                html.Label("Año de Estreno:", style={"fontSize": "11px", "fontWeight": "600", "color": SUBTEXT, "marginBottom": "4px", "display": "block"}),
                dcc.RangeSlider(id="filter-years", min=min(ANIOS_OPCIONES), max=max(ANIOS_OPCIONES), step=1, value=[min(ANIOS_OPCIONES), max(ANIOS_OPCIONES)], marks={y: {"label": str(y), "style": {"color": SUBTEXT, "fontSize": "10px"}} for y in range(min(ANIOS_OPCIONES), max(ANIOS_OPCIONES)+1, 3)})
            ], style={"width": "280px"}),
        ], style={"backgroundColor": CARD_BG, "padding": "12px 16px", "borderRadius": "8px", "border": f"1px solid {BORDER}", "display": "flex", "gap": "16px", "alignItems": "center", "flexWrap": "wrap", "marginBottom": "20px"}),

        # Tabs Navegación
        dcc.Tabs(
            id="main-tabs", value="tab-summary",
            colors={"border": BORDER, "primary": ACCENT, "background": CARD_BG},
            children=[
                dcc.Tab(label="📌 Informes Generales & KPIs", value="tab-summary", style={"padding": "10px"}, selected_style={"padding": "10px", "color": ACCENT, "backgroundColor": "#162032"}),
                dcc.Tab(label="📐 Estructura de Oferta & Géneros", value="tab-composition", style={"padding": "10px"}, selected_style={"padding": "10px", "color": ACCENT, "backgroundColor": "#162032"}),
                dcc.Tab(label="🌍 Cobertura Regional e Idiomas", value="tab-geo", style={"padding": "10px"}, selected_style={"padding": "10px", "color": ACCENT, "backgroundColor": "#162032"}),
                dcc.Tab(label="⚡ Matriz de Adquisición & Eficiencia", value="tab-efficiency", style={"padding": "10px"}, selected_style={"padding": "10px", "color": ACCENT, "backgroundColor": "#162032"}),
                dcc.Tab(label="📋 Explorador de Licencias", value="tab-explorer", style={"padding": "10px"}, selected_style={"padding": "10px", "color": ACCENT, "backgroundColor": "#162032"}),
            ],
            style={"marginBottom": "20px"}
        ),

        html.Div(id="tab-content")
    ]
)

# --- 5. CALLBACKS ---
@app.callback(
    Output("tab-content", "children"),
    [
        Input("main-tabs", "value"),
        Input("filter-type", "value"),
        Input("filter-genre", "value"),
        Input("filter-tier-costo", "value"),
        Input("filter-years", "value")
    ]
)
def render_tab_content(tab, selected_types, selected_genres, selected_tiers, selected_years):
    d = df[
        (df["tipo"].isin(selected_types)) &
        (df["genero_principal"].isin(selected_genres if selected_genres else GENEROS_OPCIONES)) &
        (df["anio_estreno"] >= selected_years[0]) &
        (df["anio_estreno"] <= selected_years[1])
    ]
    if selected_tiers:
        d = d[d["tier_costo"].isin(selected_tiers)]

    total_titulos = len(d)
    num_pelis = len(d[d['tipo'] == 'Película'])
    num_series = len(d[d['tipo'] == 'Serie'])

    # ─────────────────────────────────────────────────────────────────
    # TAB 1: INFORMACIÓN GENERAL & KPIS EJECUTIVOS (REDISENADA)
    # ─────────────────────────────────────────────────────────────────
    if tab == "tab-summary":
        avg_rating = d['calificacion_promedio'].mean() if total_titulos > 0 else 0.0
        premium_count = len(d[d['calificacion_promedio'] >= 7.5])
        premium_pct = (premium_count / total_titulos * 100) if total_titulos > 0 else 0.0
        avg_pop = d['popularidad'].mean() if total_titulos > 0 else 0.0
        
        d_cost = d[d['eficiencia_traccion'].notna()]
        med_eficiencia = d_cost['eficiencia_traccion'].median() if not d_cost.empty else 0.0
        
        top_genre = d['genero_principal'].value_counts().index[0] if total_titulos > 0 else "N/A"
        paises_count = len(set(p.strip() for sublist in d['pais'].dropna().astype(str).str.split(',') for p in sublist if p.strip() and p.strip() != 'nan'))
        idiomas_count = d['idioma_nombre'].nunique()

        # Tarjetas de KPIs Principales
        kpis_grid = html.Div([
            build_kpi_card("Volumen Total Licenciado", f"{total_titulos:,}", f"Base de catálogo evaluada", "#38BDF8"),
            build_kpi_card("Puntuación de Audiencia", f"{avg_rating:.2f} / 10", f"Promedio global de valoración", "#A855F7"),
            build_kpi_card("Ratio Retención Premium", f"{premium_pct:.1f}%", f"{premium_count:,} títulos con Rating ≥ 7.5", "#10B981"),
            build_kpi_card("Índice Tracción / Costo", f"{med_eficiencia:.2f} pts/$M", "Popularidad por escala de presupuesto", "#10B981"),
            build_kpi_card("Índice de Popularidad", f"{avg_pop:.1f} pts", f"Género dominante: {top_genre}", "#F59E0B"),
            build_kpi_card("Cobertura Global", f"{paises_count} Países", f"{idiomas_count} Idiomas disponibles", "#EC4899"),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "12px", "marginBottom": "20px"})

        # Desglose de Información General (Quick Facts Panels)
        info_panel = html.Div([
            html.Div([
                html.H4("🎞️ Mezcla de Formatos (Engagement)", style={"fontSize": "13px", "color": ACCENT, "margin": "0 0 10px 0", "fontWeight": "700"}),
                build_info_badge("Películas Disponibles", f"{num_pelis:,} ({num_pelis/total_titulos*100:.1f}%)" if total_titulos>0 else "0", "#38BDF8"),
                build_info_badge("Series Episódicas", f"{num_series:,} ({num_series/total_titulos*100:.1f}%)" if total_titulos>0 else "0", "#A855F7"),
                build_info_badge("Relación Películas / Series", f"{(num_pelis/num_series):.2f}x" if num_series>0 else "N/A", TEXT),
            ], style={"backgroundColor": CARD_BG, "padding": "14px 16px", "borderRadius": "8px", "border": f"1px solid {BORDER}", "flex": "1", "minWidth": "250px"}),

            html.Div([
                html.H4("💵 Escala Presupuestaria de Licencias", style={"fontSize": "13px", "color": ACCENT, "margin": "0 0 10px 0", "fontWeight": "700"}),
                build_info_badge("Bajo Costo (<$15M)", f"{len(d[d['tier_costo']=='Bajo Costo (<$15M)'])} títulos", "#10B981"),
                build_info_badge("Costo Medio ($15M-$60M)", f"{len(d[d['tier_costo']=='Costo Medio ($15M-$60M)'])} títulos", "#F59E0B"),
                build_info_badge("Blockbusters (> $60M)", f"{len(d[d['tier_costo']=='Blockbuster (> $60M)'])} títulos", "#38BDF8"),
            ], style={"backgroundColor": CARD_BG, "padding": "14px 16px", "borderRadius": "8px", "border": f"1px solid {BORDER}", "flex": "1", "minWidth": "250px"}),

            html.Div([
                html.Div([
                    html.H4("🌟 Calidad y Retención Anti-Churn", style={"fontSize": "13px", "color": ACCENT, "margin": "0 0 10px 0", "fontWeight": "700"}),
                    build_info_badge("Excelencia / Premium (8.0 - 10)", f"{len(d[d['calificacion_promedio']>=8.0]):,} títulos", "#10B981"),
                    build_info_badge("Calidad Aceptable (6.5 - 7.9)", f"{len(d[(d['calificacion_promedio']>=6.5)&(d['calificacion_promedio']<8.0)]):,} títulos", "#38BDF8"),
                    build_info_badge("Riesgo / Descarte (< 5.0)", f"{len(d[d['calificacion_promedio']<5.0]):,} títulos", "#EF4444"),
                ])
            ], style={"backgroundColor": CARD_BG, "padding": "14px 16px", "borderRadius": "8px", "border": f"1px solid {BORDER}", "flex": "1", "minWidth": "250px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "20px"})

        # Gráfico Único y Principal: Balance por Género y Formato
        fig_mix = px.histogram(
            d, x="genero_principal", color="tipo", barmode="group",
            title="<b>Balance del Catálogo: Distribución de Volumen por Género Principal y Formato (Películas vs. Series)</b>",
            color_discrete_map=COLOR_MAP,
            labels={"genero_principal": "Género Principal", "count": "Número de Títulos"}
        )
        fig_mix.update_layout(**LAYOUT_BASE)

        return html.Div([
            create_section_header("Resumen Ejecutivo de Información General", "Diagnóstico integral del catálogo activo en la plataforma para guiar las decisiones de la Gerencia de Adquisición."),
            kpis_grid,
            info_panel,
            dcc.Graph(figure=fig_mix),
            html.P("📌 Diagnóstico Estratégico: Muestra la densidad de la oferta por categoría. La Gerencia debe orientar las nuevas compras a balancear los géneros donde existe una brecha de series episódicas (p. ej. Acción y Terror), clave para el consumo continuo.", style={"fontSize": "11px", "color": SUBTEXT, "marginTop": "6px"})
        ])

    # ─────────────────────────────────────────────────────────────────
    # TAB 2: ESTRUCTURA Y BOXPLOT DE NIVELES DE CALIFICACIÓN (MEJORADA)
    # ─────────────────────────────────────────────────────────────────
    elif tab == "tab-composition":
        fig_tree = px.treemap(d, path=['genero_principal', 'tipo'], title="<b>Densidad de Catálogo por Género y Formato (Treemap)</b>", color='tipo', color_discrete_map=COLOR_MAP)
        fig_tree.update_layout(**LAYOUT_BASE)

        # 1. Calcular mediana de calificación por género y ordenar de mayor a menor
        mediana_por_genero = d.groupby("genero_principal")["calificacion_promedio"].median().sort_values(ascending=False)
        generos_ordenados = mediana_por_genero.index.tolist()

        # 2. Asignar Nivel de Calificación Media
        def categorizar_nivel_media(med):
            if med >= 6.3: return "🟢 Nivel Superior (Mediana ≥ 6.3)"
            elif med >= 5.5: return "🔵 Nivel Aceptable (5.5 - 6.2)"
            else: return "🔴 Nivel Bajo / Riesgo (< 5.5)"

        d_box = d.copy()
        d_box["nivel_calificacion_media"] = d_box["genero_principal"].map(lambda g: categorizar_nivel_media(mediana_por_genero[g]))

        COLOR_BOX_TIERS = {
            "🟢 Nivel Superior (Mediana ≥ 6.3)": "#10B981", # Verde Esmeralda
            "🔵 Nivel Aceptable (5.5 - 6.2)": "#38BDF8",   # Azul Celeste
            "🔴 Nivel Bajo / Riesgo (< 5.5)": "#EF4444"     # Rojo
        }

        # 3. Crear Boxplot Ordenado y Agrupado por Niveles
        fig_box = px.box(
            d_box,
            x="genero_principal",
            y="calificacion_promedio",
            color="nivel_calificacion_media",
            color_discrete_map=COLOR_BOX_TIERS,
            category_orders={"genero_principal": generos_ordenados},
            labels={
                "genero_principal": "Género Principal (Ordenado por Calificación Mediana)", 
                "calificacion_promedio": "Calificación Promedio (Rating)",
                "nivel_calificacion_media": "Nivel de Calificación Media"
            },
            title="<b>Consistencia y Niveles de Calificación Media por Género (Agrupados por Desempeño)</b>"
        )
        fig_box.update_layout(**LAYOUT_BASE)

        # Líneas horizontales de umbral de calidad
        fig_box.add_hline(
            y=7.0, line_dash="dash", line_color="#F59E0B", line_width=1.5,
            annotation_text=" Umbral Rating Premium (7.0)", annotation_position="top left",
            annotation_font=dict(color="#F59E0B", size=10)
        )
        fig_box.add_hline(
            y=5.5, line_dash="dash", line_color="#94A3B8", line_width=1.5,
            annotation_text=" Umbral Mínimo Aceptable (5.5)", annotation_position="bottom left",
            annotation_font=dict(color="#94A3B8", size=10)
        )

        return html.Div([
            create_section_header("Evaluación de Categorías y Niveles de Calificación", "Análisis de la densidad de la oferta y desglose de la consistencia de calidad percibida por género."),
            dcc.Graph(figure=fig_tree),
            html.P("📌 Justificación: Muestra la proporción relativa de cada género en la biblioteca para evitar sobre-contratar géneros saturados.", style={"fontSize": "11px", "color": SUBTEXT, "marginBottom": "20px"}),
            dcc.Graph(figure=fig_box),
            html.P("📌 Hallazgos Claros en Boxplot: Los géneros destacados en verde (a la izquierda) mantienen niveles medianos superiores a 6.3 de rating con bajo riesgo de descontento (p. ej. Animación, Crimen). Los géneros en rojo muestran alta volatilidad y requieren filtros estrictos de entrada.", style={"fontSize": "11px", "color": SUBTEXT})
        ])

    # ─────────────────────────────────────────────────────────────────
    # TAB 3: COBERTURA REGIONAL E IDIOMAS
    # ─────────────────────────────────────────────────────────────────
    elif tab == "tab-geo":
        paises_exploded = d.dropna(subset=['pais']).copy()
        paises_exploded['pais_split'] = paises_exploded['pais'].astype(str).str.split(',')
        paises_exploded = paises_exploded.explode('pais_split')
        paises_exploded['pais_split'] = paises_exploded['pais_split'].str.strip()
        paises_counts = paises_exploded['pais_split'].value_counts().reset_index()
        paises_counts.columns = ['pais', 'cantidad']

        fig_map = px.choropleth(paises_counts, locations="pais", locationmode="country names", color="cantidad", hover_name="pais", color_continuous_scale="Viridis", title="Origen Geográfico del Contenido Licenciado")
        fig_map.update_layout(**LAYOUT_BASE)
        fig_map.update_geos(bgcolor=CARD_BG, showcoastlines=True, coastlinecolor=BORDER, showland=True, landcolor="#162032")

        top_idiomas = d['idioma_nombre'].value_counts().head(6)
        otros_count = d['idioma_nombre'].value_counts().iloc[6:].sum() if len(d['idioma_nombre'].value_counts()) > 6 else 0
        df_lang = pd.DataFrame({'idioma': list(top_idiomas.index) + (['Otros'] if otros_count > 0 else []), 'cantidad': list(top_idiomas.values) + ([otros_count] if otros_count > 0 else [])})

        fig_donut = px.pie(df_lang, names="idioma", values="cantidad", hole=0.5, title="Participación Idiomática (Gráfico de Dona)", color_discrete_sequence=px.colors.qualitative.Dark24)
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        fig_donut.update_layout(**LAYOUT_BASE)

        return html.Div([
            create_section_header("Mapeo de Origen Regional e Idiomas", "Planificación de compras regionales y localización de contenido para expansión de suscriptores internacionales."),
            html.Div([
                html.Div([dcc.Graph(figure=fig_map), html.P("📌 Justificación: Permite detectar mercados emergentes (p. ej. Corea del Sur, España) para cerrar acuerdos de exclusividad regional.", style={"fontSize": "11px", "color": SUBTEXT})], style={"flex": "1.2", "minWidth": "450px"}),
                html.Div([dcc.Graph(figure=fig_donut), html.P("📌 Justificación: Orienta la asignación del presupuesto de doblaje/subtitulado según la cuota de contenido extranjero.", style={"fontSize": "11px", "color": SUBTEXT})], style={"flex": "0.8", "minWidth": "350px"}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"})
        ])

    # ─────────────────────────────────────────────────────────────────
    # TAB 4: MATRIZ DE ADQUISICIÓN Y EFICIENCIA (VISIBILIDAD MEJORADA)
    # ─────────────────────────────────────────────────────────────────
    elif tab == "tab-efficiency":
        # Panel Explicativo de Definición de Cuadrantes
        definition_panel = html.Div([
            html.H4("🎯 Guía de Interpretación de los 4 Cuadrantes de Decisión", style={"fontSize": "13px", "color": ACCENT, "margin": "0 0 8px 0", "fontWeight": "700"}),
            html.Div([
                html.Div([
                    html.Span("💎 JOYAS OCULTAS / MAX ROI:", style={"color": "#10B981", "fontWeight": "700", "fontSize": "11px"}),
                    html.Span(" Alto Rating / Popularidad Moderada o Bajo Costo. Prioridad de compra a bajo costo para fidelizar.", style={"fontSize": "11px", "color": TEXT})
                ], style={"flex": "1", "minWidth": "220px"}),

                html.Div([
                    html.Span("🌟 HITS MASIVOS / BLOCKBUSTERS:", style={"color": "#38BDF8", "fontWeight": "700", "fontSize": "11px"}),
                    html.Span(" Alto Rating / Alta Popularidad o Alto Presupuesto. Renovación prioritaria como contenido ancla.", style={"fontSize": "11px", "color": TEXT})
                ], style={"flex": "1", "minWidth": "220px"}),

                html.Div([
                    html.Span("🔥 ÉXITOS VIRALES:", style={"color": "#F59E0B", "fontWeight": "700", "fontSize": "11px"}),
                    html.Span(" Rating Moderado / Alta Popularidad. Licencias temporales útiles para la captación inicial.", style={"fontSize": "11px", "color": TEXT})
                ], style={"flex": "1", "minWidth": "220px"}),

                html.Div([
                    html.Span("⚠️ ZONA DE RIESGO / INEFICIENCIA:", style={"color": "#EF4444", "fontWeight": "700", "fontSize": "11px"}),
                    html.Span(" Bajo Rating / Baja Popularidad o Alto Costo con Tracción Pobre. Candidatos a no-renovación.", style={"fontSize": "11px", "color": TEXT})
                ], style={"flex": "1", "minWidth": "220px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"})
        ], style={"backgroundColor": CARD_BG, "padding": "12px 16px", "borderRadius": "8px", "border": f"1px solid {BORDER}", "marginBottom": "20px"})

        # MATRIZ 1: POPULARIDAD VS RATING
        threshold_y1 = 7.0   # Umbral Rating Premium
        threshold_x1 = 35.0  # Umbral Alta Popularidad

        fig_quadrant1 = px.scatter(
            d,
            x="popularidad",
            y="calificacion_promedio",
            color="tipo",
            hover_name="titulo",
            hover_data={"genero_principal": True, "anio_estreno": True, "popularidad": ":.1f", "calificacion_promedio": ":.2f"},
            color_discrete_map=COLOR_MAP,
            labels={"popularidad": "Índice de Popularidad (Tracción)", "calificacion_promedio": "Calificación Promedio (Rating)"},
            title="<b>MATRIZ 1: SELECCIÓN DE ADQUISICIÓN (POPULARIDAD VS. RATING)</b>",
            opacity=0.75
        )
        fig_quadrant1.update_layout(**LAYOUT_BASE)
        fig_quadrant1.update_xaxes(type="log", dtick=1, range=[0, 3.5])  # Escala logarítmica limpia (1, 10, 100, 1000)
        fig_quadrant1.update_yaxes(range=[0, 10.5])

        # Umbrales
        fig_quadrant1.add_hline(y=threshold_y1, line_dash="dash", line_color="#F59E0B", line_width=2, annotation_text=" Umbral Rating Premium (7.0)", annotation_position="top left", annotation_font=dict(color="#F59E0B", size=11))
        fig_quadrant1.add_vline(x=threshold_x1, line_dash="dash", line_color="#38BDF8", line_width=2, annotation_text="Umbral Pop. Alta (35 pts) ", annotation_position="top right", annotation_font=dict(color="#38BDF8", size=11))

        # Cuadrantes
        fig_quadrant1.add_shape(type="rect", x0=threshold_x1, x1=5000, y0=threshold_y1, y1=10.5, fillcolor="rgba(56, 189, 248, 0.12)", line=dict(color="#38BDF8", width=1), layer="below")
        fig_quadrant1.add_shape(type="rect", x0=1, x1=threshold_x1, y0=threshold_y1, y1=10.5, fillcolor="rgba(16, 185, 129, 0.15)", line=dict(color="#10B981", width=1), layer="below")
        fig_quadrant1.add_shape(type="rect", x0=1, x1=threshold_x1, y0=0, y1=threshold_y1, fillcolor="rgba(239, 68, 68, 0.10)", line=dict(color="#EF4444", width=1), layer="below")
        fig_quadrant1.add_shape(type="rect", x0=threshold_x1, x1=5000, y0=0, y1=threshold_y1, fillcolor="rgba(245, 158, 11, 0.12)", line=dict(color="#F59E0B", width=1), layer="below")

        # Anotaciones
        fig_quadrant1.add_annotation(x=0.8, y=9.8, xref="x", yref="y", text="<b>💎 JOYAS OCULTAS</b><br><i>Alto Rating / Pop. Moderada</i><br><span style='font-size:10px;color:#A7F3D0'>Prioridad Compras (Bajo Costo)</span>", showarrow=False, align="left", bgcolor="rgba(6, 78, 59, 0.9)", bordercolor="#10B981", borderwidth=1.5, borderpad=6, font=dict(color="#FFFFFF", size=11))
        fig_quadrant1.add_annotation(x=2000, y=9.8, xref="x", yref="y", text="<b>🌟 HITS MASIVOS</b><br><i>Alto Rating / Alta Pop.</i><br><span style='font-size:10px;color:#BAE6FD'>Renovación Prioritaria (Ancla)</span>", showarrow=False, align="right", bgcolor="rgba(12, 74, 110, 0.9)", bordercolor="#38BDF8", borderwidth=1.5, borderpad=6, font=dict(color="#FFFFFF", size=11))
        fig_quadrant1.add_annotation(x=2000, y=1.2, xref="x", yref="y", text="<b>🔥 ÉXITOS VIRALES</b><br><i>Rating Mod. / Alta Pop.</i><br><span style='font-size:10px;color:#FDE68A'>Captación Temporal</span>", showarrow=False, align="right", bgcolor="rgba(120, 53, 15, 0.9)", bordercolor="#F59E0B", borderwidth=1.5, borderpad=6, font=dict(color="#FFFFFF", size=11))
        fig_quadrant1.add_annotation(x=0.8, y=1.2, xref="x", yref="y", text="<b>⚠️ ZONA DE RIESGO</b><br><i>Rating Mod. / Pop. Baja</i><br><span style='font-size:10px;color:#FCA5A5'>Evaluar Descarte / No Renovar</span>", showarrow=False, align="left", bgcolor="rgba(127, 29, 29, 0.9)", bordercolor="#EF4444", borderwidth=1.5, borderpad=6, font=dict(color="#FFFFFF", size=11))

        # MATRIZ 2: PRESUPUESTO VS POPULARIDAD
        d_cost = d[d['presupuesto_m'] > 0].copy()
        threshold_y2 = 35.0  # Popularidad
        threshold_x2 = 30.0  # Presupuesto $30M

        fig_quadrant2 = px.scatter(
            d_cost,
            x="presupuesto_m",
            y="popularidad",
            size="calificacion_promedio",
            color="tipo",
            hover_name="titulo",
            hover_data={"genero_principal": True, "presupuesto_m": ":.1f", "popularidad": ":.1f", "calificacion_promedio": ":.2f"},
            size_max=18,
            color_discrete_map=COLOR_MAP,
            labels={"presupuesto_m": "Presupuesto Estimado ($M USD)", "popularidad": "Índice de Popularidad (Tracción)"},
            title="<b>MATRIZ 2: EFICIENCIA FINANCIERA (PRESUPUESTO VS. POPULARIDAD)</b>",
            opacity=0.8
        )
        fig_quadrant2.update_layout(**LAYOUT_BASE)
        fig_quadrant2.update_xaxes(type="log", dtick=1, range=[-0.5, 2.7])
        fig_quadrant2.update_yaxes(type="log", dtick=1, range=[0.5, 3.5])

        # Umbrales
        fig_quadrant2.add_hline(y=threshold_y2, line_dash="dash", line_color="#38BDF8", line_width=2, annotation_text=" Umbral Tracción Requerida (35 pts)", annotation_position="top left", annotation_font=dict(color="#38BDF8", size=11))
        fig_quadrant2.add_vline(x=threshold_x2, line_dash="dash", line_color="#F59E0B", line_width=2, annotation_text="Umbral Alto Costo ($30M USD) ", annotation_position="top right", annotation_font=dict(color="#F59E0B", size=11))

        # Cuadrantes
        fig_quadrant2.add_shape(type="rect", x0=0.1, x1=threshold_x2, y0=threshold_y2, y1=5000, fillcolor="rgba(16, 185, 129, 0.15)", line=dict(color="#10B981", width=1), layer="below")
        fig_quadrant2.add_shape(type="rect", x0=threshold_x2, x1=1000, y0=threshold_y2, y1=5000, fillcolor="rgba(56, 189, 248, 0.12)", line=dict(color="#38BDF8", width=1), layer="below")
        fig_quadrant2.add_shape(type="rect", x0=0.1, x1=threshold_x2, y0=1, y1=threshold_y2, fillcolor="rgba(148, 163, 184, 0.10)", line=dict(color="#64748B", width=1), layer="below")
        fig_quadrant2.add_shape(type="rect", x0=threshold_x2, x1=1000, y0=1, y1=threshold_y2, fillcolor="rgba(239, 68, 68, 0.12)", line=dict(color="#EF4444", width=1), layer="below")

        # Anotaciones
        fig_quadrant2.add_annotation(x=0.2, y=2000, xref="x", yref="y", text="<b>⚡ ALTA EFICIENCIA / MAX ROI</b><br><i>Bajo Presupuesto / Alta Pop.</i><br><span style='font-size:10px;color:#A7F3D0'>Máxima prioridad de adquisición</span>", showarrow=False, align="left", bgcolor="rgba(6, 78, 59, 0.9)", bordercolor="#10B981", borderwidth=1.5, borderpad=6, font=dict(color="#FFFFFF", size=11))
        fig_quadrant2.add_annotation(x=500, y=2000, xref="x", yref="y", text="<b>🏰 BLOCKBUSTERS CONSOLIDADOS</b><br><i>Alto Presupuesto / Alta Pop.</i><br><span style='font-size:10px;color:#BAE6FD'>Inversión Justificada (Marketing)</span>", showarrow=False, align="right", bgcolor="rgba(12, 74, 110, 0.9)", bordercolor="#38BDF8", borderwidth=1.5, borderpad=6, font=dict(color="#FFFFFF", size=11))
        fig_quadrant2.add_annotation(x=500, y=2.0, xref="x", yref="y", text="<b>🛑 INEFICIENCIA FINANCIERA</b><br><i>Alto Presupuesto / Pop. Mod.</i><br><span style='font-size:10px;color:#FCA5A5'>Evitar / Re-negociar a la baja</span>", showarrow=False, align="right", bgcolor="rgba(127, 29, 29, 0.9)", bordercolor="#EF4444", borderwidth=1.5, borderpad=6, font=dict(color="#FFFFFF", size=11))
        fig_quadrant2.add_annotation(x=0.2, y=2.0, xref="x", yref="y", text="<b>🌱 CONTENIDO NATIVO DE NICHO</b><br><i>Bajo Presupuesto / Pop. Mod.</i><br><span style='font-size:10px;color:#CBD5E1'>Completar catálogo específico</span>", showarrow=False, align="left", bgcolor="rgba(51, 65, 85, 0.9)", bordercolor="#64748B", borderwidth=1.5, borderpad=6, font=dict(color="#FFFFFF", size=11))

        # Gráfico de Tendencia Histórica
        trend_df = d.groupby(['anio_estreno', 'tipo']).size().reset_index(name='cantidad')
        fig_trend = px.line(trend_df, x="anio_estreno", y="cantidad", color="tipo", color_discrete_map=COLOR_MAP, title="<b>Evolución Histórica de Títulos en la Biblioteca por Año de Estreno</b>")
        fig_trend.update_layout(**LAYOUT_BASE)

        return html.Div([
            create_section_header("Matriz de Adquisición y Eficiencia de Licencias", "Herramientas analíticas de 4 cuadrantes para priorizar compras según poder de atracción, retención y costo estimado."),
            definition_panel,
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_quadrant1),
                    html.P("📌 Diagnóstico Matriz 1: Permite aislar 'Joyas Ocultas' (alto rating, popularidad moderada) para negociar licencias accesibles con máximo valor de retención.", style={"fontSize": "11px", "color": SUBTEXT, "marginTop": "4px"})
                ], style={"flex": "1", "minWidth": "480px"}),
                
                html.Div([
                    dcc.Graph(figure=fig_quadrant2),
                    html.P("📌 Diagnóstico Matriz 2: Mapea si los títulos de alto presupuesto realmente generan tracción proporcional antes de renovar contratos de alto costo.", style={"fontSize": "11px", "color": SUBTEXT, "marginTop": "4px"})
                ], style={"flex": "1", "minWidth": "480px"}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "20px"}),
            
            dcc.Graph(figure=fig_trend),
            html.P("📌 Control de Contratos: Mapeo del ciclo de antigüedad para programar renovaciones antes del vencimiento de derechos.", style={"fontSize": "11px", "color": SUBTEXT})
        ])

    # ─────────────────────────────────────────────────────────────────
    # TAB 5: EXPLORADOR GRANULAR
    # ─────────────────────────────────────────────────────────────────
    elif tab == "tab-explorer":
        cols_tabla = [
            'titulo', 'tipo', 'anio_estreno', 'genero_principal',
            'calificacion_promedio', 'popularidad', 'pais',
            'idioma_nombre', 'tier_costo', 'eficiencia_traccion'
        ]
        records = d[cols_tabla].fillna("N/A").to_dict("records")

        return html.Div([
            create_section_header("Matriz Granular de Evaluación de Licencias por Título", "Tabla interactiva para mesas de negociación directas con distribuidores y estudios."),
            dash_table.DataTable(
                data=records,
                columns=[
                    {"name": "Título", "id": "titulo"},
                    {"name": "Tipo", "id": "tipo"},
                    {"name": "Año Estreno", "id": "anio_estreno"},
                    {"name": "Género Principal", "id": "genero_principal"},
                    {"name": "Rating", "id": "calificacion_promedio"},
                    {"name": "Popularidad", "id": "popularidad"},
                    {"name": "País", "id": "pais"},
                    {"name": "Idioma", "id": "idioma_nombre"},
                    {"name": "Escala de Costo", "id": "tier_costo"},
                    {"name": "Tracción/Costo (pts/$M)", "id": "eficiencia_traccion"},
                ],
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                page_action="native",
                page_size=12,
                style_table={"overflowX": "auto", "borderRadius": "8px", "border": f"1px solid {BORDER}"},
                style_header={"backgroundColor": "#0F172A", "color": ACCENT, "fontWeight": "bold", "border": f"1px solid {BORDER}", "fontSize": "12px"},
                style_cell={"backgroundColor": CARD_BG, "color": TEXT, "border": f"1px solid {BORDER}", "padding": "8px 10px", "textAlign": "left", "fontSize": "11px"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#162032"}]
            )
        ])

if __name__ == "__main__":
    app.run(debug=True)