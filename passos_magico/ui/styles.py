"""CSS global: dark mode e ajustes de layout (acento #EE145B)."""

_PM_ACCENT = "#EE145B"
_PM_ACCENT_RGB = "238, 20, 91"
# Brilhos estilo banner (magenta + ciano)
_PM_SPARKLE_PINK = "255, 0, 127"
_PM_SPARKLE_CYAN = "0, 191, 255"

# Manchas suaves + “estrelas” (radial em posições fixas); primeira camada = topo
_PM_BOKEH_MAIN = f"""
        radial-gradient(circle at 11% 14%, rgba({_PM_SPARKLE_PINK}, 0.5) 0, transparent 1.5px),
        radial-gradient(circle at 23% 38%, rgba({_PM_SPARKLE_CYAN}, 0.42) 0, transparent 1.2px),
        radial-gradient(circle at 67% 12%, rgba({_PM_SPARKLE_PINK}, 0.35) 0, transparent 1px),
        radial-gradient(circle at 88% 22%, rgba({_PM_SPARKLE_CYAN}, 0.38) 0, transparent 1.4px),
        radial-gradient(circle at 15% 62%, rgba({_PM_SPARKLE_CYAN}, 0.45) 0, transparent 1.3px),
        radial-gradient(circle at 42% 71%, rgba({_PM_SPARKLE_PINK}, 0.4) 0, transparent 1.1px),
        radial-gradient(circle at 54% 48%, rgba({_PM_SPARKLE_CYAN}, 0.32) 0, transparent 1px),
        radial-gradient(circle at 78% 55%, rgba({_PM_SPARKLE_PINK}, 0.48) 0, transparent 1.6px),
        radial-gradient(circle at 91% 68%, rgba({_PM_SPARKLE_CYAN}, 0.36) 0, transparent 1.2px),
        radial-gradient(circle at 7% 84%, rgba({_PM_SPARKLE_PINK}, 0.38) 0, transparent 1px),
        radial-gradient(circle at 33% 91%, rgba({_PM_SPARKLE_CYAN}, 0.4) 0, transparent 1.4px),
        radial-gradient(circle at 61% 88%, rgba({_PM_SPARKLE_PINK}, 0.33) 0, transparent 1.1px),
        radial-gradient(circle at 95% 91%, rgba({_PM_SPARKLE_CYAN}, 0.42) 0, transparent 1.3px),
        radial-gradient(circle at 48% 19%, rgba({_PM_SPARKLE_PINK}, 0.28) 0, transparent 1px),
        radial-gradient(circle at 72% 33%, rgba({_PM_SPARKLE_CYAN}, 0.3) 0, transparent 1px),
        radial-gradient(ellipse 90% 55% at 18% 28%, rgba({_PM_SPARKLE_PINK}, 0.09) 0%, transparent 55%),
        radial-gradient(ellipse 85% 60% at 82% 72%, rgba({_PM_SPARKLE_CYAN}, 0.08) 0%, transparent 52%),
        radial-gradient(ellipse 70% 45% at 50% 50%, rgba({_PM_SPARKLE_PINK}, 0.05) 0%, transparent 50%)
"""

_PM_HERO_CYAN = "#00d4d8"
_PM_HERO_PINK = "#ff2da0"

_PM_BOKEH_SIDEBAR = f"""
        radial-gradient(circle at 20% 20%, rgba({_PM_SPARKLE_PINK}, 0.32) 0, transparent 1.2px),
        radial-gradient(circle at 75% 35%, rgba({_PM_SPARKLE_CYAN}, 0.3) 0, transparent 1.1px),
        radial-gradient(circle at 45% 70%, rgba({_PM_SPARKLE_PINK}, 0.28) 0, transparent 1px),
        radial-gradient(circle at 88% 82%, rgba({_PM_SPARKLE_CYAN}, 0.34) 0, transparent 1.2px),
        radial-gradient(ellipse 100% 70% at 50% 40%, rgba({_PM_SPARKLE_PINK}, 0.06) 0%, transparent 55%),
        radial-gradient(ellipse 80% 50% at 30% 75%, rgba({_PM_SPARKLE_CYAN}, 0.05) 0%, transparent 48%)
"""

GLOBAL_CSS = f"""
<style>
    #MainMenu {{visibility: hidden;}}
    header[data-testid="stHeader"] {{
        visibility: visible !important;
        display: block !important;
        position: relative !important;
        z-index: 100002 !important;
        pointer-events: auto !important;
        background: linear-gradient(
            180deg,
            rgba(10, 12, 18, 0.98) 0%,
            rgba(14, 10, 14, 0.94) 100%
        );
        border-bottom: 1px solid rgba({_PM_ACCENT_RGB}, 0.22);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
    }}
    header[data-testid="stHeader"] * {{
        pointer-events: auto !important;
    }}
    div[data-testid="stToolbar"] {{visibility: hidden;}}
    /* Não ocultar <footer> inteiro: no Streamlit recente a barra de chat pode ficar lá — só esconde link “Made with Streamlit” */
    footer a[href*="streamlit.io"] {{display: none !important;}}
    /* Evita que decorações cubram o clique no header */
    div[data-testid="stDecoration"] {{
        pointer-events: none !important;
    }}
    /* Botão flutuante quando a sidebar está recolhida */
    [data-testid="collapsedControl"],
    button[data-testid="baseButton-header"],
    header[data-testid="stHeader"] button[kind="header"] {{
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        z-index: 100003 !important;
    }}
    [data-testid="collapsedControl"] {{
        background: rgba(22, 27, 34, 0.95) !important;
        border: 1px solid rgba({_PM_ACCENT_RGB}, 0.4) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    }}
    [data-testid="collapsedControl"]:hover {{
        border-color: rgba({_PM_ACCENT_RGB}, 0.7) !important;
    }}
    .stApp {{
        background-color: #121218;
        background-image:
            {_PM_BOKEH_MAIN},
            linear-gradient(
                165deg,
                #080a0e 0%,
                #0f0a10 28%,
                #121018 55%,
                #0c0e14 100%
            );
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #e6edf3;
    }}
    section[data-testid="stMain"] {{
        overflow-x: hidden;
    }}
    section[data-testid="stSidebar"] {{
        background-color: #0c080f;
        background-image:
            {_PM_BOKEH_SIDEBAR},
            linear-gradient(
                165deg,
                #120818 0%,
                #180a14 38%,
                #0c0810 100%
            );
        background-attachment: fixed;
        background-repeat: no-repeat;
        border-right: 1px solid rgba({_PM_ACCENT_RGB}, 0.28);
        box-shadow: 4px 0 28px rgba(0, 0, 0, 0.4);
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.25rem;
    }}
    .pm-branded-kicker {{
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        color: rgba(255, 255, 255, 0.92);
        text-transform: uppercase;
        margin: 0 0 0.5rem 0;
    }}
    .pm-branded-slogan {{
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        line-height: 1.35;
        margin: 0 0 0.85rem 0;
        color: #f0f3f6;
    }}
    .pm-branded-slogan .pm-hero-magia,
    .pm-branded-slogan .pm-hero-passo {{
        color: {_PM_HERO_PINK};
        font-weight: 800;
    }}
    .pm-branded-slogan .pm-hero-inov {{
        color: {_PM_HERO_CYAN};
        font-weight: 800;
    }}
    .pm-sidebar-hint {{
        font-size: 0.72rem;
        line-height: 1.4;
        color: #7d7d8c;
        margin: 0 0 1rem 0;
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba({_PM_ACCENT_RGB}, 0.12);
    }}
    .pm-sidebar-hint strong {{
        color: {_PM_ACCENT};
        font-weight: 600;
    }}
    .pm-ollama-pill {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.45rem 0.65rem;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-bottom: 1rem;
        border: 1px solid rgba({_PM_ACCENT_RGB}, 0.22);
        background: rgba(22, 27, 34, 0.85);
    }}
    .pm-ollama-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }}
    .pm-ollama-ok {{ background: #3fb950; box-shadow: 0 0 10px rgba(63, 185, 80, 0.55); }}
    .pm-ollama-off {{ background: #f85149; box-shadow: 0 0 10px rgba(248, 81, 73, 0.45); }}
    /* Sem título "Navegação" acima do radio (fallback se o Streamlit ainda pintar o rótulo) */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stRadio"])
        [data-testid="stWidgetLabel"] {{
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
        clip: rect(0, 0, 0, 0) !important;
        position: absolute !important;
        width: 1px !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {{
        gap: 0.5rem;
        flex-direction: column;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {{
        display: flex !important;
        align-items: center;
        padding: 0.65rem 0.75rem !important;
        margin: 0 !important;
        border-radius: 10px !important;
        border: 1px solid rgba(48, 54, 61, 0.95) !important;
        background: rgba(22, 27, 34, 0.65) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
        cursor: pointer;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {{
        border-color: rgba({_PM_ACCENT_RGB}, 0.45) !important;
        background: rgba(30, 22, 32, 0.92) !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {{
        border-color: rgba({_PM_ACCENT_RGB}, 0.75) !important;
        background: linear-gradient(
            135deg,
            rgba({_PM_ACCENT_RGB}, 0.14) 0%,
            rgba(80, 20, 50, 0.1) 100%
        ) !important;
        box-shadow:
            0 0 0 1px rgba({_PM_ACCENT_RGB}, 0.28),
            0 4px 20px rgba(0, 0, 0, 0.35) !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:last-child {{
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #e6edf3 !important;
    }}
    .pm-tool-help {{
        font-size: 0.82rem;
        line-height: 1.45;
        color: #9a9aaa;
        padding: 0.65rem 0.75rem;
        margin-top: 0.35rem;
        margin-bottom: 0.75rem;
        border-radius: 8px;
        border: 1px solid rgba({_PM_ACCENT_RGB}, 0.18);
        background: rgba(13, 17, 23, 0.5);
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 1.75rem;
    }}
    div[data-testid="stSelectbox"] label {{
        font-size: 0.85rem !important;
        color: #8b949e !important;
    }}
    /*
     * Não estilizar stBottom/stChatInput: regras extras costumam quebrar o position:fixed
     * do Streamlit e empurrar a barra para fora da área visível.
     * Só um respiro leve no conteúdo principal para o último balão não colar no rodapé.
     */
    section[data-testid="stMain"] .block-container {{
        padding-bottom: 0.75rem;
    }}
</style>
"""


def inject_global_css() -> None:
    import streamlit as st

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
