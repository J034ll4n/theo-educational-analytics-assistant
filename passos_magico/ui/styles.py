"""CSS global: dark mode e ajustes de layout (acento #EE145B)."""

_PM_ACCENT = "#EE145B"
_PM_ACCENT_RGB = "238, 20, 91"
# Brilhos estilo banner (magenta + ciano)
_PM_SPARKLE_PINK = "255, 0, 127"
_PM_SPARKLE_CYAN = "0, 191, 255"

# Fundo discreto (sem “estrelas” — leitura mais simples)
_PM_BOKEH_MAIN = f"""
        radial-gradient(ellipse 80% 50% at 20% 20%, rgba({_PM_SPARKLE_PINK}, 0.06) 0%, transparent 50%),
        radial-gradient(ellipse 70% 45% at 85% 75%, rgba({_PM_SPARKLE_CYAN}, 0.05) 0%, transparent 50%)
"""

_PM_HERO_CYAN = "#00d4d8"
_PM_HERO_PINK = "#ff2da0"

_PM_BOKEH_SIDEBAR = f"""
        radial-gradient(ellipse 100% 60% at 50% 30%, rgba({_PM_SPARKLE_PINK}, 0.05) 0%, transparent 55%)
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
    /*
     * Não ocultar a toolbar inteira: em Streamlit recente o botão ☰ da sidebar fica aqui;
     * esconder só o que for claramente extra (deploy / menu cloud).
     */
    div[data-testid="stToolbar"] {{
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
    }}
    div[data-testid="stToolbar"] a[href*="streamlit.io/cloud"],
    div[data-testid="stToolbar"] button[title="Deploy"],
    div[data-testid="stToolbar"] [data-testid="stToolbarDeployButton"] {{
        display: none !important;
    }}
    /* Não ocultar <footer> inteiro: no Streamlit recente a barra de chat pode ficar lá — só esconde link “Made with Streamlit” */
    footer a[href*="streamlit.io"] {{display: none !important;}}
    /* Evita que decorações cubram o clique no header */
    div[data-testid="stDecoration"] {{
        pointer-events: none !important;
    }}
    /* Botão da sidebar (☰) — header, toolbar e controles colapsados */
    [data-testid="collapsedControl"],
    button[data-testid="baseButton-header"],
    header[data-testid="stHeader"] button[kind="header"],
    header[data-testid="stHeader"] [data-testid="stHeaderActionElements"] button,
    div[data-testid="stToolbar"] button {{
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
    header[data-testid="stHeader"] svg,
    div[data-testid="stToolbar"] svg {{
        fill: #e6edf3 !important;
        color: #e6edf3 !important;
        opacity: 1 !important;
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
    /*
     * Altura mínima do contentor — sem forçar display/flex nos filhos do Streamlit (isso partia o layout).
     */
    div[data-testid="stAppViewContainer"] {{
        min-height: 100vh;
        min-height: 100dvh;
    }}
    /* Área principal: scroll vertical (deixa o motor de layout nativo do Streamlit intacto) */
    section[data-testid="stMain"] {{
        overflow-x: hidden;
        overflow-y: auto;
        -webkit-overflow-scrolling: touch;
        min-height: 0;
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
        position: relative;
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.25rem;
    }}
    /* Fallback st.image na sidebar (sem data URI) */
    section[data-testid="stSidebar"] div[data-testid="stImage"] img {{
        max-height: 220px;
        opacity: 0.95;
    }}
    section[data-testid="stSidebar"] .pm-sidebar-brand-layout {{
        margin: 0 0 1.15rem 0;
        width: 100%;
    }}
    section[data-testid="stSidebar"] .pm-sidebar-brand-stage {{
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        width: 100%;
        padding: 0.25rem 0 0.5rem 0;
        overflow: visible;
    }}
    section[data-testid="stSidebar"] .pm-sidebar-brand-visual {{
        position: relative;
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        max-width: 100%;
        flex-shrink: 0;
        margin-bottom: 0.95rem;
    }}
    section[data-testid="stSidebar"] .pm-sidebar-slogan-hero {{
        display: block;
        width: 100%;
        max-width: min(100%, 560px);
        height: auto;
        max-height: clamp(22rem, 104vmin, 31rem);
        margin: 0 auto;
        object-fit: contain;
        object-position: center center;
        opacity: 1;
        filter: none;
    }}
    section[data-testid="stSidebar"] .pm-sidebar-hands-glow {{
        position: absolute;
        left: 50%;
        top: 52%;
        transform: translate(-50%, -50%);
        width: min(42%, 240px);
        height: min(42%, 240px);
        border-radius: 50%;
        background: radial-gradient(
            circle at 50% 50%,
            rgba(255, 46, 147, 0.42) 0%,
            rgba(0, 229, 255, 0.28) 45%,
            transparent 72%
        );
        filter: blur(20px);
        z-index: 1;
        pointer-events: none;
        opacity: 0.75;
    }}
    section[data-testid="stSidebar"] .pm-sidebar-brand-copy {{
        position: relative;
        z-index: 0;
        text-align: center;
        width: 100%;
        padding: 0 0.25rem 0.15rem 0.25rem;
    }}
    section[data-testid="stSidebar"]
        .pm-sidebar-brand-visual
        ~ .pm-sidebar-brand-copy.pm-brand-hero {{
        margin-top: 0;
    }}
    /* Texto da marca alinhado ao centro da imagem */
    .pm-sidebar-brand-text {{
        text-align: center;
        max-width: 100%;
    }}
    .pm-sidebar-brand-text .pm-branded-kicker,
    .pm-sidebar-brand-text .pm-sidebar-hint,
    .pm-sidebar-brand-text .pm-branded-subkicker {{
        text-align: center;
        margin-left: auto;
        margin-right: auto;
    }}
    .pm-branded-kicker {{
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        color: rgba(255, 255, 255, 0.92);
        text-transform: uppercase;
        margin: 0 0 0.5rem 0;
    }}
    .pm-brand-hero {{
        margin-top: 0.2rem;
    }}
    section[data-testid="stSidebar"] .pm-sidebar-app-title {{
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        color: #f0f3f6;
        text-align: center;
        margin: 0.35rem 0 0.25rem 0;
        line-height: 1.25;
    }}
    section[data-testid="stSidebar"] .pm-sidebar-tagline {{
        font-size: 0.82rem;
        font-weight: 500;
        line-height: 1.45;
        color: rgba(240, 243, 246, 0.72);
        margin: 0.35rem auto 0.65rem auto;
        max-width: 18rem;
        letter-spacing: 0.02em;
    }}
    section[data-testid="stSidebar"] .pm-branded-subkicker {{
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: rgba(201, 209, 217, 0.72);
        margin: 0.15rem 0 0.45rem 0;
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
    /* Rótulo do menu: use label_visibility no st.radio — não esconda stWidgetLabel com :has()
       (em versões novas do Streamlit isso quebrava rótulos ou o próprio menu). */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {{
        gap: 0.5rem;
        flex-direction: column;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radio"] {{
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
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radio"]:hover {{
        border-color: rgba({_PM_ACCENT_RGB}, 0.45) !important;
        background: rgba(30, 22, 32, 0.92) !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked),
    section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radio"][aria-checked="true"] {{
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
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:last-child,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radio"] > div:last-child {{
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
    section[data-testid="stMain"] div[data-testid="stMetricValue"] {{
        font-size: 1.75rem;
    }}
    section[data-testid="stMain"] div[data-testid="stSelectbox"] label {{
        font-size: 0.85rem !important;
        color: #8b949e !important;
    }}
    /*
     * Barra de chat: afastar do fundo da janela (respiro inferior).
     * Valores modestos para não interferir com position:fixed em algumas versões.
     */
    footer {{
        padding-bottom: 1rem !important;
    }}
    div[data-testid="stBottom"] {{
        bottom: 0.85rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        padding-bottom: 0.35rem !important;
    }}
    div[data-testid="stChatInput"] {{
        margin-bottom: 0.35rem !important;
    }}
    div[data-testid="stVerticalBlockFooter"] {{
        padding-bottom: 0.75rem !important;
    }}
    section[data-testid="stMain"] .block-container {{
        padding-bottom: 4rem;
    }}
    /*
     * Abas (st.tabs): o Base Web às vezes deixa painéis ou blocos internos com opacidade < 1,
     * o que faz a «Matriz de priorização» parecer apagada. Forçar leitura normal no conteúdo das abas.
     */
    section[data-testid="stMain"] div[data-testid="stTabs"] [role="tabpanel"],
    section[data-testid="stMain"] div[data-testid="stTabs"] [data-baseweb="tab-panel"] {{
        opacity: 1 !important;
    }}
    section[data-testid="stMain"] div[data-testid="stTabs"] [role="tabpanel"] > div,
    section[data-testid="stMain"] div[data-testid="stTabs"] [data-baseweb="tab-panel"] > div {{
        opacity: 1 !important;
    }}
    section[data-testid="stMain"] div[data-testid="stTabs"] div[data-testid="stVerticalBlock"],
    section[data-testid="stMain"] div[data-testid="stTabs"] div[data-testid="stHorizontalBlock"] {{
        opacity: 1 !important;
    }}
    /* Página Previsão de risco — só na área principal (evita conflito com sidebar) */
    section[data-testid="stMain"] .pm-risk-page-header {{
        font-size: 0.92rem;
        line-height: 1.5;
        color: #c9d1d9;
        padding: 0.75rem 0.9rem;
        border-radius: 8px;
        border: 1px solid rgba(48, 54, 61, 0.95);
        background: rgba(22, 27, 34, 0.75);
        margin-bottom: 0.75rem;
    }}
    section[data-testid="stMain"] .pm-risk-page-header-lead {{
        margin: 0 0 0.75rem 0;
        color: #e6edf3;
    }}
    section[data-testid="stMain"] .pm-risk-page-header-lead strong {{
        color: #f0f3f6;
    }}
    section[data-testid="stMain"] .pm-risk-page-bullets {{
        margin: 0;
        padding-left: 1.15rem;
        color: #c9d1d9;
    }}
    section[data-testid="stMain"] .pm-risk-page-bullets li {{
        margin: 0 0 0.45rem 0;
        line-height: 1.5;
    }}
    section[data-testid="stMain"] .pm-risk-page-bullets li:last-child {{
        margin-bottom: 0;
    }}
    section[data-testid="stMain"] .pm-risk-section-sub {{
        font-size: 0.88rem;
        line-height: 1.45;
        color: #8b949e;
        margin: -0.15rem 0 0.95rem 0;
        max-width: 52rem;
    }}
    section[data-testid="stMain"] .pm-risk-section-tight-top {{
        margin-top: 0.6rem !important;
    }}
    section[data-testid="stMain"] .pm-risk-sim-column-lead {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: #8b949e;
        margin: 0 0 0.55rem 0;
    }}
    section[data-testid="stMain"] .pm-risk-intro {{
        font-size: 0.92rem;
        line-height: 1.5;
        color: #c9d1d9;
        padding: 0.75rem 0.9rem;
        border-radius: 8px;
        border: 1px solid rgba(48, 54, 61, 0.95);
        background: rgba(22, 27, 34, 0.75);
        margin-bottom: 0.75rem;
    }}
    section[data-testid="stMain"] .pm-risk-intro strong {{
        color: #f0f3f6;
    }}
    section[data-testid="stMain"] .pm-risk-step {{
        display: inline-block;
        min-width: 1.35rem;
        text-align: center;
        font-weight: 800;
        font-size: 0.72rem;
        padding: 0.15rem 0.45rem;
        border-radius: 6px;
        background: rgba({_PM_ACCENT_RGB}, 0.35);
        color: #fff;
        margin-right: 0.35rem;
    }}
    section[data-testid="stMain"] .pm-risk-hero {{
        padding: 0.85rem 1rem;
        border-radius: 8px;
        border: 1px solid rgba(48, 54, 61, 0.95);
        background: rgba(22, 27, 34, 0.8);
        margin-bottom: 0.85rem;
    }}
    section[data-testid="stMain"] .pm-risk-pct {{
        font-size: clamp(2.4rem, 5vw, 3.1rem);
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }}
    section[data-testid="stMain"] .pm-risk-bandtag {{
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        margin-top: 0.35rem;
    }}
    section[data-testid="stMain"] .pm-risk-zones {{
        display: flex;
        gap: 0.5rem;
        margin-top: 0.85rem;
    }}
    section[data-testid="stMain"] .pm-risk-zone {{
        flex: 1;
        text-align: center;
        font-size: 0.72rem;
        line-height: 1.35;
        padding: 0.55rem 0.35rem;
        border-radius: 8px;
        border: 1px solid rgba(48, 54, 61, 0.95);
        color: #8b949e;
    }}
    section[data-testid="stMain"] .pm-risk-zone span {{
        display: block;
        font-weight: 700;
        font-size: 0.68rem;
        color: #e6edf3;
        margin-bottom: 0.15rem;
    }}
    section[data-testid="stMain"] .pm-risk-zone.pm-active {{
        border-width: 2px;
    }}
    section[data-testid="stMain"] .pm-risk-section-title-muted {{
        font-size: 0.98rem !important;
        font-weight: 700 !important;
        color: #e6edf3 !important;
        margin: 1rem 0 0.35rem 0 !important;
        padding-bottom: 0;
        border-bottom: none;
    }}
    section[data-testid="stMain"] .pm-risk-section-sub-muted {{
        font-size: 0.82rem;
        line-height: 1.4;
        color: #8b949e;
        margin: 0 0 0.65rem 0;
        max-width: 48rem;
    }}
    section[data-testid="stMain"] .pm-risk-section-title {{
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #f0f3f6 !important;
        margin: 1.25rem 0 0.5rem 0 !important;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid rgba({_PM_ACCENT_RGB}, 0.2);
    }}
    section[data-testid="stMain"] .pm-risk-section-title.pm-risk-compare-title {{
        margin-top: 1.5rem !important;
    }}
    section[data-testid="stMain"] .pm-risk-sim-guide {{
        font-size: 0.88rem;
        line-height: 1.45;
        color: #e6edf3;
        padding: 0.65rem 0.85rem;
        margin: 0 0 0.75rem 0;
        border-radius: 8px;
        border: 1px solid rgba(48, 54, 61, 0.95);
        background: rgba(22, 27, 34, 0.75);
    }}
    section[data-testid="stMain"] .pm-risk-sim-guide strong {{
        color: #fff;
    }}
    section[data-testid="stMain"] .pm-risk-context-readonly {{
        font-size: 0.88rem;
        line-height: 1.5;
        color: #c9d1d9;
        padding: 0.65rem 0.85rem;
        margin: 0 0 0.75rem 0;
        border-radius: 8px;
        border: 1px solid rgba(48, 54, 61, 0.95);
        background: rgba(18, 22, 28, 0.75);
    }}
    section[data-testid="stMain"] .pm-risk-context-readonly strong {{
        color: #f0f3f6;
    }}
    section[data-testid="stMain"] .pm-risk-context-readonly span {{
        color: #e6edf3;
        font-weight: 600;
    }}
    section[data-testid="stMain"] .pm-risk-pedagogy-card {{
        padding: 0.75rem 0.9rem;
        border-radius: 8px;
        border: 1px solid rgba(48, 54, 61, 0.95);
        background: rgba(18, 22, 28, 0.75);
        min-height: auto;
    }}
    section[data-testid="stMain"] .pm-risk-pedagogy-card h4 {{
        margin: 0.35rem 0 0.65rem 0;
        font-size: 1rem;
        font-weight: 700;
        color: #f0f3f6;
    }}
    section[data-testid="stMain"] .pm-risk-pedagogy-badge {{
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin: 0 0 0.25rem 0;
    }}
    section[data-testid="stMain"] .pm-risk-ped-badge-ctx {{
        color: {_PM_HERO_CYAN};
    }}
    section[data-testid="stMain"] .pm-risk-ped-badge-traj {{
        color: {_PM_HERO_PINK};
    }}
    section[data-testid="stMain"] .pm-risk-pedagogy-metric {{
        font-size: 0.88rem;
        color: #aeb7c2;
        margin: 0 0 0.65rem 0;
        line-height: 1.45;
    }}
    section[data-testid="stMain"] .pm-risk-pedagogy-card p {{
        font-size: 0.88rem;
        line-height: 1.5;
        color: #c9d1d9;
        margin: 0 0 0.5rem 0;
    }}
    section[data-testid="stMain"] .pm-risk-delta-box {{
        font-size: 0.9rem;
        line-height: 1.45;
        color: #c9d1d9;
        padding: 0.65rem 0.85rem;
        min-height: 5.5rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border-radius: 10px;
        border: 1px solid rgba(48, 54, 61, 0.95);
        background: rgba(18, 22, 28, 0.85);
    }}
    section[data-testid="stMain"] .pm-risk-panel-title {{
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: rgba(201, 209, 217, 0.95);
        margin: 0.85rem 0 0.45rem 0;
    }}
    section[data-testid="stMain"] .pm-risk-panel-title:first-of-type {{
        margin-top: 0.15rem;
    }}
    section[data-testid="stMain"] .pm-risk-threshold-critical {{
        font-size: 0.92rem;
        line-height: 1.45;
        padding: 0.55rem 0.75rem;
        border-radius: 10px;
        margin-bottom: 0.65rem;
        border: 1px solid rgba(248, 81, 73, 0.55);
        background: rgba(248, 81, 73, 0.12);
        color: #f0f3f6;
    }}
    section[data-testid="stMain"] .pm-risk-threshold-ok {{
        font-size: 0.92rem;
        line-height: 1.45;
        padding: 0.55rem 0.75rem;
        border-radius: 10px;
        margin-bottom: 0.65rem;
        border: 1px solid rgba(63, 185, 80, 0.45);
        background: rgba(63, 185, 80, 0.1);
        color: #f0f3f6;
    }}
    section[data-testid="stMain"] .pm-risk-sim-pct {{
        letter-spacing: -0.02em;
    }}
    /* Dashboards (classes legadas — páginas novas usam st.title nativo) */
    section[data-testid="stMain"] .pm-dash-hero {{
        padding: 0.75rem 0.9rem;
        border-radius: 8px;
        border: 1px solid rgba(48, 54, 61, 0.95);
        background: rgba(22, 27, 34, 0.75);
        margin-bottom: 0.75rem;
    }}
    section[data-testid="stMain"] .pm-dash-hero h1 {{
        font-size: 1.35rem;
        font-weight: 700;
        color: #f0f3f6;
        margin: 0 0 0.35rem 0;
    }}
    section[data-testid="stMain"] .pm-dash-hero p {{
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.45;
        color: #aeb7c2;
    }}
    section[data-testid="stMain"] .pm-dash-kpi-title {{
        font-size: 0.85rem;
        font-weight: 600;
        color: #8b949e;
        margin: 0 0 0.5rem 0;
    }}
    section[data-testid="stMain"] .pm-dash-section-title {{
        font-size: 1rem;
        font-weight: 600;
        color: #f0f3f6;
        margin: 0.35rem 0 0.65rem 0;
    }}
    @media (max-width: 768px) {{
        section[data-testid="stMain"] [data-testid="column"] {{
            min-width: 100% !important;
        }}
    }}
</style>
"""


def inject_global_css() -> None:
    import streamlit as st

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
