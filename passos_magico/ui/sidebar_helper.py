"""Fallback para reabrir a sidebar quando o ícone do cabeçalho não responde ao clique."""

from __future__ import annotations

import time

_OPEN_SCRIPT = """
<script>
(function () {
  function clickSidebarControl(doc) {
    var selectors = [
      '[data-testid="collapsedControl"]',
      '[data-testid="stSidebarCollapsedControl"]',
      'button[aria-label="Open sidebar"]',
      'button[aria-label="Close sidebar"]',
      'button[aria-label*="sidebar" i]',
      'button[title*="sidebar" i]'
    ];
    var i, el;
    for (i = 0; i < selectors.length; i++) {
      el = doc.querySelector(selectors[i]);
      if (el) { el.click(); return true; }
    }
    var hdr = doc.querySelector('header[data-testid="stHeader"]');
    if (hdr) {
      var buttons = hdr.querySelectorAll('button');
      if (buttons.length) { buttons[0].click(); return true; }
    }
    return false;
  }
  try {
    var roots = [window.parent, window.top, window];
    var r, d;
    for (r = 0; r < roots.length; r++) {
      try {
        d = roots[r].document;
        if (d && clickSidebarControl(d)) return;
      } catch (e) {}
    }
  } catch (err) {}
})();
</script>
"""


def render_open_sidebar_button() -> None:
    """Botão na área principal que dispara o mesmo clique do controle nativo da sidebar."""
    import streamlit as st
    import streamlit.components.v1 as components

    c1, _ = st.columns([1, 6])
    with c1:
        if st.button(
            "☰ Menu",
            key="pm_open_sidebar_btn",
            help="Abre ou foca o painel lateral (mesmo efeito do ícone no topo).",
        ):
            st.session_state["_pm_open_sidebar_js"] = True

    if st.session_state.pop("_pm_open_sidebar_js", False):
        # O Streamlit reaproveita o iframe quando o HTML é idêntico — o <script> só roda na 1ª vez.
        # Mudamos o markup a cada clique para forçar nova execução.
        bump = int(time.time_ns() % 10**12)
        unique_html = (
            f'<span data-pm-nonce="{bump}" style="display:none"></span>'
            f"<!-- pm-open-sidebar {bump} -->"
            + _OPEN_SCRIPT
        )
        components.html(unique_html, height=1, width=1)
