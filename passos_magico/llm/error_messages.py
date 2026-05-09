"""Mensagens curtas em PT-BR para erros de SQL / DuckDB (evitar jargão cru na UI)."""

from __future__ import annotations

import re


def humanize_sql_execution_error(message: str) -> str:
    """
    Resumo amigável para o gestor. O texto técnico completo pode ir num expander à parte.
    """
    if not message or not str(message).strip():
        return "Não consegui concluir a análise sobre a base carregada. Tente reformular com ano, fase ou indicador (ex.: INDE, IDA)."

    m = str(message)
    low = m.lower()

    if "query vazia" in low:
        return "A consulta veio vazia — vale repetir a pergunta com um pouco mais de detalhe (ano, fase ou indicador)."

    if "apenas consultas select" in low or "only select" in low:
        return "Só são permitidas leituras seguras na base (tipo **consulta**). Se apareceu outro tipo de comando, foi bloqueado de propósito."

    if "uma única instrução" in low or "single statement" in low:
        return "Só pode ir **uma** consulta de cada vez. Evite colar vários comandos seguidos."

    if "palavra-chave não permitida" in low or "not permitted" in low:
        return "A consulta usou uma palavra-chave que não é permitida neste painel (só leitura dos dados)."

    if "parênteses desbalanceados" in low or "incompleto" in low and "sql" in low:
        return "A consulta gerada parece **incompleta** (estrutura quebrada). Vale tentar de novo com uma pergunta mais simples ou filtros explícitos (Ano, Fase)."

    if "referenced column" in low or ("column" in low and "not found" in low):
        col = _extract_quoted_identifier(low, r'referenced column\s+"([^"]+)"')
        if not col:
            col = _extract_quoted_identifier(low, r"referenced column\s+'([^']+)'")
        if col:
            return (
                f"A consulta citou o campo **`{col}`**, que **não existe** na base carregada. "
                "Use só nomes de colunas que aparecem no dicionário ou na lista enviada ao modelo."
            )
        return (
            "A consulta citou um **nome de campo** que **não existe** na base carregada. "
            "Confira o dicionário de dados ou peça uma métrica com colunas conhecidas (Ano, Fase, Turma, INDE, IDA, …)."
        )

    if "group by clause cannot contain aggregates" in low or "cannot contain aggregates" in low:
        return "A consulta misturou **média/contagem** dentro do **agrupamento** de forma inválida. Peça uma média por fase ou por ano, sem agregado dentro do `GROUP BY`."

    if "binder error" in low:
        if "date_part" in low or "year(" in low or "varchar" in low:
            return (
                "Há um **tipo de dado** incompatível (muitas vezes **data em texto**). "
                "Datas como `data_nasc` precisam de conversão (`TRY_CAST(... AS DATE)`) antes de tirar o ano."
            )
        if "integer" in low and "date" in low:
            return (
                "Não dá para misturar **ano número** com **data** diretamente (ex.: `Ano - data`). "
                "Use **ano de nascimento** com `year(TRY_CAST(data_nasc AS DATE))` e subtraia do **Ano** da ficha."
            )
        return "O motor de dados **não aceitou** a forma da consulta (Binder). Tente uma pergunta mais simples ou veja o detalhe técnico abaixo."

    if "parser error" in low or "syntax error" in low:
        return "A consulta tem um **erro de sintaxe**. Reformule a pergunta ou peça só uma tabela simples (ex.: média de INDE por Ano)."

    if "token" in low and "não corresponde a coluna" in low:
        return m  # já vem amigável do guard interno

    return (
        "Não consegui **executar** a consulta sobre a base carregada. "
        "Se for nome de campo, confira o dicionário; senão, simplifique a pergunta (um indicador, um ano)."
    )


def _extract_quoted_identifier(low: str, pattern: str) -> str | None:
    r = re.search(pattern, low, re.IGNORECASE)
    return r.group(1) if r else None
