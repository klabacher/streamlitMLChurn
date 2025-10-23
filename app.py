
import io
import re
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
#  APP DE INFERÊNCIA — 100% ALINHADO AO NOTEBOOK DE TREINAMENTO
#  Objetivo: prever "Revertido" (0 = não reverte, 1 = reverte)
#  Features usadas NO TREINO (exatamente estas, nesta ordem):
#    ['Estado', 'Ticket (R$)', 'Modalidade de Contrato',
#     'dias_ativo', 'mes_efetivacao', 'dia_semana_efetivacao']
#  Observação: colunas como "Prazo", "Motivo de cancelamento",
#  "Status Financeiro" e meta_* NÃO foram usadas no treino.
#  Portanto, NÃO entram no modelo nesta versão.
# ============================================================

st.set_page_config(
    page_title="Previsão de Reversão — Alinhado ao Treino",
    page_icon="🧠",
    layout="wide",
)

MODEL_PATH = "melhor_modelo_corrigido.pkl"

# Colunas mínimas que o arquivo deve ter para conseguirmos construir as features do modelo
REQUIRED_RAW_COLS = [
    "Data Efetivado",
    "Ticket (R$)",
    "Estado",
    "Modalidade de Contrato",
]

# Exatamente as colunas que alimentam o pipeline salvo no notebook
FEATURES_FOR_MODEL = [
    "Estado",
    "Ticket (R$)",
    "Modalidade de Contrato",
    "dias_ativo",
    "mes_efetivacao",
    "dia_semana_efetivacao",
]


# ---------------------------
# Utilidades
# ---------------------------
def _clean_money_to_float(series: pd.Series) -> pd.Series:
    """
    Converte coluna com valores monetários brasileiros para float.
    Exemplos de entradas válidas: 'R$ 1.234,56', '1234,56', '1.234', 1234.56 etc.
    """
    s = series.astype(str).str.strip()
    s = (
        s.str.replace(r"[R$\s]", "", regex=True)  # remove R$ e espaços
         .str.replace(".", "", regex=False)       # remove separador de milhar
         .str.replace(",", ".", regex=False)      # troca decimal
    )
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")


def _make_tz_naive(dt_series: pd.Series) -> pd.Series:
    """
    Garante que a série datetime fique **tz-naive** (sem timezone).
    Funciona tanto para entradas tz-aware quanto tz-naive.
    """
    try:
        # Se for tz-aware, converte para "sem tz"
        return dt_series.dt.tz_convert(None)
    except Exception:
        try:
            return dt_series.dt.tz_localize(None)
        except Exception:
            return dt_series


def _compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de 'Data Efetivado', cria:
      - dias_ativo (agora - Data Efetivado)
      - mes_efetivacao
      - dia_semana_efetivacao (0=Seg ... 6=Dom)
    """
    # Tenta parse considerando formato BR primeiro; se falhar, tenta ISO
    dt = pd.to_datetime(df["Data Efetivado"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    if dt.isna().any():
        # Tenta novamente sem dayfirst (casos raros), antes de imputar
        dt2 = pd.to_datetime(df.loc[dt.isna(), "Data Efetivado"], errors="coerce", dayfirst=False, infer_datetime_format=True)
        dt.loc[dt.isna()] = dt2

    # Força a série para tz-naive (remove qualquer timezone)
    dt = _make_tz_naive(dt)

    # Imputação de datas inválidas com mediana
    if dt.isna().any():
        if dt.notna().any():
            mediana = dt[dt.notna()].median()
        else:
            # caso extremo: se TODAS as datas forem inválidas, define mediana = hoje (naive)
            mediana = pd.Timestamp(datetime.now())
        dt = dt.fillna(mediana)

    # Usa 'now' tz-naive para evitar erro "tz-naive vs tz-aware"
    now = pd.Timestamp(datetime.now())
    df["dias_ativo"] = (now - dt).dt.days.clip(lower=0).astype(int)
    df["mes_efetivacao"] = dt.dt.month.astype(int)
    df["dia_semana_efetivacao"] = dt.dt.dayofweek.astype(int)
    return df


def _ensure_minimum_schema(df: pd.DataFrame) -> None:
    """Gera erro claro caso as colunas mínimas não estejam presentes."""
    cols = [c.strip() for c in df.columns]
    raw = set(cols)
    missing = [c for c in REQUIRED_RAW_COLS if c not in raw]
    if missing:
        raise ValueError(
            "Arquivo não possui as colunas mínimas obrigatórias: "
            + ", ".join(f"`{c}`" for c in missing)
        )


def _prepare_dataframe_for_model(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Normaliza, valida e cria exatamente as features esperadas pelo modelo treinado.
    Retorna (df_model, avisos).
    """
    avisos: list[str] = []

    # Normaliza nomes (tira espaços extras nas pontas)
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # Valida colunas mínimas
    _ensure_minimum_schema(df)

    # Tipos/correções básicas
    # Estado e Modalidade — strings, preenchendo desconhecidos
    for col in ["Estado", "Modalidade de Contrato"]:
        if col not in df.columns:
            df[col] = "Desconhecido"
            avisos.append(f"Coluna ausente '{col}' criada com 'Desconhecido'.")
        df[col] = df[col].astype(str).replace({"": "Desconhecido"})

    # Ticket (R$) — monetário → float; nulos → mediana
    if "Ticket (R$)" in df.columns:
        df["Ticket (R$)"] = _clean_money_to_float(df["Ticket (R$)"])
        if df["Ticket (R$)"].isna().any():
            med = float(np.nanmedian(df["Ticket (R$)"]))
            df["Ticket (R$)"] = df["Ticket (R$)"].fillna(med)
            avisos.append(f"Valores nulos em 'Ticket (R$)' preenchidos com a mediana ({med:.2f}).")
    else:
        # Cria e preenche com zero; será avisado
        df["Ticket (R$)"] = 0.0
        avisos.append("Coluna 'Ticket (R$)' ausente — criada com 0.0 (verifique o arquivo).")

    # Datas → features temporais
    if "Data Efetivado" not in df.columns:
        # cria coluna vazia e deixa _compute_time_features imputar
        df["Data Efetivado"] = pd.NaT
        avisos.append("Coluna 'Data Efetivado' ausente — dias_ativo/mes/dia_semana imputados a partir de mediana.")
    df = _compute_time_features(df)

    # Seleciona e reordena EXATAMENTE as features que o modelo conhece
    df_model = df[FEATURES_FOR_MODEL].copy()

    # Sanity check final de tipos
    # Numéricos esperados:
    numeric_expected = ["Ticket (R$)", "dias_ativo", "mes_efetivacao", "dia_semana_efetivacao"]
    for col in numeric_expected:
        if not pd.api.types.is_numeric_dtype(df_model[col]):
            try:
                df_model[col] = pd.to_numeric(df_model[col], errors="coerce")
            except Exception:
                pass
            if df_model[col].isna().any():
                med = float(np.nanmedian(df_model[col]))
                df_model[col] = df_model[col].fillna(med)

    # Categóricas esperadas:
    for col in ["Estado", "Modalidade de Contrato"]:
        df_model[col] = df_model[col].astype(str)

    return df_model, avisos


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    try:
        mdl = joblib.load(model_path)
    except FileNotFoundError:
        return None, "Arquivo de modelo não encontrado."
    except Exception as e:
        return None, f"Falha ao carregar o modelo: {e}"
    return mdl, None


@st.cache_data(show_spinner=False)
def read_any_table(uploaded_file) -> pd.DataFrame | None:
    """
    Lê CSV/Excel com heurística robusta (Excel com múltiplas abas = concat).
    """
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            # sep=None => sniff do separador (ponto e vírgula, vírgula, etc.)
            return pd.read_csv(uploaded_file, sep=None, engine="python")
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            xl = pd.ExcelFile(uploaded_file)
            dfs = [pd.read_excel(xl, sh) for sh in xl.sheet_names]
            return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    except Exception:
        return None
    return None


# ---------------------------
# Interface
# ---------------------------
st.title("🧠 Plataforma de Previsão de Reversão (TMB - Churn)")

st.markdown(
    """
**Como funciona:** Este app reproduz **exatamente** as features do notebook de treinamento.  
As colunas necessárias do arquivo de entrada são:
- `Data Efetivado` (data)
- `Ticket (R$)` (valor monetário)
- `Estado` (UF)
- `Modalidade de Contrato` (categórica)
"""
)

model, err = load_model(MODEL_PATH)
if model is None:
    st.error(f"❌ Não foi possível carregar o modelo `{MODEL_PATH}`. {err or ''}")
    st.stop()

with st.sidebar:
    st.markdown("### ⚙️ Configurações de Predição")
    threshold = st.slider(
        "Limiar (threshold) para classificar como **Vai reverter (classe 1)**",
        min_value=0.05, max_value=0.95, value=0.50, step=0.01,
        help=(
            "Em bases desbalanceadas, um limiar ≠ 0.50 pode melhorar o recall/precisão. "
            "Use este controle para ajustar a sensibilidade."
        ),
    )
    st.divider()
    st.markdown("### 📄 Baixe um template de entrada")
    template = pd.DataFrame(
        {c: [] for c in REQUIRED_RAW_COLS}
    )
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        template.to_excel(w, index=False, sheet_name="Template")
    st.download_button("📥 Template (.xlsx)", data=out.getvalue(),
                       file_name="template_previsao.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.header("1) Envie seu arquivo de dados")
file = st.file_uploader("CSV ou Excel (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"], accept_multiple_files=False)

df_raw = None
if file:
    df_raw = read_any_table(file)
    if df_raw is None or df_raw.empty:
        st.error("❌ Não consegui ler o arquivo. Verifique o formato/codificação.")
        st.stop()
    st.success(f"✅ Arquivo lido — {df_raw.shape[0]} linhas × {df_raw.shape[1]} colunas.")
    with st.expander("🔎 Visualizar amostra dos dados", expanded=False):
        st.dataframe(df_raw.head(20))

st.header("2) Processar e Prever")
btn = st.button("🚀 Executar previsão", type="primary", use_container_width=True)

if btn:
    try:
        if df_raw is None:
            st.warning("Envie um arquivo primeiro.")
            st.stop()

        with st.spinner("Preparando dados..."):
            df_for_model, avisos = _prepare_dataframe_for_model(df_raw)
            for a in avisos:
                st.warning(a)

            # Conferência de esquema
            missing_in_model = [c for c in FEATURES_FOR_MODEL if c not in df_for_model.columns]
            if missing_in_model:
                st.error(
                    "Schema inconsistente. Faltam colunas para o modelo: "
                    + ", ".join(f"`{c}`" for c in missing_in_model)
                )
                st.stop()

        with st.spinner("Gerando previsões..."):
            # predict_proba -> prob da classe positiva (índice 1)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df_for_model)[:, 1]
            else:
                # fallback raro (alguns modelos não têm proba)
                # normaliza scores para [0,1] por segurança
                scores = model.decision_function(df_for_model)
                proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

            pred = (proba >= threshold).astype(int)

        df_out = df_raw.copy()
        df_out["Probabilidade_Reversao"] = np.round(proba, 4)
        df_out["Predicao"] = np.where(pred == 1, "Vai reverter", "Não vai reverter")

        st.success("✅ Previsão concluída!")

        # Resumo
        colA, colB, colC = st.columns(3)
        total = len(df_out)
        positivos = int((pred == 1).sum())
        media_prob = float(np.mean(proba)) if len(proba) else 0.0
        colA.metric("Registros analisados", f"{total}")
        colB.metric("Previstos como 'Vai reverter'", f"{positivos}", f"{positivos/total:.1%}")
        # colC.metric("Probabilidade média (classe 1)", f"{media_prob:.1%}")

        # Checagem: se todas as previsões forem 0, alerta de threshold
        if positivos == 0:
            st.warning(
                "Todas as predições ficaram como **'Não vai reverter'**. "
                "Considere ajustar o threshold na barra lateral e/ou revisar os dados de entrada."
            )

        st.divider()
        st.subheader("Resultados")
        st.dataframe(df_out)

        # Downloads
        buff_xlsx = io.BytesIO()
        with pd.ExcelWriter(buff_xlsx, engine="xlsxwriter") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Previsoes")
        st.download_button(
            "📥 Baixar Excel",
            data=buff_xlsx.getvalue(),
            file_name="previsoes_reversao.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        csv_bytes = df_out.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "📄 Baixar CSV",
            data=csv_bytes,
            file_name="previsoes_reversao.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.divider()
        with st.expander("🧪 Debug — Dados enviados ao modelo"):
            st.write("Features e amostra de linhas após o pré-processamento (exatamente como o modelo espera):")
            st.write(FEATURES_FOR_MODEL)
            st.dataframe(df_for_model.head(15))

    except Exception as e:
        st.error("❌ Erro durante a execução.")
        st.exception(e)
