import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import io
import plotly.express as px  # NOVO: Importando a biblioteca para gráficos

# --- Configuração Inicial da Página ---
st.set_page_config(
    page_title="Plataforma de Previsão de Reversão",
    page_icon="🤖",
    layout="wide"
)

# --- CARREGAMENTO DO MODELO (HARDCODED) ---
NOME_ARQUIVO_MODELO = "modelo_v1.pkl"

@st.cache_resource
def carregar_modelo(caminho_modelo):
    try:
        modelo = joblib.load(caminho_modelo)
        return modelo
    except FileNotFoundError:
        return None

modelo = carregar_modelo(NOME_ARQUIVO_MODELO)

# --- LISTA DE COLUNAS CRÍTICAS ---
# Usada para validação e para mostrar ao usuário
COLUNAS_CRITICAS_BASE = [
    'Prazo', 'Produtor', 'Produto', 'Estado', 'Status Financeiro',
    'Ticket (R$)', 'Modalidade de Contrato', 'Data Efetivado'
]

COLUNAS_CRITICAS = [
    'Prazo', 'Produtor', 'Produto', 'Estado', 'Status Financeiro',
    'Ticket (R$)', 'Modalidade de Contrato', 'dias_ativo',
    'mes_efetivacao', 'dia_semana_efetivacao'
]

# --- Funções de Processamento ---

def carregar_dados(uploaded_file):
    df = None
    if uploaded_file.name.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
            st.success(f"✅ Arquivo CSV **{uploaded_file.name}** carregado!")
            return df
        except Exception as e:
            st.error(f"❌ Erro ao ler o arquivo CSV. Detalhe: {e}")
            return None
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        try:
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            if len(sheet_names) > 1:
                option = st.selectbox("Selecione a aba do Excel para usar:", ["**Processar Todas as Abas**"] + sheet_names)
                if option == "**Processar Todas as Abas**":
                    df_list = [pd.read_excel(xls, sheet_name=name) for name in sheet_names]
                    df = pd.concat(df_list, ignore_index=True)
                    st.success(f"✅ Todas as {len(sheet_names)} abas foram combinadas!")
                else:
                    df = pd.read_excel(xls, sheet_name=option)
                    st.success(f"✅ Aba **'{option}'** carregada!")
            else:
                df = pd.read_excel(xls)
                st.success(f"✅ Arquivo Excel **{uploaded_file.name}** carregado!")
            return df
        except Exception as e:
            st.error(f"❌ Erro ao ler o arquivo Excel. Detalhe: {e}")
            return None
    return None

def preparar_dados_robusto(df_original):
    df = df_original.copy()
    colunas_presentes = df.columns.tolist()
    avisos = []

    if ' Ticket (R$) ' in colunas_presentes:
        df.rename(columns={' Ticket (R$) ': 'Ticket (R$)'}, inplace=True)

    colunas_data = ['Criado Em', 'Data Efetivado']
    for col in colunas_data:
        if col in colunas_presentes:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    if 'Data Efetivado' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Data Efetivado']):
        df['dias_ativo'] = (datetime.now() - df['Data Efetivado']).dt.days
        df['mes_efetivacao'] = df['Data Efetivado'].dt.month
        df['dia_semana_efetivacao'] = df['Data Efetivado'].dt.dayofweek
    else:
        avisos.append("Aviso: Features de data ('dias_ativo', etc.) não puderam ser geradas. Verifique a coluna 'Data Efetivado'.")

    # Bloco de verificação de colunas críticas
    colunas_processadas = df.columns.tolist()
    colunas_faltantes = [col for col in COLUNAS_CRITICAS if col not in colunas_processadas]
    if colunas_faltantes:
        mensagem_erro = f"O arquivo enviado não pode ser processado. As seguintes colunas obrigatórias não foram encontradas após o processamento inicial: **{', '.join(colunas_faltantes)}**"
        raise ValueError(mensagem_erro)

    # O pré-processamento continua APÓS a validação
    if 'Ticket (R$)' in df.columns:
        try:
            df['Ticket (R$)'] = df['Ticket (R$)'].astype(str).str.replace('R$ ', '', regex=False).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
        except (AttributeError, ValueError): pass

    return df, avisos

# --- Interface do Streamlit ---

st.title("🤖 Plataforma de Análise e Previsão de Reversão")

if modelo is None:
    st.error(f"❌ **ERRO CRÍTICO:** O arquivo do modelo (`{NOME_ARQUIVO_MODELO}`) não foi encontrado.")
    st.stop()

# NOVO: Seção de instruções sobre as colunas necessárias
with st.expander("⚠️ Clique aqui para ver as colunas necessárias no seu arquivo"):
    st.info("Para que o modelo funcione corretamente, seu arquivo CSV ou Excel precisa conter as seguintes colunas. A ordem não importa, mas os nomes devem ser idênticos.")
    
    # Exibe as colunas em um formato de lista para melhor visualização
    cols_html = "".join([f"<li><code>{col}</code></li>" for col in COLUNAS_CRITICAS_BASE])
    st.markdown(f"<ul>{cols_html}</ul>", unsafe_allow_html=True)
    
st.markdown("---")

# --- Passo 1: Upload do Arquivo de Dados ---
st.header("Passo 1: Carregue o arquivo de dados")
uploaded_data_file = st.file_uploader("Selecione o arquivo CSV ou Excel", type=["csv", "xlsx", "xls"])

df_raw = None
if uploaded_data_file:
    df_raw = carregar_dados(uploaded_data_file)
    if df_raw is not None:
        st.write(f"Resumo: **{df_raw.shape[0]}** linhas e **{df_raw.shape[1]}** colunas.")

# --- Passo 2: Processamento e Previsão ---
st.header("Passo 2: Realizar as Previsões")

if st.button("Executar Previsão", type="primary", use_container_width=True):
    if df_raw is None:
        st.error("❌ **Ação necessária:** Por favor, carregue um arquivo de dados válido no Passo 1.")
    else:
        with st.spinner('Aguarde... Processando dados e realizando previsões...'):
            try:
                df_processed, avisos = preparar_dados_robusto(df_raw.copy())
                
                previsoes = modelo.predict(df_processed)
                probabilidades = modelo.predict_proba(df_processed)[:, 1]

                df_resultados = df_raw.copy()
                df_resultados['Previsão'] = ["Vai reverter" if p == 1 else "Não vai reverter" for p in previsoes]
                # NOVO: Mantém a probabilidade numérica para os gráficos
                df_resultados['Probabilidade_numerica'] = probabilidades
                # Cria a coluna formatada para exibição
                df_resultados['Probabilidade de Reversão'] = df_resultados['Probabilidade_numerica'].apply(lambda p: f"{p * 100:.1f}%")
                
                st.success("✅ Previsões realizadas com sucesso!")
                st.session_state['df_resultados'] = df_resultados

            except ValueError as e:
                st.error(f"❌ **Erro de Validação:** {e}")
            except Exception as e:
                st.error("❌ **Ocorreu um erro inesperado durante a previsão.**")
                st.code(f"Detalhe técnico do erro: {e}")

# --- Exibição e Download dos Resultados ---
if 'df_resultados' in st.session_state:
    df_final = st.session_state['df_resultados']
    
    st.markdown("---")
    st.header("Resultados da Previsão")
    
    # NOVO: Dashboard com Métricas e Gráficos
    st.subheader("Dashboard Resumo")
    
    # Métricas
    total_clientes = len(df_final)
    total_reversao = df_final[df_final['Previsão'] == 'Vai reverter'].shape[0]
    prob_media = df_final['Probabilidade_numerica'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Clientes Analisados", f"{total_clientes}")
    col2.metric("Previsão de Reversão", f"{total_reversao}", f"{total_reversao / total_clientes:.1%} do total")
    col3.metric("Probabilidade Média de Reversão", f"{prob_media:.1%}")

    # Gráficos
    col_graf1, col_graf2 = st.columns(2)
    with col_graf1:
        st.markdown("##### Distribuição das Previsões")
        contagem_previsoes = df_final['Previsão'].value_counts().reset_index()
        contagem_previsoes.columns = ['Previsão', 'Contagem']
        fig_pie = px.pie(contagem_previsoes, names='Previsão', values='Contagem',
                         color='Previsão', color_discrete_map={'Vai reverter': '#FF4B4B', 'Não vai reverter': '#00C0F2'})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_graf2:
        st.markdown("##### Distribuição das Probabilidades de Reversão")
        fig_hist = px.histogram(df_final, x='Probabilidade_numerica', nbins=20,
                                title='Frequência por Faixa de Probabilidade',
                                labels={'Probabilidade_numerica': 'Probabilidade de Reversão'})
        fig_hist.update_layout(yaxis_title='Quantidade de Clientes')
        st.plotly_chart(fig_hist, use_container_width=True)

    # Tabela de dados detalhados
    st.subheader("Dados Detalhados")
    # Exibe a tabela sem a coluna numérica auxiliar
    st.dataframe(df_final.drop(columns=['Probabilidade_numerica']))

    # Download
    st.subheader("Download dos Resultados")

    # PREPARA O ARQUIVO EXCEL EM MEMÓRIA
    output_excel = io.BytesIO()
    # Usamos o mesmo dataframe final, sem a coluna numérica auxiliar
    df_to_download = df_final.drop(columns=['Probabilidade_numerica'])
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        df_to_download.to_excel(writer, index=False, sheet_name='Previsoes')
    excel_data = output_excel.getvalue()

    # PREPARA O ARQUIVO CSV EM MEMÓRIA (JÁ EXISTENTE NO SEU CÓDIGO)
    csv_data = df_to_download.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')

    # CRIA DUAS COLUNAS PARA OS BOTÕES
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Baixar em Excel (.xlsx)",
            data=excel_data,
            file_name='previsoes_reversao.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            use_container_width=True
        )
    with col2:
        st.download_button(
            label="📄 Baixar em CSV (.csv)",
            data=csv_data,
            file_name='previsoes_reversao.csv',
            mime='text/csv',
            use_container_width=True
        )

st.markdown("---")
st.write("Desenvolvido com Streamlit. Pela equipe de E&I 🚀")