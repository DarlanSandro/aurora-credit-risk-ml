"""
Aplicativo Streamlit para Concessão Automatizada de Crédito e Análise de Risco.
Projeto Aurora Crédito Digital - Case Consumer Credit Risk.
Permite entrada de formulário, predição de probabilidade, decisão econômica (limiar 0.59)
e explicabilidade local com as Top 5 variáveis SHAP.
"""

import os
import sys
import warnings

# Suprimir avisos internos de bibliotecas
warnings.filterwarnings("ignore")

# Garantir que o backend do Matplotlib não exija interface gráfica interativa (Tkinter)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Garantir que o diretório atual esteja no sys.path para carregar preparacao_dados
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
if diretorio_atual not in sys.path:
    sys.path.insert(0, diretorio_atual)

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Importar o preparador de dados para deserialização do pipeline
import preparacao_dados
from preparacao_dados import PreparadorDados

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Aurora Crédito Digital | Avaliação de Risco",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache para carregar o modelo apenas uma vez
@st.cache_resource
def carregar_modelo():
    caminho_modelo = os.path.join(diretorio_atual, "modelo_credito_xgb.joblib")
    if not os.path.exists(caminho_modelo):
        st.error(f"Arquivo de modelo '{caminho_modelo}' não encontrado na pasta!")
        st.stop()
    
    # Garantir classe no __main__ para compatibilidade do pickle
    sys.modules["__main__"].PreparadorDados = PreparadorDados
    
    artefatos = joblib.load(caminho_modelo)
    return artefatos

# Carregamento dos artefatos
artefatos = carregar_modelo()
pipeline = artefatos["pipeline"]
limiar_otimo = float(artefatos.get("limiar_otimo", 0.59))
colunas_esperadas = artefatos.get("colunas_entrada", [
    "idade", "renda_mensal", "dependentes", "uso_limite_rotativo", "razao_divida",
    "linhas_credito_abertas", "financiamentos_imobiliarios",
    "atrasos_30_59_dias", "atrasos_60_89_dias", "atrasos_90_mais_dias"
])

# Cabeçalho Principal
st.title("🏦 Aurora Crédito Digital — Avaliação de Risco de Crédito")
st.markdown(
    """
    Sistema automatizado de triagem e inteligência de crédito para concessão sustentável.
    Utiliza o modelo campeão **XGBoost** calibrado com a esteira oficial de engenharia de atributos
    e a régua ótima de corte financeiro (**Limiar = 59,00%**).
    """
)
st.divider()

# Barra lateral com parâmetros do sistema
with st.sidebar:
    st.header("⚙️ Parâmetros do Modelo")
    st.info(f"**Algoritmo:** XGBoost (GridSearchCV)")
    st.metric(label="Régua de Corte Ótima", value=f"{limiar_otimo * 100:.1f}%")
    st.markdown(
        """
        **Matriz de Perdas Asimétricas:**
        * **Falso Negativo (Calote Aprovado):** Custo de **R$ 5.000,00**
        * **Falso Positivo (Bom Pagador Negado):** Custo de **R$ 500,00**
        * **Relação de Custo:** 10 : 1
        """
    )
    st.markdown("---")
    st.caption("Desenvolvido para conformidade regulatória BACEN e Código de Defesa do Consumidor.")

# Formulário de entrada dos dados do cliente
st.subheader("📋 Dados Cadastrais e Financeiros do Proponente")

with st.form("form_avaliacao_credito"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 👤 Perfil do Cliente e Capacidade Financeira")
        idade = st.number_input(
            "Idade do Cliente (anos)",
            min_value=18,
            max_value=105,
            value=38,
            step=1,
            help="Idade cronológica do proponente."
        )
        renda_mensal = st.number_input(
            "Renda Mensal Declarada (R$)",
            min_value=0.0,
            max_value=200000.0,
            value=6500.0,
            step=250.0,
            format="%.2f",
            help="Renda bruta mensal informada pelo cliente."
        )
        dependentes = st.number_input(
            "Número de Dependentes",
            min_value=0,
            max_value=15,
            value=1,
            step=1,
            help="Quantidade de membros familiares que dependem da renda."
        )
        uso_limite_rotativo = st.number_input(
            "Uso do Limite de Crédito Rotativo (Proporção)",
            min_value=0.0,
            max_value=5.0,
            value=0.35,
            step=0.05,
            format="%.2f",
            help="Total utilizado no cheque especial e rotativo do cartão dividido pelo limite total. Exemplo: 0.35 = 35%."
        )
        razao_divida = st.number_input(
            "Razão de Dívida / DTI (Proporção)",
            min_value=0.0,
            max_value=5.0,
            value=0.28,
            step=0.05,
            format="%.2f",
            help="Total de pagamentos mensais de dívidas dividido pela renda mensal. Exemplo: 0.28 = 28%."
        )

    with col2:
        st.markdown("##### 💳 Histórico e Compromissos de Crédito")
        linhas_credito_abertas = st.number_input(
            "Linhas de Crédito e Empréstimos Abertos",
            min_value=0,
            max_value=50,
            value=6,
            step=1,
            help="Quantidade de empréstimos, cartões ativos e financiamentos contratados."
        )
        financiamentos_imobiliarios = st.number_input(
            "Financiamentos Imobiliários / Hipotecas",
            min_value=0,
            max_value=25,
            value=1,
            step=1,
            help="Quantidade de contratos de crédito habitacional ativos."
        )
        atrasos_30_59_dias = st.number_input(
            "Atrasos de 30 a 59 dias (Últimos 2 Anos)",
            min_value=0,
            max_value=98,
            value=0,
            step=1,
            help="Frequência de atrasos leves no histórico de crédito recente."
        )
        atrasos_60_89_dias = st.number_input(
            "Atrasos de 60 a 89 dias (Últimos 2 Anos)",
            min_value=0,
            max_value=98,
            value=0,
            step=1,
            help="Frequência de atrasos moderados no histórico de crédito recente."
        )
        atrasos_90_mais_dias = st.number_input(
            "Atrasos de 90 dias ou mais (Últimos 2 Anos)",
            min_value=0,
            max_value=98,
            value=0,
            step=1,
            help="Frequência de inadimplência severa prévia."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    botao_avaliar = st.form_submit_button(
        "🚀 Avaliar Risco e Tomar Decisão de Crédito",
        type="primary",
        use_container_width=True
    )

# Processamento ao clicar no botão
if botao_avaliar:
    st.divider()
    st.subheader("📊 Resultado da Avaliação de Crédito")

    # 1. Montagem do DataFrame para o cliente
    dados_cliente = {
        "idade": idade,
        "renda_mensal": renda_mensal,
        "dependentes": dependentes,
        "uso_limite_rotativo": uso_limite_rotativo,
        "razao_divida": razao_divida,
        "linhas_credito_abertas": linhas_credito_abertas,
        "financiamentos_imobiliarios": financiamentos_imobiliarios,
        "atrasos_30_59_dias": atrasos_30_59_dias,
        "atrasos_60_89_dias": atrasos_60_89_dias,
        "atrasos_90_mais_dias": atrasos_90_mais_dias
    }

    df_cliente = pd.DataFrame([dados_cliente])[colunas_esperadas]

    # 2. Predição com o Pipeline Completo
    with st.spinner("Processando atributos e estimando probabilidade de inadimplência..."):
        probabilidade_calote = float(pipeline.predict_proba(df_cliente)[0, 1])
        decisao_recusa = probabilidade_calote >= limiar_otimo

    # 3. Apresentação das Métricas Principais
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.metric(
            label="Probabilidade de Inadimplência",
            value=f"{probabilidade_calote * 100:.2f}%",
            help="Probabilidade estimada pelo XGBoost de ocorrência de calote (atraso >= 90 dias)."
        )
    with col_res2:
        st.metric(
            label="Limiar Decisório de Corte",
            value=f"{limiar_otimo * 100:.2f}%",
            help="Se a probabilidade for igual ou superior a este limiar, o crédito deve ser recusado."
        )
    with col_res3:
        status_texto = "ALTO RISCO" if decisao_recusa else "BAIXO RISCO"
        st.metric(
            label="Classificação de Risco",
            value=status_texto
        )

    # 4. Decisão Econômica do Modelo
    if decisao_recusa:
        st.error(
            f"""
            ### ❌ DECISÃO: CRÉDITO RECUSADO
            A probabilidade de calote estimada (**{probabilidade_calote * 100:.2f}%**) é **superior ao limiar ótimo** de **{limiar_otimo * 100:.2f}%**.
            Conceder crédito a este perfil apresenta custo esperado desfavorável para a carteira sob a matriz de perdas (Custo FN = R$ 5.000,00).
            """
        )
    else:
        st.success(
            f"""
            ### ✅ DECISÃO: CRÉDITO APROVADO
            A probabilidade de calote estimada (**{probabilidade_calote * 100:.2f}%**) está **abaixo do limiar ótimo** de **{limiar_otimo * 100:.2f}%**.
            O proponente apresenta perfil de risco compatível com as margens operacionais da Aurora Crédito Digital.
            """
        )

    st.markdown("---")

    # 5. Explicabilidade SHAP das Top 5 Features (Local SHAP)
    st.subheader("🔍 Explicabilidade da Decisão — Top 5 Fatores Determinantes (SHAP)")
    st.markdown(
        """
        O gráfico e a tabela abaixo detalham as 5 variáveis que mais influenciaram na decisão individual deste proponente:
        * 🔴 **Barras Vermelhas (Positivas):** Fatores que **aumentaram** a probabilidade de inadimplência (empurraram para recusa).
        * 🟢 **Barras Verdes (Negativas):** Fatores que **reduziram** a probabilidade de inadimplência (favoreceram a aprovação).
        """
    )

    with st.spinner("Calculando valores de impacto SHAP para este cliente..."):
        # Processar atributos com a esteira até antes do estimador
        X_cliente_proc = pipeline[:-1].transform(df_cliente)
        modelo_xgb = pipeline.named_steps["modelo"]

        # Explicador SHAP TreeExplainer
        explicador = shap.TreeExplainer(modelo_xgb)
        shap_values_cliente = explicador(X_cliente_proc)

        # Vetor de valores SHAP para a amostra
        valores_shap = shap_values_cliente[0].values
        colunas_transformadas = list(X_cliente_proc.columns)

        # Mapeamento dos nomes para exibição amigável
        nomes_legiveis = {
            "idade": "Idade",
            "renda_mensal": "Renda Mensal",
            "dependentes": "Número de Dependentes",
            "uso_limite_rotativo": "Uso do Limite Rotativo",
            "razao_divida": "Razão de Dívida (DTI)",
            "linhas_credito_abertas": "Linhas de Crédito Abertas",
            "financiamentos_imobiliarios": "Financiamentos Imobiliários",
            "atrasos_30_59_dias": "Atrasos 30-59 Dias",
            "atrasos_60_89_dias": "Atrasos 60-89 Dias",
            "atrasos_90_mais_dias": "Atrasos 90+ Dias",
            "renda_faltante_flag": "Indicador de Renda Ausente",
            "dependentes_faltantes_flag": "Indicador de Dependentes Ausentes",
            "flag_atraso_extremo": "Flag de Atraso Crítico (>=96)",
            "renda_por_dependente": "Renda per Capita",
            "sobra_caixa": "Sobra de Caixa Mensal"
        }

        df_shap_local = pd.DataFrame({
            "Código Atributo": colunas_transformadas,
            "Fator de Risco": [nomes_legiveis.get(c, c) for c in colunas_transformadas],
            "Impacto SHAP": valores_shap,
            "Impacto Absoluto": np.abs(valores_shap),
            "Valor do Cliente": [X_cliente_proc[c].iloc[0] for c in colunas_transformadas]
        })

        # Filtrar Top 5 pelo maior impacto absoluto
        df_top5 = df_shap_local.sort_values(by="Impacto Absoluto", ascending=False).head(5)

        # Gráfico Horizontal das Top 5 Features
        fig, ax = plt.subplots(figsize=(10, 4.8))
        df_plot = df_top5.sort_values(by="Impacto SHAP", ascending=True)

        cores = ["#e74c3c" if val > 0 else "#27ae60" for val in df_plot["Impacto SHAP"]]
        barras = ax.barh(df_plot["Fator de Risco"], df_plot["Impacto SHAP"], color=cores, height=0.55)

        ax.axvline(0, color="#34495e", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_xlabel("Impacto no Risco de Crédito (SHAP Value)", fontsize=11, fontweight="bold")
        ax.set_title("Top 5 Variáveis Mais Impactantes na Predição Individual", fontsize=12, fontweight="bold", pad=15)
        ax.grid(axis="x", linestyle=":", alpha=0.6)

        # Margem simétrica generosa para que os textos nunca colidam com os rótulos do eixo Y
        max_abs = max(abs(df_plot["Impacto SHAP"].min()), abs(df_plot["Impacto SHAP"].max()))
        margem = max(max_abs * 1.45, 0.15)
        ax.set_xlim(-margem, margem)
        ax.tick_params(axis="y", labelsize=10, pad=8)

        # Adicionar rótulos nas barras com alinhamento dinâmico
        for barra in barras:
            largura = barra.get_width()
            offset = margem * 0.03 if largura >= 0 else -margem * 0.03
            ha = "left" if largura >= 0 else "right"
            cor_texto = "#c0392b" if largura >= 0 else "#1e8449"
            ax.text(
                largura + offset, barra.get_y() + barra.get_height() / 2,
                f"{largura:+.3f}",
                va="center", ha=ha, fontsize=10, fontweight="bold",
                color=cor_texto
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Tabela textual explicativa para Adverse Action Reasons
        st.markdown("##### 📄 Justificativa Formal da Decisão (*Adverse Action Reasons*)")
        
        tabela_exibicao = df_top5.copy()
        tabela_exibicao["Efeito na Decisão"] = tabela_exibicao["Impacto SHAP"].apply(
            lambda x: "🚨 Pressionou para Recusa (Aumenta Risco)" if x > 0 else "🛡️ Favoreceu Aprovação (Reduz Risco)"
        )
        tabela_exibicao["Valor Informado"] = tabela_exibicao["Valor do Cliente"].apply(
            lambda v: f"{v:,.2f}" if isinstance(v, (int, float)) else str(v)
        )
        tabela_exibicao["Magnitude SHAP"] = tabela_exibicao["Impacto SHAP"].apply(lambda v: f"{v:+.4f}")

        st.dataframe(
            tabela_exibicao[["Fator de Risco", "Valor Informado", "Efeito na Decisão", "Magnitude SHAP"]],
            use_container_width=True,
            hide_index=True
        )

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Sistema de Decisão de Crédito — Aurora Crédito Digital | Case Consumer Credit Risk")
