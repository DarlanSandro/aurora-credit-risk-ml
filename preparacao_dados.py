"""
Modulo de preparacao de dados, engenharia de features e pipeline
compativel com Scikit-Learn e SHAP para analise de risco de credito.
Alinhado rigorosamente com o gabarito oficial da FNAT / Aurora Credito Digital.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SEED = 42

def preparar_dados(df, mediana_renda=None, mediana_dependentes=None):
    """
    Funcao de preparacao de dados e engenharia de atributos oficial do gabarito.
    """
    df_proc = df.copy()
    df_proc['renda_faltante_flag'] = df_proc['renda_mensal'].isna().astype(int)
    df_proc['dependentes_faltantes_flag'] = df_proc['dependentes'].isna().astype(int)

    if mediana_renda is None:
        mediana_renda = float(df_proc['renda_mensal'].median())
    if mediana_dependentes is None:
        mediana_dependentes = float(df_proc['dependentes'].median())

    df_proc['renda_mensal'] = df_proc['renda_mensal'].fillna(mediana_renda)
    df_proc['dependentes'] = df_proc['dependentes'].fillna(mediana_dependentes)

    colunas_atraso = ['atrasos_30_59_dias', 'atrasos_60_89_dias', 'atrasos_90_mais_dias']
    df_proc['flag_atraso_extremo'] = (
        (df_proc['atrasos_30_59_dias'] >= 96) |
        (df_proc['atrasos_60_89_dias'] >= 96) |
        (df_proc['atrasos_90_mais_dias'] >= 96)
    ).astype(int)

    for col in colunas_atraso:
        df_proc[col] = df_proc[col].clip(upper=20)

    df_proc['renda_mensal'] = df_proc['renda_mensal'].clip(upper=50000.0)
    df_proc['renda_por_dependente'] = df_proc['renda_mensal'] / (df_proc['dependentes'] + 1)
    df_proc['sobra_caixa'] = df_proc['renda_mensal'] * (1 - df_proc['razao_divida'])

    return df_proc, mediana_renda, mediana_dependentes


class PreparadorDados(BaseEstimator, TransformerMixin):
    """
    Transformer customizado para integracao com o Pipeline do Scikit-Learn.
    """
    def __init__(self):
        self.mediana_renda_ = None
        self.mediana_dependentes_ = None
        self.colunas_saida_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        X_tratado, self.mediana_renda_, self.mediana_dependentes_ = preparar_dados(X_df)
        self.colunas_saida_ = list(X_tratado.columns)
        return self

    def transform(self, X):
        X_tratado, _, _ = preparar_dados(X, self.mediana_renda_, self.mediana_dependentes_)
        return X_tratado

    def get_feature_names_out(self, input_features=None):
        if hasattr(self, "colunas_saida_") and self.colunas_saida_ is not None:
            return np.array(self.colunas_saida_, dtype=object)
        colunas_padrao = [
            "idade", "renda_mensal", "dependentes", "uso_limite_rotativo",
            "razao_divida", "linhas_credito_abertas", "financiamentos_imobiliarios",
            "atrasos_30_59_dias", "atrasos_60_89_dias", "atrasos_90_mais_dias",
            "renda_faltante_flag", "dependentes_faltantes_flag", "flag_atraso_extremo",
            "renda_por_dependente", "sobra_caixa"
        ]
        return np.array(colunas_padrao, dtype=object)


def obter_parametros_treinamento(df):
    """Calcula parametros de imputacao e regras a partir do conjunto de treino."""
    return {
        "mediana_renda": float(df["renda_mensal"].median()),
        "mediana_dependentes": float(df["dependentes"].median()),
        "teto_renda": 50000.0,
        "teto_atraso": 20,
    }


def transformar_dados(df, parametros=None):
    """
    Aplica as regras de preparacao retornando estritamente um DataFrame (sem tupla),
    garantindo compatibilidade com celulas anteriores e Scikit-Learn.
    """
    if isinstance(parametros, dict):
        med_r = parametros.get("mediana_renda")
        med_d = parametros.get("mediana_dependentes")
    elif isinstance(parametros, (tuple, list)) and len(parametros) >= 2:
        med_r, med_d = parametros[0], parametros[1]
    else:
        med_r, med_d = None, None
    df_proc, _, _ = preparar_dados(df, mediana_renda=med_r, mediana_dependentes=med_d)
    return df_proc


# Aliases para compatibilidade retroativa com celulas anteriores
TransformadorPreparoCredito = PreparadorDados

def criar_pipeline(modelo="passthrough"):
    """Constroi o pipeline completo em 3 etapas."""
    imputador = SimpleImputer(strategy="median").set_output(transform="pandas")
    pipeline = Pipeline([
        ("preparo", PreparadorDados()),
        ("imputacao", imputador),
        ("modelo", modelo)
    ])
    return pipeline

def pipeline_preparacao(df_completo, proporcao_teste=0.25, seed=42):
    """Funcao utilitaria para divisao estratificada e preparacao."""
    X = df_completo.drop(columns=["inadimplente_2anos"])
    y = df_completo["inadimplente_2anos"]

    X_treino_bruto, X_teste_bruto, y_treino, y_teste = train_test_split(
        X, y, test_size=proporcao_teste, random_state=seed, stratify=y
    )

    X_treino, med_r, med_d = preparar_dados(X_treino_bruto)
    X_teste, _, _ = preparar_dados(X_teste_bruto, med_r, med_d)
    parametros = {"mediana_renda": med_r, "mediana_dependentes": med_d, "teto_renda": 50000.0, "teto_atraso": 20}

    return X_treino, X_teste, y_treino, y_teste, parametros
