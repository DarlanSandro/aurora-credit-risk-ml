# 🏦 Aurora Crédito Digital — Previsão de Risco de Crédito e Decisão Econômica

Aplicação de Machine Learning de ponta a ponta desenvolvida para otimização da concessão de crédito ao consumo (Case FNAT), equilibrando poder preditivo com custo financeiro assimétrico e governança regulatória.

---

## 🎯 Objetivo de Negócio
Reduzir as perdas por inadimplência severa (atrasos $\ge$ 90 dias em 2 anos) minimizando a matriz de custo operacional:
* **Falso Negativo (Aprovar inadimplente):** Custo de **R$ 5.000,00** (perda do principal).
* **Falso Positivo (Negar bom pagador):** Custo de **R$ 500,00** (custo de oportunidade da margem).
* **Relação de Custo:** 10 : 1.

---

## 🚀 Resultados dos Modelos (Holdout de Teste)

| Modelo | ROC AUC (Teste) | PR AUC (Teste) | Limiar Ótimo ($\tau^*$) | Menor Custo Total (R$) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Otimizado)** | **0,8686** | **0,4101** | **0,59** | **R$ 6.103.500,00** |
| Random Forest (balanced) | 0,8621 | 0,3895 | 0,42 | R$ 6.320.000,00 |
| Árvore de Decisão (`depth=6`) | 0,8459 | 0,3514 | 0,09 | R$ 6.617.000,00 |
| Dummy Classifier (Baseline) | 0,5000 | 0,0668 | 0.07 | R$ 12.535.000,00 |

* **Economia Gerada:** O XGBoost proporcionou uma economia líquida de **R$ 513.500,00** frente à Árvore de Decisão e mais de **R$ 6,4 milhões** frente ao baseline.

---

## 🔍 Explicabilidade e Governança (SHAP)
O sistema incorpora explicabilidade local (*Adverse Action Reasons*) com **TreeExplainer (SHAP)**, permitindo justificar formalmente ao cliente e ao BACEN os fatores determinantes de cada recusa ou aprovação de crédito.

---

## 💻 Como Executar Localmente

1. Clone o repositório:
```bash
git clone https://github.com/DarlanSandro/aurora-credit-risk-ml.git
cd aurora-credit-risk-ml