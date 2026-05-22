import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

st.set_page_config(page_title="Teorema Central do Limite", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1a1a2e; }
    h2, h3 { color: #16213e; }
    .explicacao {
        background: #e8f4f8;
        border-left: 5px solid #2196F3;
        padding: 15px 20px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .destaque {
        background: #fff3cd;
        border-left: 5px solid #ff9800;
        padding: 12px 18px;
        border-radius: 4px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("Teorema Central do Limite (TCL)")
st.markdown("**Distribuição Gamma(α=2, β=1) — Simulação interativa**")

with st.expander("📖 O que é o Teorema Central do Limite?", expanded=True):
    st.markdown("""
    <div class="explicacao">
    <b>Teorema Central do Limite (TCL)</b><br><br>
    O TCL diz que, independente da forma da distribuição original de uma população,
    a distribuição das <b>médias amostrais</b> se aproxima de uma <b>distribuição Normal</b>
    conforme o tamanho da amostra <b>n</b> cresce.<br><br>
    <b>Em outras palavras:</b> mesmo que a população seja assimétrica (como a Gamma),
    se você tirar muitas amostras e calcular a média de cada uma,
    essas médias formarão um sino — a curva Normal.<br><br>
    📌 Regra prática: para n ≥ 30, a aproximação já é bastante boa.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Média verdadeira (μ)", "2.0", help="α/β = 2/1 = 2")
    with col2:
        st.metric("Desvio padrão (σ)", "1.414", help="√α / β = √2 ≈ 1.414")
    with col3:
        st.metric("Erro padrão teórico (EP)", "σ / √n", help="Diminui conforme n cresce")

st.divider()

# ── Parâmetros ──
alpha_param = 2
beta_param  = 1
media_pop   = alpha_param / beta_param
desvio_pop  = np.sqrt(alpha_param) / beta_param
m           = 1000

# ── Slider + Gráfico lado a lado ──
st.subheader("⚙️ Parâmetros da Simulação")
col_slider, col_graf = st.columns([1, 2])

with col_slider:
    n = st.slider("Tamanho da amostra (n)", min_value=2, max_value=200, value=10, step=1)
    
    cols_btn = st.columns(6)
    valores = [2, 5, 10, 30, 50, 100]
    for i, val in enumerate(valores):
        if cols_btn[i].button(str(val), use_container_width=True):
            n = val

    ep_teorico = desvio_pop / np.sqrt(n)
    np.random.seed(123)
    medias = np.array([
        np.mean(np.random.gamma(shape=alpha_param, scale=1/beta_param, size=n))
        for _ in range(m)
    ])
    media_obs = np.mean(medias)
    ep_obs    = np.std(medias)
    erro_pct  = abs(media_obs - media_pop) / media_pop * 100

    st.metric("x̄ observada", f"{media_obs:.4f}", delta=f"{media_obs - media_pop:+.4f} vs μ")
    st.metric("EP observado", f"{ep_obs:.4f}")
    st.metric("EP teórico (σ/√n)", f"{ep_teorico:.4f}")
    st.metric("Erro em relação a μ", f"{erro_pct:.3f}%")

    st.markdown("---")
    st.markdown("**📝 Como a aproximação evolui com n:**")

if n < 10:
    st.markdown(f"""
    <div class="destaque">
    ⚠️ <b>n = {n} (pequeno):</b> a distribuição das médias ainda carrega a assimetria da Gamma.
    A curva Normal (azul) se afasta bastante da distribuição exata (vermelha).
    O TCL ainda não se aplica bem — as médias variam muito e o histograma é assimétrico.
    </div>
    """, unsafe_allow_html=True)
elif n < 30:
    st.markdown(f"""
    <div class="destaque">
    📌 <b>n = {n} (moderado):</b> a distribuição começa a se aproximar da Normal.
    As curvas vermelha e azul já se sobrepõem parcialmente, mas ainda há diferença visível
    nas caudas. A aproximação pelo TCL melhora, mas não é perfeita.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="explicacao">
    ✅ <b>n = {n} (grande):</b> o TCL está em plena ação. A curva Normal (azul tracejado)
    praticamente coincide com a distribuição exata (vermelha). O histograma tem formato
    de sino simétrico centrado em μ = 2, confirmando que a média amostral converge
    para a distribuição Normal independente da forma original da população.
    </div>
    """, unsafe_allow_html=True)
    
with col_graf:
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    ax.hist(medias, bins=35, density=True, color="#4fc3f7",
            alpha=0.75, edgecolor="none", label="Médias amostrais")

    x = np.linspace(medias.min() * 0.85, medias.max() * 1.15, 300)
    ax.plot(x, stats.gamma.pdf(x, a=n * alpha_param, scale=1 / (n * beta_param)),
            color="red", lw=2, label="Exata (Gamma)")
    ax.plot(x, stats.norm.pdf(x, loc=media_pop, scale=ep_teorico),
            color="blue", lw=2, linestyle="--", label="TCL (Normal)")
    ax.axvline(media_pop, color="orange", lw=1.5, linestyle=":",
               label=f"μ = {media_pop}")

    ax.set_title(f"n = {n},  m = {m}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Média Amostral", fontsize=10)
    ax.set_ylabel("Densidade", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor("#ddd")

    st.pyplot(fig)
    plt.close()

st.divider()

# ── Aba de convergência ──
st.subheader("🔀 Convergência para vários n")
st.caption("Cada cor representa um tamanho amostral. Note como as distribuições ficam mais estreitas e simétricas.")

tamanhos_fixos = [2, 5, 10, 30, 50, 100]
fig2, ax2 = plt.subplots(figsize=(9, 4))
fig2.patch.set_facecolor("#f8f9fa")
ax2.set_facecolor("#ffffff")

palette = plt.cm.plasma(np.linspace(0.1, 0.85, len(tamanhos_fixos)))
np.random.seed(123)
for cor, n_fix in zip(palette, tamanhos_fixos):
    med = np.array([
        np.mean(np.random.gamma(shape=alpha_param, scale=1/beta_param, size=n_fix))
        for _ in range(1000)
    ])
    ax2.hist(med, bins=35, density=True, alpha=0.55,
             color=cor, edgecolor="none", label=f"n = {n_fix}")

ax2.axvline(media_pop, color="black", lw=2, linestyle="--", label=f"μ = {media_pop}")
ax2.set_title("Convergência das médias para μ conforme n aumenta", fontsize=12, fontweight="bold")
ax2.set_xlabel("Média Amostral", fontsize=10)
ax2.set_ylabel("Densidade", fontsize=10)
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(axis="y", alpha=0.3)
for spine in ax2.spines.values():
    spine.set_edgecolor("#ddd")

st.pyplot(fig2)
plt.close()

st.markdown("""
- 🟣 **n=2:** distribuição espalhada e assimétrica — as médias variam muito.
- 🟠 **n=10/30:** distribuição mais concentrada e simétrica.
- 🟡 **n=100:** distribuição estreita, centrada em μ=2 — quase Normal perfeita.
- A linha preta tracejada é a **média verdadeira μ=2**.
""")

st.divider()

# ── Tabela ──
st.subheader("📋 Tabela Comparativa")

np.random.seed(123)
rows = []
for n_t in tamanhos_fixos:
    med = np.array([
        np.mean(np.random.gamma(shape=alpha_param, scale=1/beta_param, size=n_t))
        for _ in range(1000)
    ])
    rows.append({
        "n": n_t,
        "x̄ observada": round(float(np.mean(med)), 4),
        "EP observado": round(float(np.std(med)), 4),
        "EP teórico (σ/√n)": round(desvio_pop / np.sqrt(n_t), 4),
        "Erro % em relação a μ": round(abs(float(np.mean(med)) - media_pop) / media_pop * 100, 3),
        "Aproximação Normal": "✅ Boa" if n_t >= 30 else ("⚠️ Razoável" if n_t >= 10 else "❌ Fraca")
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)
st.caption("μ = 2.0  |  σ = √2 ≈ 1.4142  |  m = 1000 amostras")
st.divider()
st.subheader("📝 Conclusão")
st.markdown("""
<div class="explicacao">
<b>O que os resultados mostram:</b><br><br>

Para a distribuição <b>Gamma(α=2, β=1)</b>, com média verdadeira <b>μ = 2</b> e desvio padrão <b>σ = √2 ≈ 1.414</b>,
simulamos m = 1000 amostras para cada tamanho amostral n ∈ {2, 5, 10, 30, 50, 100}.<br><br>

<b>1. A média amostral é um estimador não-viesado:</b> em todos os casos, x̄ ficou muito próxima de μ = 2,
com erros sempre abaixo de 3%, independente do tamanho de n.<br><br>

<b>2. O erro padrão diminui com n:</b> conforme n cresce, as médias amostrais ficam cada vez mais
concentradas em torno de μ — o EP observado se aproxima do EP teórico σ/√n, confirmando que
amostras maiores produzem estimativas mais precisas.<br><br>

<b>3. A aproximação Normal melhora com n:</b> para n pequeno (2 e 5), o histograma ainda reflete
a assimetria da Gamma original. A partir de n = 10 a Normal já é uma aproximação razoável,
e para n ≥ 30 a curva Normal praticamente coincide com a distribuição exata.<br><br>

<b>Conclusão geral:</b> os resultados confirmam empiricamente o Teorema Central do Limite —
independente da distribuição original da população, a distribuição das médias amostrais
converge para a Normal conforme n aumenta. A regra prática de n ≥ 30 se mostrou válida
neste experimento.
</div>
""", unsafe_allow_html=True)