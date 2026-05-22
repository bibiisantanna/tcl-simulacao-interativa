import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# ── Configuração da página ──
st.set_page_config(
    page_title="Teorema Central do Limite",
    layout="wide"
)

# ── Estilo ──
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1a1a2e; }
    h2, h3 { color: #16213e; }
    .stSlider label { font-weight: bold; color: #16213e; }
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

# ── Título ──
st.title("Teorema Central do Limite (TCL)")
st.markdown("**Distribuição Gamma(α=2, β=1) — Simulação interativa**")

# ── Explicação didática ──
with st.expander("📖 O que é o Teorema Central do Limite? (clique para ver)", expanded=True):
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

# ── Controles ──
st.subheader("⚙️ Parâmetros da Simulação")
col_s1, col_s2 = st.columns(2)
with col_s1:
    n = st.slider("Tamanho da amostra (n)", min_value=2, max_value=200, value=10, step=1,
                  help="Quantas observações em cada amostra")
with col_s2:
    m = st.slider("Número de amostras (m)", min_value=100, max_value=1000, value=1000, step=100,
                  help="Quantas vezes o experimento é repetido")

alpha_param = 2
beta_param  = 1
media_pop   = alpha_param / beta_param
desvio_pop  = np.sqrt(alpha_param) / beta_param
ep_teorico  = desvio_pop / np.sqrt(n)

# ── Simulação ──
np.random.seed(42)
medias = np.array([
    np.mean(np.random.gamma(shape=alpha_param, scale=1/beta_param, size=n))
    for _ in range(m)
])
media_obs  = np.mean(medias)
ep_obs     = np.std(medias)
erro_pct   = abs(media_obs - media_pop) / media_pop * 100

st.divider()

# ── Métricas rápidas ──
st.subheader("📈 Resultados para n = {} e m = {}".format(n, m))
c1, c2, c3, c4 = st.columns(4)
c1.metric("x̄ observada", f"{media_obs:.4f}", delta=f"{media_obs - media_pop:+.4f} vs μ")
c2.metric("EP observado", f"{ep_obs:.4f}")
c3.metric("EP teórico (σ/√n)", f"{ep_teorico:.4f}")
c4.metric("Erro em relação a μ", f"{erro_pct:.3f}%")

if erro_pct < 1:
    st.success("✅ A média amostral está muito próxima da média verdadeira (erro < 1%).")
elif erro_pct < 3:
    st.info("ℹ️ A média amostral está razoavelmente próxima da média verdadeira.")
else:
    st.warning("⚠️ Aumente m para reduzir a variabilidade das médias.")

st.divider()

# ── Gráficos principais ──
st.subheader("🔬 Histograma das Médias Amostrais")

tab1, tab2 = st.tabs(["📊 Gráfico principal (n atual)", "🔀 Convergência (vários n)"])

with tab1:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    ax.hist(medias, bins=35, density=True, color="#4fc3f7",
            alpha=0.75, edgecolor="none", label="Médias amostrais")

    x = np.linspace(medias.min() * 0.85, medias.max() * 1.15, 300)
    ax.plot(x, stats.gamma.pdf(x, a=n * alpha_param, scale=1 / (n * beta_param)),
            color="red", lw=2.5, label="Distribuição exata (Gamma)")
    ax.plot(x, stats.norm.pdf(x, loc=media_pop, scale=ep_teorico),
            color="blue", lw=2.5, linestyle="--", label="Aproximação pelo TCL (Normal)")
    ax.axvline(media_pop, color="orange", lw=2, linestyle=":",
               label=f"μ verdadeira = {media_pop}")

    ax.set_title(f"Distribuição das médias amostrais  |  n={n}, m={m}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Média Amostral", fontsize=11)
    ax.set_ylabel("Densidade", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor("#ddd")

    st.pyplot(fig)
    plt.close()

    if n < 10:
        st.markdown('<div class="destaque">⚠️ <b>n pequeno:</b> a curva Normal (azul tracejado) ainda não se encaixa bem no histograma. Aumente n e veja o TCL em ação!</div>', unsafe_allow_html=True)
    elif n < 30:
        st.markdown('<div class="destaque">📌 <b>n moderado:</b> a Normal já se aproxima, mas a curva exata (Gamma, vermelha) ainda é mais precisa.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="explicacao">✅ <b>n ≥ 30:</b> a aproximação Normal é excelente — as duas curvas praticamente se sobrepõem. O TCL está em plena ação!</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("**Como a distribuição das médias muda conforme n cresce:**")
    st.caption("Cada cor representa um tamanho amostral diferente. Note como as distribuições ficam mais estreitas e simétricas.")

    tamanhos_fixos = [2, 5, 10, 30, 50, 100]
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.patch.set_facecolor("#f8f9fa")
    ax2.set_facecolor("#ffffff")

    palette = plt.cm.plasma(np.linspace(0.1, 0.85, len(tamanhos_fixos)))
    np.random.seed(42)
    for cor, n_fix in zip(palette, tamanhos_fixos):
        med = np.array([
            np.mean(np.random.gamma(shape=alpha_param, scale=1/beta_param, size=n_fix))
            for _ in range(1000)
        ])
        ax2.hist(med, bins=35, density=True, alpha=0.55,
                 color=cor, edgecolor="none", label=f"n = {n_fix}")

    ax2.axvline(media_pop, color="black", lw=2.5, linestyle="--",
                label=f"μ = {media_pop}")
    ax2.set_title("Convergência das médias para μ conforme n aumenta",
                  fontsize=13, fontweight="bold")
    ax2.set_xlabel("Média Amostral", fontsize=11)
    ax2.set_ylabel("Densidade", fontsize=11)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#ddd")

    st.pyplot(fig2)
    plt.close()

    st.markdown("""
    **Como interpretar este gráfico:**
    - 🟣 **n=2 (roxo escuro):** distribuição bem espalhada e assimétrica — as médias variam muito.
    - 🟠 **n=10/30 (laranja):** distribuição mais concentrada e simétrica.
    - 🟡 **n=100 (amarelo):** distribuição bem estreita, centrada em μ=2 — quase perfeita Normal.
    - A linha preta tracejada é a **média verdadeira μ=2**. Todas as distribuições giram em torno dela!
    """)

st.divider()

# ── Tabela comparativa ──
st.subheader("📋 Tabela Comparativa — Todos os Tamanhos Amostrais")
st.caption("Como x̄, EP observado e EP teórico se comportam para diferentes valores de n")

np.random.seed(42)
rows = []
for n_t in [2, 5, 10, 30, 50, 100]:
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

st.caption("μ verdadeira = 2.0  |  σ = √2 ≈ 1.4142  |  m = 1000 amostras por experimento")