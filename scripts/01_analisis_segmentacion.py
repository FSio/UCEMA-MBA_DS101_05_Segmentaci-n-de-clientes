"""TP Segmentacion de clientes (Mall Customers).
Corre todo el analisis, guarda figuras en outputs/figuras y tablas en outputs/tablas.
Uso:  python scripts/01_analisis_segmentacion.py
Semilla global: 42.
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
FIG = BASE / "outputs" / "figuras"; TAB = BASE / "outputs" / "tablas"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
SEED = 42
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})
PAL = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
out = {}

def save(name):
    plt.tight_layout(); plt.savefig(FIG / name, dpi=150); plt.close()

# ------------------------------------------------------------------ 1. Exploracion
df = pd.read_csv(BASE / "fuentes" / "Mall_Customers.csv")
df.columns = ["id", "genero", "edad", "ingreso", "score"]
out["n"] = len(df); out["faltantes"] = int(df.isna().sum().sum())
out["duplicados"] = int(df.duplicated().sum()); out["duplicados_sin_id"] = int(df.drop(columns="id").duplicated().sum())
out["genero"] = df.genero.value_counts().to_dict()
desc = df[["edad", "ingreso", "score"]].describe().round(2)
desc.to_csv(TAB / "01_descriptivos.csv")
out["desc"] = desc.to_dict()
corr = df[["edad", "ingreso", "score"]].corr(method="pearson").round(3); corr.to_csv(TAB / "02_correlaciones.csv")
sp = df[["edad", "ingreso", "score"]].corr(method="spearman").round(3)
out["corr"] = corr.to_dict(); out["corr_spearman"] = sp.to_dict()
# p-valor de las correlaciones
out["pvals"] = {f"{a}-{b}": float(stats.pearsonr(df[a], df[b])[1]) for a, b in (("edad", "ingreso"), ("edad", "score"), ("ingreso", "score"))}
# genero vs variables
gen = {}
for v in ("edad", "ingreso", "score"):
    f, m = df[df.genero == "Female"][v], df[df.genero == "Male"][v]
    gen[v] = dict(media_F=float(f.mean()), media_M=float(m.mean()), p_welch=float(stats.ttest_ind(f, m, equal_var=False)[1]),
                  p_mannwhitney=float(stats.mannwhitneyu(f, m)[1]))
out["genero_tests"] = gen

fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
for a, v, t in zip(axs, ("edad", "ingreso", "score"), ("Edad (años)", "Ingreso anual (k$)", "Spending Score (1-100)")):
    a.hist(df[v], bins=15, color="#4c72b0", edgecolor="white"); a.set_title(t); a.axvline(df[v].mean(), color="k", ls="--", lw=1)
axs[0].set_ylabel("Clientes")
save("01_distribuciones.png")

fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
for a, (x, y, tx, ty) in zip(axs, (("edad", "ingreso", "Edad", "Ingreso (k$)"), ("edad", "score", "Edad", "Spending Score"), ("ingreso", "score", "Ingreso (k$)", "Spending Score"))):
    for g, c in (("Female", "#d62728"), ("Male", "#1f77b4")):
        s = df[df.genero == g]; a.scatter(s[x], s[y], s=18, alpha=.75, c=c, label=g)
    a.set_xlabel(tx); a.set_ylabel(ty); a.set_title(f"r = {df[x].corr(df[y]):.2f}")
axs[0].legend(fontsize=8)
save("02_dispersion_pares.png")

fig, ax = plt.subplots(figsize=(4.2, 3.6))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(["Edad", "Ingreso", "Score"]); ax.set_yticklabels(["Edad", "Ingreso", "Score"])
for i in range(3):
    for j in range(3): ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")
ax.spines[:].set_visible(False)
save("03_correlaciones.png")

# score por tramos de edad
df["tramo_edad"] = pd.cut(df.edad, [17, 29, 39, 49, 59, 71], labels=["18-29", "30-39", "40-49", "50-59", "60+"])
te = df.groupby("tramo_edad", observed=True).agg(n=("id", "count"), score_medio=("score", "mean"), ingreso_medio=("ingreso", "mean")).round(1)
te.to_csv(TAB / "03_tramos_edad.csv"); out["tramos_edad"] = te.reset_index().astype({"tramo_edad": str}).to_dict("records")

# ---- Hopkins (tendencia a agrupar) por conjunto de variables
def hopkins(X, m=40, seed=SEED):
    rng = np.random.default_rng(seed); n, d = X.shape
    idx = rng.choice(n, m, replace=False)
    u = cdist(rng.uniform(X.min(0), X.max(0), (m, d)), X).min(1)
    w = np.array([np.sort(cdist(X[[i]], np.delete(X, i, 0))[0])[0] for i in idx])
    return float(u.sum() / (u.sum() + w.sum()))

df["genero_num"] = (df.genero == "Male").astype(int)
sets = {"A: Ingreso + Score": ["ingreso", "score"],
        "B: Edad + Ingreso + Score": ["edad", "ingreso", "score"],
        "C: Genero + Edad + Ingreso + Score": ["genero_num", "edad", "ingreso", "score"],
        "D: Edad + Score": ["edad", "score"]}
vt = []
for nm, cols in sets.items():
    X = StandardScaler().fit_transform(df[cols])
    hops = np.mean([hopkins(X, seed=s) for s in range(20)])
    best = max(((k, silhouette_score(X, KMeans(k, n_init=20, random_state=SEED).fit_predict(X))) for k in range(2, 9)), key=lambda t: t[1])
    vt.append(dict(conjunto=nm, hopkins=round(float(hops), 3), k_mejor_silueta=best[0], silueta=round(float(best[1]), 3)))
vt = pd.DataFrame(vt); vt.to_csv(TAB / "04_conjuntos_variables.csv", index=False); out["conjuntos"] = vt.to_dict("records")

# ------------------------------------------------------------------ 2. Clustering (conjunto principal A, y B como contraste)
def evalk(cols, kmax=10):
    X = StandardScaler().fit_transform(df[cols]); r = []
    for k in range(2, kmax + 1):
        km = KMeans(k, n_init=50, random_state=SEED).fit(X)
        r.append(dict(k=k, inercia=km.inertia_, silueta=silhouette_score(X, km.labels_), CH=calinski_harabasz_score(X, km.labels_),
                      DB=davies_bouldin_score(X, km.labels_)))
    X1 = KMeans(1, n_init=1, random_state=SEED).fit(X).inertia_
    return X, pd.DataFrame(r), X1

XA, mA, in1A = evalk(["ingreso", "score"])
XB, mB, in1B = evalk(["edad", "ingreso", "score"])
mA.round(4).to_csv(TAB / "05_metricas_k_A.csv", index=False); mB.round(4).to_csv(TAB / "06_metricas_k_B.csv", index=False)
out["metricas_A"] = mA.round(4).to_dict("records"); out["metricas_B"] = mB.round(4).to_dict("records"); out["inercia_k1_A"] = float(in1A)

# gap statistic (A)
def gap(X, kmax=10, B=30, seed=SEED):
    rng = np.random.default_rng(seed); lo, hi = X.min(0), X.max(0)
    def W(Z, k): return KMeans(k, n_init=10, random_state=SEED).fit(Z).inertia_
    res = []
    for k in range(1, kmax + 1):
        lw = np.log(W(X, k)); ref = np.array([np.log(W(rng.uniform(lo, hi, X.shape), k)) for _ in range(B)])
        res.append((k, ref.mean() - lw, ref.std() * np.sqrt(1 + 1 / B)))
    ks = [k for k, g, s in res]; g = [g for _, g, _ in res]; s = [s for _, _, s in res]
    kopt = next((ks[i] for i in range(len(ks) - 1) if g[i] >= g[i + 1] - s[i + 1]), ks[-1])
    return pd.DataFrame(res, columns=["k", "gap", "s"]), kopt
gA, kgapA = gap(XA); gA.round(4).to_csv(TAB / "07_gap_A.csv", index=False); out["k_gap_A"] = int(kgapA)
gB, kgapB = gap(XB); out["k_gap_B"] = int(kgapB)

# Figura codo / silueta / CH / DB / gap  (conjunto A)
fig, axs = plt.subplots(2, 3, figsize=(12, 6.4)); axs = axs.ravel()
ks = [1] + mA.k.tolist(); ine = [in1A] + mA.inercia.tolist()
axs[0].plot(ks, ine, "o-"); axs[0].set_title("Codo: inercia (SSE)"); axs[0].axvline(5, color="r", ls=":")
axs[1].plot(mA.k, mA.silueta, "o-", c="#2ca02c"); axs[1].set_title("Silueta media (mas alto = mejor)"); axs[1].axvline(5, color="r", ls=":")
axs[2].plot(mA.k, mA.CH, "o-", c="#9467bd"); axs[2].set_title("Calinski-Harabasz (mas alto = mejor)"); axs[2].axvline(5, color="r", ls=":")
axs[3].plot(mA.k, mA.DB, "o-", c="#ff7f0e"); axs[3].set_title("Davies-Bouldin (mas bajo = mejor)"); axs[3].axvline(5, color="r", ls=":")
axs[4].errorbar(gA.k, gA.gap, yerr=gA.s, fmt="o-", c="#8c564b", capsize=3); axs[4].set_title(f"Gap statistic (regla de Tibshirani: k={kgapA})")
# caida relativa de inercia
dec = [(ine[i] - ine[i + 1]) / ine[i] for i in range(len(ine) - 1)]
axs[5].bar(ks[1:], [100 * d for d in dec], color="#4c72b0"); axs[5].set_title("Caida % de inercia al pasar de k-1 a k")
for a in axs: a.set_xlabel("k")
save("04_eleccion_k_A.png")
out["caida_inercia_A"] = {int(k): float(100 * d) for k, d in zip(ks[1:], dec)}

fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
for a, c, t in zip(axs, ("inercia", "silueta", "DB"), ("Inercia", "Silueta", "Davies-Bouldin")):
    a.plot(mB.k, mB[c], "o-"); a.set_title(f"Edad+Ingreso+Score: {t}"); a.set_xlabel("k")
save("05_eleccion_k_B.png")

# ---- Modelo final: A con k=5 (se verifica abajo que k=5 es el elegido por criterios)
K = int(mA.loc[mA.silueta.idxmax(), "k"]); out["k_max_silueta_A"] = K
K = 5
km = KMeans(K, n_init=100, random_state=SEED).fit(XA)
lab_raw = km.labels_
# etiquetas ordenadas por (ingreso medio, score medio) para que sean estables y legibles
tmp = df.assign(l=lab_raw).groupby("l")[["ingreso", "score"]].mean()
orden = {old: new for new, old in enumerate(tmp.sort_values(["ingreso", "score"]).index)}
df["seg"] = pd.Series(lab_raw).map(orden).values
sil = silhouette_samples(XA, df.seg)
df["sil"] = sil
out["silueta_final"] = float(silhouette_score(XA, df.seg)); out["inercia_final"] = float(km.inertia_)
out["DB_final"] = float(davies_bouldin_score(XA, df.seg)); out["CH_final"] = float(calinski_harabasz_score(XA, df.seg))

# nombres por perfil (se asignan despues de mirar los centroides: ver informe)
prof = df.groupby("seg").agg(n=("id", "count"), edad=("edad", "mean"), ingreso=("ingreso", "mean"), score=("score", "mean"),
                             pct_mujeres=("genero", lambda s: 100 * (s == "Female").mean()), sil=("sil", "mean"),
                             ingreso_min=("ingreso", "min"), ingreso_max=("ingreso", "max"), score_min=("score", "min"), score_max=("score", "max"),
                             edad_min=("edad", "min"), edad_max=("edad", "max"), edad_sd=("edad", "std")).round(1)
prof["pct"] = (100 * prof.n / len(df)).round(1)
prof.to_csv(TAB / "08_perfil_segmentos.csv"); out["perfil"] = prof.reset_index().to_dict("records")
df[["id", "genero", "edad", "ingreso", "score", "seg"]].to_csv(TAB / "09_clientes_con_segmento.csv", index=False)

# figura principal
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for s in range(K):
    d = df[df.seg == s]; axs[0].scatter(d.ingreso, d.score, s=28, c=PAL[s], label=f"Seg. {s} (n={len(d)})", alpha=.85)
cen = df.groupby("seg")[["ingreso", "score"]].mean()
axs[0].scatter(cen.ingreso, cen.score, c="k", marker="X", s=130, zorder=5)
for s, r in cen.iterrows(): axs[0].annotate(str(s), (r.ingreso + 1.5, r.score + 2), weight="bold")
axs[0].set_xlabel("Ingreso anual (k$)"); axs[0].set_ylabel("Spending Score"); axs[0].set_title("Segmentos K-means (k = 5) sobre Ingreso y Score"); axs[0].legend(fontsize=8, loc="upper right")
data = [df[df.seg == s].edad for s in range(K)]
bp = axs[1].boxplot(data, patch_artist=True, tick_labels=[f"Seg. {s}" for s in range(K)])
for p, c in zip(bp["boxes"], PAL): p.set_facecolor(c); p.set_alpha(.7)
axs[1].set_ylabel("Edad"); axs[1].set_title("Edad por segmento (variable NO usada para agrupar)")
save("06_segmentos_ingreso_score.png")

# silueta por cliente
fig, ax = plt.subplots(figsize=(6.5, 4.6)); y0 = 0
for s in range(K):
    v = np.sort(sil[df.seg.values == s]); ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=PAL[s], alpha=.85); ax.text(-0.08, y0 + len(v) / 2, f"{s}"); y0 += len(v) + 4
ax.axvline(out["silueta_final"], color="r", ls="--"); ax.set_xlabel("Silueta"); ax.set_yticks([]); ax.set_title(f"Silueta por cliente (media = {out['silueta_final']:.3f})")
save("07_silueta_clientes.png")

# perfil radar-like: barras de z-scores
z = df.groupby("seg")[["edad", "ingreso", "score"]].mean(); z = (z - df[["edad", "ingreso", "score"]].mean()) / df[["edad", "ingreso", "score"]].std()
fig, ax = plt.subplots(figsize=(7.4, 3.8)); w = .25
for i, (c, lbl) in enumerate((("edad", "Edad"), ("ingreso", "Ingreso"), ("score", "Score"))): ax.bar(np.arange(K) + (i - 1) * w, z[c], w, label=lbl)
ax.axhline(0, c="k", lw=.8); ax.set_xticks(range(K)); ax.set_xticklabels([f"Seg. {s}" for s in range(K)]); ax.set_ylabel("Desvios respecto de la media general"); ax.legend(); ax.set_title("Perfil de cada segmento (z-scores)")
save("08_perfil_zscores.png"); z.round(2).to_csv(TAB / "10_perfil_z.csv")

# genero por segmento + chi2
ct = pd.crosstab(df.seg, df.genero); chi = stats.chi2_contingency(ct)
out["genero_por_seg"] = ct.to_dict(); out["chi2_genero"] = dict(chi2=float(chi[0]), p=float(chi[1]), dof=int(chi[2]))
# edad ANOVA/Kruskal
out["kruskal_edad"] = float(stats.kruskal(*data)[1])
# tramo de edad x segmento
tab = pd.crosstab(df.seg, df.tramo_edad); tab.to_csv(TAB / "11_tramo_edad_por_segmento.csv"); out["tramo_por_seg"] = tab.to_dict()

# ------------------------------------------------------------------ 3. Verificacion independiente
ver = {}
# (a) K-means propio (Lloyd en numpy, mismos centroides iniciales de sklearn no: kmeans++ propio con 200 inicios)
def lloyd(X, k, seed, iters=300):
    rng = np.random.default_rng(seed)
    c = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = cdist(X, np.array(c)).min(1) ** 2; c.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    c = np.array(c)
    for _ in range(iters):
        l = cdist(X, c).argmin(1); nc = np.array([X[l == j].mean(0) if (l == j).any() else c[j] for j in range(k)])
        if np.allclose(nc, c): break
        c = nc
    l = cdist(X, c).argmin(1); return l, float(((X - c[l]) ** 2).sum())
best = min((lloyd(XA, K, s) for s in range(200)), key=lambda t: t[1])
ver["lloyd_inercia"] = best[1]; ver["sklearn_inercia"] = out["inercia_final"]; ver["ARI_lloyd_vs_sklearn"] = float(adjusted_rand_score(best[0], df.seg))
# (b) silueta calculada a mano
D = cdist(XA, XA); lab = df.seg.values; s_manual = []
for i in range(len(XA)):
    same = (lab == lab[i]); same[i] = False
    a = D[i, same].mean(); b = min(D[i, lab == j].mean() for j in range(K) if j != lab[i]); s_manual.append((b - a) / max(a, b))
ver["silueta_manual"] = float(np.mean(s_manual)); ver["silueta_sklearn"] = out["silueta_final"]
# (c) centroides recalculados en unidades originales con groupby vs. inverse_transform
sc = StandardScaler().fit(df[["ingreso", "score"]]); cent_inv = sc.inverse_transform(km.cluster_centers_)
cent_inv = pd.DataFrame(cent_inv, columns=["ingreso", "score"]); cent_inv.index = [orden[i] for i in range(K)]
ver["max_dif_centroides"] = float(np.abs(cent_inv.sort_index().values - cen.sort_index().values).max())
# (d) estabilidad: 200 semillas, ARI contra el modelo final; y bootstrap 80%
aris = [adjusted_rand_score(df.seg, KMeans(K, n_init=1, random_state=s).fit_predict(XA)) for s in range(200)]
ver["ARI_200_semillas_min"] = float(np.min(aris)); ver["ARI_200_semillas_pct_perfecto"] = float(100 * np.mean(np.array(aris) > .999))
rng = np.random.default_rng(SEED); bs = []
for _ in range(200):
    idx = rng.choice(len(XA), int(.8 * len(XA)), replace=False)
    l = KMeans(K, n_init=10, random_state=int(rng.integers(1e6))).fit(XA[idx]).labels_
    bs.append(adjusted_rand_score(df.seg.values[idx], l))
ver["ARI_bootstrap80_media"] = float(np.mean(bs)); ver["ARI_bootstrap80_p5"] = float(np.percentile(bs, 5)); ver["ARI_bootstrap80_min"] = float(np.min(bs))
# (e) otros metodos
ward = AgglomerativeClustering(K, linkage="ward").fit_predict(XA)
gmm_bic = {k: GaussianMixture(k, n_init=10, random_state=SEED).fit(XA).bic(XA) for k in range(1, 11)}
gm = GaussianMixture(K, n_init=10, random_state=SEED).fit(XA).predict(XA)
ver["ARI_ward_vs_kmeans"] = float(adjusted_rand_score(ward, df.seg)); ver["ARI_gmm_vs_kmeans"] = float(adjusted_rand_score(gm, df.seg))
ver["gmm_bic"] = {int(k): float(v) for k, v in gmm_bic.items()}; ver["gmm_k_bic"] = int(min(gmm_bic, key=gmm_bic.get))
ver["silueta_ward"] = float(silhouette_score(XA, ward)); ver["silueta_gmm"] = float(silhouette_score(XA, gm))
db = []
for eps in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
    for ms in (4, 5, 8):
        l = DBSCAN(eps=eps, min_samples=ms).fit_predict(XA); nc = len(set(l)) - (1 if -1 in l else 0)
        db.append(dict(eps=eps, min_samples=ms, grupos=nc, ruido=int((l == -1).sum())))
ver["dbscan"] = db
# k=5 vs otros k: tamano de grupos y silueta; coincidencia entre k=5 y k=6, 4
for k2 in (3, 4, 6):
    l2 = KMeans(k2, n_init=50, random_state=SEED).fit_predict(XA); ver[f"ARI_k5_vs_k{k2}"] = float(adjusted_rand_score(df.seg, l2))
# conjunto B (3 variables): k sugerido y cuanto coincide con A
kB = int(mB.loc[mB.silueta.idxmax(), "k"]); out["k_max_silueta_B"] = kB
lB = KMeans(kB, n_init=50, random_state=SEED).fit_predict(XB); ver["ARI_B_vs_A"] = float(adjusted_rand_score(df.seg, lB))
ver["silueta_B_kmax"] = float(silhouette_score(XB, lB))
out["ver"] = ver

# figura de comparacion de metodos
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
for a, l, t in zip(axs, (df.seg.values, ward, gm), ("K-means (k=5)", "Jerarquico Ward (k=5)", "Mezcla gaussiana (k=5)")):
    a.scatter(df.ingreso, df.score, c=[PAL[i] for i in l], s=20); a.set_title(t); a.set_xlabel("Ingreso (k$)")
axs[0].set_ylabel("Score")
save("09_comparacion_metodos.png")
fig, ax = plt.subplots(figsize=(4.8, 3.3)); ax.plot(list(gmm_bic), list(gmm_bic.values()), "o-"); ax.set_xlabel("k"); ax.set_ylabel("BIC (menor = mejor)"); ax.set_title("Mezcla gaussiana: BIC")
save("10_gmm_bic.png")

# k=3 y k=4 y k=6 visual
fig, axs = plt.subplots(1, 4, figsize=(14, 3.5))
for a, k2 in zip(axs, (3, 4, 5, 6)):
    l2 = KMeans(k2, n_init=50, random_state=SEED).fit_predict(XA); a.scatter(df.ingreso, df.score, c=[PAL[i] for i in l2], s=16)
    a.set_title(f"k = {k2} (silueta {silhouette_score(XA, l2):.3f})"); a.set_xlabel("Ingreso (k$)")
axs[0].set_ylabel("Score")
save("11_comparacion_k.png")

# ---- Variante con edad: 3D / 3 variables
dfB = df.assign(segB=lB)
fig = plt.figure(figsize=(6.4, 5)); ax = fig.add_subplot(111, projection="3d")
for s in range(kB): d = dfB[dfB.segB == s]; ax.scatter(d.edad, d.ingreso, d.score, c=PAL[s], s=16, label=f"G{s}")
ax.set_xlabel("Edad"); ax.set_ylabel("Ingreso"); ax.set_zlabel("Score"); ax.set_title(f"Edad+Ingreso+Score, k={kB}"); ax.legend(fontsize=8)
save("12_clusters_3D_B.png")
perfB = dfB.groupby("segB").agg(n=("id", "count"), edad=("edad", "mean"), ingreso=("ingreso", "mean"), score=("score", "mean")).round(1); perfB.to_csv(TAB / "12_perfil_B.csv"); out["perfil_B"] = perfB.reset_index().to_dict("records")

# ---- Tamano economico: peso del ingreso y del score de cada segmento (indicador simple)
df["ingreso_x_score"] = df.ingreso * df.score / 100
eco = df.groupby("seg").agg(clientes=("id", "count"), ingreso_total=("ingreso", "sum"), score_prom=("score", "mean"), ing_x_score=("ingreso_x_score", "sum")).round(1)
eco["pct_ing_x_score"] = (100 * eco.ing_x_score / eco.ing_x_score.sum()).round(1); eco.to_csv(TAB / "13_peso_economico_indicativo.csv"); out["eco"] = eco.reset_index().to_dict("records")

json.dump(out, open(BASE / "outputs" / "tablas" / "resultados.json", "w"), indent=1, default=float)
print("OK")
