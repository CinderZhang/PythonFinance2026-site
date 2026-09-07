"""Week 4 worked example on the Week 3 panel: tangency on two windows, out-of-sample vs 1/N,
long-only with caps. Monthly, eight names, 2015-07..2025-06. Inputs: panel_monthly_with_factors.csv."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
pd.set_option("display.width", 160)
NAMES = ["AAPL","MSFT","JNJ","KO","XOM","JPM","PG","WMT"]; N=len(NAMES)
df = pd.read_csv("panel_monthly_with_factors.csv", index_col=0)
ex = df[NAMES].sub(df["RF"], axis=0); exspy = df["SPY"]-df["RF"]
A, B = ex.loc["2015-07":"2020-06"], ex.loc["2020-07":"2025-06"]
print("window A", A.index[0], A.index[-1], len(A), "| window B", B.index[0], B.index[-1], len(B))
print("parameters: cov %d + means %d = %d ; observations per window %d ; obs per parameter %.2f" % (N*(N+1)//2, N, N*(N+1)//2+N, len(A), len(A)/(N*(N+1)//2+N)))
def stats(r):  # r = monthly excess return series of a portfolio
    return dict(ann_ret=r.mean()*12, ann_vol=r.std(ddof=1)*np.sqrt(12), sharpe=r.mean()/r.std(ddof=1)*np.sqrt(12))
def tangency(w_ex):
    mu = w_ex.mean().values*12; S = w_ex.cov(ddof=1).values*12
    w = np.linalg.solve(S, mu); return w/w.sum()
def port(w, R): return R.values@w
wA, wB = tangency(A), tangency(B)
tab = pd.DataFrame({"window A":wA, "window B":wB}, index=NAMES); tab["swing pp"]=(tab["window B"]-tab["window A"])*100
print("\n== TABLE 1: unconstrained tangency weights (excess returns), two windows ==")
print((tab*[100,100,1]).round(1).to_string()); print("gross leverage: A %.2f  B %.2f" % (np.abs(wA).sum(), np.abs(wB).sum()))
print("largest long A: %s %.0f%% ; largest short A: %s %.0f%%" % (NAMES[wA.argmax()], wA.max()*100, NAMES[wA.argmin()], wA.min()*100))
sA_in = stats(pd.Series(port(wA,A))); print("in-sample (A on A): ret %.2f%% vol %.2f%% Sharpe %.3f" % (sA_in["ann_ret"]*100, sA_in["ann_vol"]*100, sA_in["sharpe"]))
# out of sample: fixed weights, reset monthly (constant-mix), no re-estimation
ew = np.ones(N)/N
print("\n== TABLE 2: out of sample on window B, weights fixed from A, reset monthly ==")
for nm, w in [("tangency (from A)", wA), ("1/N", ew)]:
    s = stats(pd.Series(port(w,B))); print(f"{nm:20s} ret {s['ann_ret']*100:6.2f}%  vol {s['ann_vol']*100:6.2f}%  Sharpe {s['sharpe']:6.3f}  gross {np.abs(w).sum():.2f}")
s_mvo = stats(pd.Series(port(wA,B))); s_ew = stats(pd.Series(port(ew,B)))
print("1/N beat the optimizer's out-of-sample Sharpe by a factor of %.1f" % (s_ew["sharpe"]/s_mvo["sharpe"]) if s_mvo["sharpe"]>0 else "optimizer out-of-sample Sharpe is negative")
# in-sample Sharpe of 1/N on A, for the honest comparison
print("in-sample on A: tangency %.3f vs 1/N %.3f" % (sA_in["sharpe"], stats(pd.Series(port(ew,A)))["sharpe"]))
# buy-and-hold variant for the convention note
def bh(w, R):
    wealth = (1+R).cumprod().values@w  # shares fixed at t0, value path
    return pd.Series(wealth).pct_change().fillna(wealth[0]-1)
print("convention check on B: tangency buy-and-hold Sharpe %.3f vs reset-monthly %.3f ; min wealth on path %.2f" % (stats(bh(wA,B))["sharpe"], s_mvo["sharpe"], ((1+B).cumprod().values@wA).min()))
# ---- remedy: long-only with position cap ----
def maxsharpe(w_ex, cap):
    mu = w_ex.mean().values*12; S = w_ex.cov(ddof=1).values*12
    f = lambda w: -(w@mu)/np.sqrt(w@S@w)
    res = minimize(f, np.ones(N)/N, bounds=[(0,cap)]*N, constraints=[{"type":"eq","fun":lambda w: w.sum()-1}], method="SLSQP", options={"ftol":1e-12,"maxiter":500})
    assert res.success, res.message; return res.x
print("\n== TABLE 3: long-only with cap, estimated on A, tested on B ==")
rows=[]
for cap in [1.0, 0.25, 0.20, 0.15]:
    w = maxsharpe(A, cap); sin = stats(pd.Series(port(w,A))); sout = stats(pd.Series(port(w,B)))
    rows.append(dict(cap=cap, names_held=int((w>1e-4).sum()), largest=w.max(), in_sharpe=sin["sharpe"], oos_ret=sout["ann_ret"], oos_vol=sout["ann_vol"], oos_sharpe=sout["sharpe"]))
    if cap==0.20: w20A=w
r = pd.DataFrame(rows).set_index("cap"); print(r.round(3).to_string())
print("1/N out-of-sample Sharpe for reference: %.3f" % s_ew["sharpe"])
print("cap 20%% weights from A:", dict(zip(NAMES, (w20A*100).round(1))))
w20B = maxsharpe(B, 0.20); print("cap 20%% weights from B:", dict(zip(NAMES, (w20B*100).round(1)))); print("cap-20 swing max %.1f pp vs unconstrained max %.1f pp" % ((np.abs(w20B-w20A)*100).max(), (np.abs(wB-wA)*100).max()))
# full-sample long-only cap 20 for Week 5 handoff
wF = maxsharpe(ex, 0.20); print("\nfull-sample (120m) long-only cap 20%% weights:", dict(zip(NAMES,(wF*100).round(1))), "gross %.2f" % np.abs(wF).sum())
sF = stats(pd.Series(port(wF, ex))); print("full-sample in-sample: ret %.2f%% vol %.2f%% Sharpe %.3f (in-sample, labelled)" % (sF["ann_ret"]*100, sF["ann_vol"]*100, sF["sharpe"]))
pd.Series(wF, index=NAMES).to_csv("week4_weights_longonly_cap20.csv")
