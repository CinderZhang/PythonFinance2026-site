"""Week 5 worked example on the Week 4 portfolio: what it is exposed to, what keeping it costs,
and how bad it got (worst month, maximum drawdown). Monthly, eight names, 2015-07..2025-06.
Inputs: panel_monthly_with_factors.csv (Week 3) and week4_weights_longonly_cap20.csv (Week 4 Step 5)."""
import numpy as np, pandas as pd, statsmodels.api as sm
pd.set_option("display.width", 160)
NAMES = ["AAPL","MSFT","JNJ","KO","XOM","JPM","PG","WMT"]; N = len(NAMES)
df = pd.read_csv("panel_monthly_with_factors.csv", index_col=0); df.index = pd.PeriodIndex(df.index, freq="M")
w = pd.read_csv("week4_weights_longonly_cap20.csv", index_col=0).iloc[:,0].reindex(NAMES).values
print("Week 4 portfolio (long-only, cap 20%, full window):", dict(zip(NAMES, (w*100).round(1))), "gross %.2f" % np.abs(w).sum())
R = df[NAMES]; rf = df["RF"]
port = pd.Series(R.values @ w, index=df.index, name="portfolio")     # fixed weights, reset monthly
ew   = R.mean(axis=1).rename("1/N")
spy  = df["SPY"].rename("SPY")
def ann(r): return dict(ret=r.mean()*12, vol=r.std(ddof=1)*np.sqrt(12), sharpe=(r-rf).mean()/(r-rf).std(ddof=1)*np.sqrt(12))
for nm, s in [("portfolio", port), ("1/N", ew), ("SPY", spy)]:
    a = ann(s); print(f"{nm:10s} ann mean {a['ret']*100:6.2f}%  vol {a['vol']*100:6.2f}%  Sharpe {a['sharpe']:.3f}  (in-sample, full window)")

# ---- TABLE 1: what the portfolio is exposed to (Week 3's regression, on the portfolio) ----
print("\n== TABLE 1: factor exposure, monthly excess returns on French factors, 120 months ==")
def reg(y, cols):
    X = sm.add_constant(df[cols]); m = sm.OLS(y - rf, X).fit()
    out = {"alpha_ann": m.params["const"]*12, "t_alpha": m.tvalues["const"], "R2": m.rsquared}
    for c in cols: out[f"b_{c}"] = m.params[c]; out[f"t_{c}"] = m.tvalues[c]
    out["resid_vol_ann"] = np.sqrt(m.mse_resid*12); return out
rows = []
for nm, s in [("portfolio", port), ("1/N", ew), ("SPY", spy)]:
    for label, cols in [("market", ["Mkt-RF"]), ("FF3", ["Mkt-RF","SMB","HML"])]:
        o = reg(s, cols); o["series"] = nm; o["model"] = label; rows.append(o)
t1 = pd.DataFrame(rows).set_index(["series","model"])
print(t1[["alpha_ann","t_alpha","b_Mkt-RF","t_Mkt-RF","b_SMB","t_SMB","b_HML","t_HML","R2","resid_vol_ann"]].round(3).to_string())
# variance decomposition for the portfolio vs the average single name (Week 3's decomposition, now diversified)
m_port = sm.OLS(port-rf, sm.add_constant(df["Mkt-RF"])).fit()
single_r2 = [sm.OLS(df[c]-rf, sm.add_constant(df["Mkt-RF"])).fit().rsquared for c in NAMES]
print("share of variance the market explains: portfolio %.1f%% ; average single name %.1f%% (range %.1f%%..%.1f%%)" % (m_port.rsquared*100, np.mean(single_r2)*100, min(single_r2)*100, max(single_r2)*100))
print("total ann vol %.2f%% = market part %.2f%% + idiosyncratic part %.2f%% (in variance: %.4f = %.4f + %.4f)" % (
    port.std(ddof=1)*np.sqrt(12)*100, abs(m_port.params["Mkt-RF"])*df["Mkt-RF"].std(ddof=1)*np.sqrt(12)*100, np.sqrt(m_port.mse_resid*12)*100,
    port.var(ddof=1)*12, m_port.params["Mkt-RF"]**2*df["Mkt-RF"].var(ddof=1)*12, m_port.mse_resid*12))

# ---- TABLE 2: what keeping it costs — four rebalancing policies on the same weights ----
print("\n== TABLE 2: rebalancing policies, 120 months, start at target weights ==")
def simulate(target, R, every=None):
    """every=None: never rebalance (weights drift). every=k: reset to target every k months.
    Returns monthly portfolio returns, one-way turnover per rebalance event (sum|dw|/2), and max drift."""
    wcur = target.copy(); rets=[]; turns=[]; drift=[]
    for t in range(len(R)):
        r = R.iloc[t].values; g = 1 + r
        rp = wcur @ r; rets.append(rp)
        wcur = wcur * g / (wcur @ g)                      # drift with prices
        drift.append(np.abs(wcur - target).max())
        if every and (t+1) % every == 0 and t+1 < len(R):
            turns.append(np.abs(target - wcur).sum()/2); wcur = target.copy()
    return pd.Series(rets, index=R.index), np.array(turns), np.array(drift), wcur
rows=[]
for nm, k in [("monthly",1),("quarterly",3),("annual",12),("never",None)]:
    s, turns, drift, wend = simulate(w, R, k); a = ann(s)
    yrs = len(R)/12; annual_turnover = turns.sum()/yrs if len(turns) else 0.0
    rows.append(dict(policy=nm, rebalances=len(turns), annual_one_way_turnover=annual_turnover, drag_bp_at_10bp_roundtrip=annual_turnover*10, drag_bp_at_50bp_roundtrip=annual_turnover*50,
                     ann_ret=a["ret"], ann_vol=a["vol"], sharpe=a["sharpe"], max_drift_pp=drift.max()*100, largest_end_weight=wend.max(), end_name=NAMES[wend.argmax()]))
t2 = pd.DataFrame(rows).set_index("policy"); print(t2.round(4).to_string())
print("(one-way turnover = sum|w_target - w_drifted| / 2 at each reset; drag = annual turnover x round-trip cost; cost figures are assumptions, labelled)")

# ---- TABLE 3: one honest risk line — worst month and maximum drawdown, with the recipe shown ----
print("\n== TABLE 3: worst month and maximum drawdown (monthly-reset series, the Week 4 convention) ==")
def drawdown(r):
    wealth = (1+r).cumprod()                  # 1. wealth path: $1 compounded through total returns
    peak = wealth.cummax()                    # 2. running peak
    dd = wealth/peak - 1                      # 3. drawdown from the running peak, every month
    trough = dd.idxmin(); pk = wealth.loc[:trough].idxmax()
    rec = wealth.loc[trough:]; back = rec[rec >= peak.loc[trough]]
    months_to_recover = (back.index[0] - trough).n if len(back) else None   # 4. None = not recovered by window end
    return dd.min(), pk, trough, months_to_recover, wealth, peak
for nm, s in [("portfolio", port), ("1/N", ew), ("SPY", spy)]:
    mdd, pk, tr, rec, wealth, peak = drawdown(s)
    print(f"{nm:10s} worst month {s.idxmin()} {s.min()*100:6.2f}%   max drawdown {mdd*100:6.2f}% (peak {pk} wealth {wealth.loc[pk]:.3f} -> trough {tr} wealth {wealth.loc[tr]:.3f}; recovered after {rec} months)")
mdd, pk, tr, rec, wealth, peak = drawdown(port)
print("recipe check on the portfolio: %.3f / %.3f - 1 = %.4f" % (wealth.loc[tr], wealth.loc[pk], wealth.loc[tr]/wealth.loc[pk]-1))
print("second-worst drawdown episode (COVID): peak 2020-01 wealth %.3f -> 2020-03 wealth %.3f = %.2f%%" % (wealth.loc["2020-01"], wealth.loc["2020-03"], (wealth.loc["2020-03"]/wealth.loc["2020-01"]-1)*100))
print("months in drawdown (below the running peak): %d of %d" % ((wealth < peak).sum(), len(wealth)))
# hand-over sheet digits
print("\n== HAND-OVER SHEET (the shape, not your answer) ==")
o = reg(port, ["Mkt-RF","SMB","HML"]); a = ann(port); mdd, pk, tr, rec, _, _ = drawdown(port)
print(f"weights: " + ", ".join(f"{n} {x*100:.0f}%" for n,x in zip(NAMES,w) if x>1e-4) + f" | gross {np.abs(w).sum():.2f} | constraints: long-only, cap 20%")
print(f"exposure (FF3, 120 months): market beta {o['b_Mkt-RF']:.2f}, size {o['b_SMB']:+.2f}, value {o['b_HML']:+.2f}, alpha {o['alpha_ann']*100:+.1f}%/yr (t={o['t_alpha']:.1f}), R2 {o['R2']:.2f}")
print(f"policy: quarterly reset | one-way turnover {t2.loc['quarterly','annual_one_way_turnover']*100:.1f}%/yr | drag {t2.loc['quarterly','drag_bp_at_10bp_roundtrip']:.1f} bp/yr at an assumed 10 bp round trip")
print(f"how bad: worst month {port.idxmin()} {port.min()*100:.1f}% | max drawdown {mdd*100:.1f}% ({pk}->{tr}, recovered in {rec} months) | all statistics on the monthly-reset series")
