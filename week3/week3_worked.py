"""Week 3 worked example: Sharpe, Jensen's alpha, risk decomposition. Monthly, 2015-07..2025-06.
Training case = the eight names Week 4 already uses. Outputs are printed verbatim into the handout."""
import io, zipfile, requests, numpy as np, pandas as pd, statsmodels.api as sm, yfinance as yf, json, datetime as dt
pd.set_option("display.width", 160); pd.set_option("display.float_format", lambda x: f"{x:9.4f}")
NAMES = ["AAPL","MSFT","JNJ","KO","XOM","JPM","PG","WMT"]
BENCH = ["SPY","RSP"]
START, END = "2015-06-01", "2025-07-01"          # daily pull; monthly window becomes 2015-07..2025-06
RETRIEVED = dt.date.today().isoformat()

# ---- 1. Ken French monthly factors ----------------------------------------------------------
def french(zipname):
    r = requests.get(f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{zipname}", timeout=60); r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content)); txt = z.read(z.namelist()[0]).decode("latin1")
    return txt
txt = french("F-F_Research_Data_Factors_CSV.zip")
lines = txt.splitlines()
start = next(i for i,l in enumerate(lines) if l.strip().startswith(","))       # header ",Mkt-RF,SMB,HML,RF"
rows = []
for l in lines[start+1:]:
    p = l.split(",")
    if len(p[0].strip()) == 6 and p[0].strip().isdigit(): rows.append(p)
    elif rows: break
ff = pd.DataFrame(rows, columns=["ym","Mkt-RF","SMB","HML","RF"]).set_index("ym").astype(float)/100
ff.index = pd.PeriodIndex(ff.index, freq="M")
print("FRENCH header line:", lines[0].strip()); print("FRENCH monthly rows:", len(ff), "last:", ff.index[-1])

# ---- 2. yfinance monthly total-return proxy (auto_adjust close, month-end) ------------------
px = yf.download(NAMES+BENCH, start=START, end=END, auto_adjust=True, progress=False)["Close"]
pm = px.resample("ME").last()
rm = pm.pct_change().dropna()
rm.index = rm.index.to_period("M")
rm = rm.loc["2015-07":"2025-06"]
print("MONTHLY rows:", len(rm), rm.index[0], "->", rm.index[-1])
df = rm.join(ff, how="inner")
assert len(df) == 120, len(df)
ex = df[NAMES+BENCH].sub(df["RF"], axis=0)        # excess returns

# ---- 3. Sharpe, three ways -----------------------------------------------------------------
def sharpe(e): return e.mean()/e.std(ddof=1)*np.sqrt(12)
def cagr(r): return (1+r).prod()**(12/len(r))-1
def annvol(r): return r.std(ddof=1)*np.sqrt(12)
tab = pd.DataFrame({
    "ann_return_CAGR": [cagr(df[c]) for c in NAMES+BENCH],
    "ann_vol": [annvol(df[c]) for c in NAMES+BENCH],
    "sharpe_excess": [sharpe(ex[c]) for c in NAMES+BENCH],
    "cagr_over_vol_rf0": [cagr(df[c])/annvol(df[c]) for c in NAMES+BENCH],
}, index=NAMES+BENCH)
print("\n== TABLE A: return, vol, Sharpe (monthly, 2015-07..2025-06, 120 months) ==")
print(tab.to_string(float_format=lambda x: f"{x:8.4f}"))
print("mean monthly RF over window: %.4f%%  (annualised %.2f%%)" % (df["RF"].mean()*100, df["RF"].mean()*12*100))

# ---- 4. Equal-weight portfolio: two conventions --------------------------------------------
ew_rebal = df[NAMES].mean(axis=1)                       # monthly rebalanced to 1/8
wealth = (1+df[NAMES]).cumprod(); bh = wealth.mean(axis=1)   # buy-and-hold 1/8 at start
ew_bh = bh.pct_change(); ew_bh.iloc[0] = bh.iloc[0]-1
print("\n== TABLE B: equal-weight panel, two rebalancing conventions ==")
for nm, s in [("EW monthly-rebalanced", ew_rebal), ("EW buy-and-hold", ew_bh)]:
    e = s - df["RF"]
    print(f"{nm:24s} CAGR {cagr(s)*100:6.2f}%  vol {annvol(s)*100:6.2f}%  Sharpe {sharpe(e):6.3f}  end wealth {(1+s).prod():6.3f}")

# ---- 5. Jensen regression on Mkt-RF ---------------------------------------------------------
def jensen(y, x):
    X = sm.add_constant(x); m = sm.OLS(y, X).fit()
    a, b = m.params.iloc[0], m.params.iloc[1]
    return dict(alpha_m=a, alpha_ann=a*12, t_alpha=m.tvalues.iloc[0], beta=b, t_beta=m.tvalues.iloc[1],
                r2=m.rsquared, resid_vol_ann=np.sqrt(m.mse_resid)*np.sqrt(12), n=int(m.nobs))
res = pd.DataFrame({c: jensen(ex[c], df["Mkt-RF"]) for c in NAMES}).T
res.loc["EW panel"] = jensen(ew_rebal-df["RF"], df["Mkt-RF"])
print("\n== TABLE C: Jensen regression, excess return on French Mkt-RF (monthly) ==")
print(res.to_string(float_format=lambda x: f"{x:8.4f}"))

# ---- 6. Risk decomposition -----------------------------------------------------------------
var_m = df["Mkt-RF"].var(ddof=1)
dec = pd.DataFrame(index=NAMES+["EW panel"])
dec["beta"] = res["beta"]; dec["total_vol_ann"] = [annvol(ex[c]) for c in NAMES]+[annvol(ew_rebal-df["RF"])]
dec["systematic_vol_ann"] = (res["beta"]**2*var_m*12)**0.5
dec["idio_vol_ann"] = res["resid_vol_ann"]
dec["systematic_share"] = res["r2"]
print("\n== TABLE D: risk decomposition (annualised) — total² = systematic² + idiosyncratic² ==")
print(dec.to_string(float_format=lambda x: f"{x:8.4f}"))
print("market (Mkt-RF) annualised vol: %.4f" % np.sqrt(var_m*12))
c = "AAPL"; print(f"check {c}: sqrt({dec.loc[c,'systematic_vol_ann']:.4f}^2 + {dec.loc[c,'idio_vol_ann']:.4f}^2) = {np.sqrt(dec.loc[c,'systematic_vol_ann']**2+dec.loc[c,'idio_vol_ann']**2):.4f} vs total {dec.loc[c,'total_vol_ann']:.4f}")

# ---- 7. Alternative market proxy: SPY and RSP excess ---------------------------------------
alt = pd.DataFrame({f"vs {b}": {c: jensen(ex[c], ex[b])["alpha_ann"] for c in NAMES} for b in BENCH})
alt["vs Mkt-RF"] = res.loc[NAMES, "alpha_ann"]
alt.loc["EW panel"] = [jensen(ew_rebal-df["RF"], ex[b])["alpha_ann"] for b in BENCH] + [res.loc["EW panel","alpha_ann"]]
print("\n== TABLE E: annualised alpha depends on which market you regress on ==")
print(alt[["vs Mkt-RF","vs SPY","vs RSP"]].to_string(float_format=lambda x: f"{x:8.4f}"))
beta_alt = pd.DataFrame({f"vs {b}": {c: jensen(ex[c], ex[b])["beta"] for c in NAMES} for b in BENCH}); beta_alt["vs Mkt-RF"]=res.loc[NAMES,"beta"]
print(beta_alt[["vs Mkt-RF","vs SPY","vs RSP"]].to_string(float_format=lambda x: f"{x:8.4f}"))

# ---- 8. Verification: 5-year monthly beta vs S&P 500 (Yahoo's stated convention) ------------
print("\n== TABLE F: verification — 5-year monthly beta vs SPY (Yahoo convention) ==")
last60 = ex.loc["2020-07":"2025-06"]
for c in ["AAPL","KO","XOM"]:
    b5 = jensen(last60[c], last60["SPY"])["beta"]
    raw5 = df.loc["2020-07":"2025-06"]
    b5raw = jensen(raw5[c], raw5["SPY"])["beta"]      # raw returns, no RF subtraction
    try: pub = yf.Ticker(c).info.get("beta")
    except Exception as e: pub = f"ERR {e}"
    print(f"{c}: mine (excess, 60m to 2025-06) {b5:.3f}   mine (raw returns) {b5raw:.3f}   Yahoo published beta (5Y monthly) {pub}   retrieved {RETRIEVED}")

# ---- 9. Extension: FF3 and daily-vs-monthly ------------------------------------------------
X3 = sm.add_constant(df[["Mkt-RF","SMB","HML"]])
print("\n== TABLE G: FF3 extension ==")
for c in ["AAPL","KO","XOM","JPM"]:
    m = sm.OLS(ex[c], X3).fit()
    print(f"{c}: alpha_ann {m.params['const']*12:7.4f}  bMkt {m.params['Mkt-RF']:6.3f}  bSMB {m.params['SMB']:6.3f}  bHML {m.params['HML']:6.3f}  R2 {m.rsquared:.3f}   (CAPM alpha_ann {res.loc[c,'alpha_ann']:7.4f}, R2 {res.loc[c,'r2']:.3f})")
txtd = french("F-F_Research_Data_Factors_daily_CSV.zip"); ld = txtd.splitlines()
sd = next(i for i,l in enumerate(ld) if l.strip().startswith(","))
rows=[l.split(",") for l in ld[sd+1:] if len(l.split(",")[0].strip())==8]
ffd = pd.DataFrame(rows, columns=["d","Mkt-RF","SMB","HML","RF"]).set_index("d").astype(float)/100
ffd.index = pd.to_datetime(ffd.index); rd = px.pct_change().dropna().join(ffd, how="inner").loc["2015-07-01":"2025-06-30"]
print("\n== TABLE H: daily vs monthly beta on Mkt-RF, same window ==")
for c in ["AAPL","KO","XOM"]:
    bd = jensen(rd[c]-rd["RF"], rd["Mkt-RF"]); print(f"{c}: monthly beta {res.loc[c,'beta']:.3f} (n={int(res.loc[c,'n'])})   daily beta {bd['beta']:.3f} (n={bd['n']})   daily R2 {bd['r2']:.3f} vs monthly R2 {res.loc[c,'r2']:.3f}")

# ---- 10. Two covariance matrices for Week 4 ------------------------------------------------
S_sample = ex[NAMES].cov(ddof=1)*12
B = res.loc[NAMES,"beta"].values.reshape(-1,1)
S_si = pd.DataFrame(B@B.T*var_m*12 + np.diag(res.loc[NAMES,"resid_vol_ann"].values**2), index=NAMES, columns=NAMES)
corr_s = ex[NAMES].corr(); corr_si = S_si/np.sqrt(np.outer(np.diag(S_si),np.diag(S_si)))
print("\n== TABLE I: sample vs single-index covariance (annualised), 3x3 corner ==")
print("sample:\n", S_sample.iloc[:3,:3].to_string(float_format=lambda x: f"{x:8.4f}"))
print("single-index:\n", S_si.iloc[:3,:3].to_string(float_format=lambda x: f"{x:8.4f}"))
print("sample corr AAPL-MSFT %.3f  KO-XOM %.3f ; single-index corr AAPL-MSFT %.3f  KO-XOM %.3f" % (corr_s.loc["AAPL","MSFT"], corr_s.loc["KO","XOM"], corr_si.loc["AAPL","MSFT"], corr_si.loc["KO","XOM"]))
print("condition number: sample %.1f  single-index %.1f" % (np.linalg.cond(S_sample.values), np.linalg.cond(S_si.values)))
# Week 4 handoff files
ex[NAMES].to_csv("returns_excess_monthly.csv"); S_sample.to_csv("cov_sample_annual.csv"); S_si.to_csv("cov_single_index_annual.csv")
df[NAMES+BENCH+["Mkt-RF","SMB","HML","RF"]].to_csv("panel_monthly_with_factors.csv")
print("\nfiles:", "returns_excess_monthly.csv cov_sample_annual.csv cov_single_index_annual.csv panel_monthly_with_factors.csv")
print("\nSPY-RSP 10y monthly: SPY CAGR %.2f%% RSP CAGR %.2f%%  gap %.2f pp/yr;  Sharpe(excess) SPY %.3f RSP %.3f" % (tab.loc["SPY","ann_return_CAGR"]*100, tab.loc["RSP","ann_return_CAGR"]*100, (tab.loc["SPY","ann_return_CAGR"]-tab.loc["RSP","ann_return_CAGR"])*100, tab.loc["SPY","sharpe_excess"], tab.loc["RSP","sharpe_excess"]))
