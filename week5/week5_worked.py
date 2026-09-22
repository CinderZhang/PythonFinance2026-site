"""Week 5 worked example: the training case ABG, valued end to end, the way the course videos value it.

Two layers, deliberately separate, because that separation IS this week's lesson:
  COMPANY  — the filed lines and the nineteen labelled assumptions for one company.
  METHOD   — project(), checks(), dcf(), tornado(). Knows nothing about ABG.
Swap the COMPANY layer and the same METHOD runs the second company. That is "one pipeline, run twice".

Data: SEC company-facts API (free, no key) for the filed lines; four numbers typed from the 10-K
itself, with their source on the line, because the feed does not carry them.
Run: uv run --with pandas python week5_worked.py
ABG is the demonstration company. Nothing here is investment advice.
"""
import json, os, urllib.request, pathlib, pandas as pd
pd.set_option("display.width", 200)

# ----------------------------------------------------------------------------- COMPANY LAYER
CIK, TICKER, NAME = "0001144980", "ABG", "Asbury Automotive Group"
UA = os.environ.get("SEC_USER_AGENT", "FIN69000 student your.name@purdue.edu")

def company_facts(cik):
    cache = pathlib.Path(f"companyfacts_{cik}.json")
    if not cache.exists():
        req = urllib.request.Request(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers={"User-Agent": UA})
        cache.write_bytes(urllib.request.urlopen(req).read())
    return json.loads(cache.read_text())

D = company_facts(CIK); F = D["facts"]["us-gaap"]

def annual(concept, unit="USD"):
    rows = [f for f in F[concept]["units"][unit] if f.get("form") == "10-K"]
    df = pd.DataFrame(rows)
    if "start" in df:
        days = (pd.to_datetime(df["end"]) - pd.to_datetime(df["start"])).dt.days + 1
        df = df[days.isna() | days.between(340, 380)]
    return df.sort_values(["end", "filed"])

def as_first_reported(concept, end):
    """The figure in the 10-K for that year itself, not a later restatement. Returns (value, filed, accession)."""
    s = annual(concept); s = s[s["end"] == end]
    if s.empty: return None, None, None
    r = s.iloc[0]; return r["val"], r["filed"], r["accn"]

def revised_later(concept, end):
    s = annual(concept); s = s[s["end"] == end]
    return (not s.empty) and s["val"].nunique() > 1

ENDS = ["2023-12-31", "2024-12-31", "2025-12-31"]
LINES = {"revenue":"Revenues", "gross profit":"GrossProfit", "SG&A":"SellingGeneralAndAdministrativeExpense",
         "operating income":"OperatingIncomeLoss", "pretax income":"IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
         "income tax":"IncomeTaxExpenseBenefit", "net income":"NetIncomeLoss", "D&A":"DepreciationDepletionAndAmortization",
         "asset impairments":"AssetImpairmentCharges", "operating cash flow":"NetCashProvidedByUsedInOperatingActivities",
         "inventory":"InventoryNet", "PP&E, net":"PropertyPlantAndEquipmentNet", "shareholders' equity":"StockholdersEquity",
         "total assets":"Assets"}
# ABG does not tag total liabilities; it is assets less equity, and the 0.2 gap is the noncontrolling interest.

# Four numbers the company-facts feed does not carry for this company after FY2023. Typed from the filing.
FROM_THE_FILING = {
  "capex":      {"2023": 142.0, "2024": 308.0, "2025": 205.0, "source": "FY2025 10-K, Consolidated Statements of Cash Flows, 'Capital expenditures'; accession 0001144980-26-000051"},
  "cash":       {"2025": 40.4,   "source": "FY2025 10-K, Consolidated Balance Sheets, 'Cash and cash equivalents'"},
  "floorplan":  {"2025": 2027.0, "source": "FY2025 10-K, Consolidated Balance Sheets, 343.1 trade + 1,683.9 non-trade"},
  "debt":       {"2025": 3572.0, "source": "FY2025 10-K, Consolidated Balance Sheets, 479.2 current + 3,092.8 long-term"},
  "shares":     {"count": 17_951_349, "source": "10-Q at 2026-06-30, 40,099,276 issued less 22,147,927 treasury; accession 0001144980-26-000094"},
}

def driver(value, label, note):
    return {"value": value, "label": label, "note": note}

# The nineteen assumptions of Video 3, each with the label that tells a reader which rows to argue with.
DRIVERS = {
 "growth":          driver(0.018,  "judgment", "organic; reported growth is two acquisitions, same-store about 1.2%"),
 "gross_margin":    driver(0.1705, "history",  "18.6 / 17.2 / 17.1% FY2023-25; the last two years, not the 2022 peak"),
 "sga_over_gp":     driver([0.665, 0.655, 0.645, 0.645, 0.645], "history then judgment", "58.7 -> 64.0 -> 64.7% history; a glide back to 64.5%"),
 "da_over_ppe":     driver(82.4/3070.4, "history", "FY2025 D&A over closing PP&E, applied to opening PP&E"),
 "impairment":      driver(120.0,  "judgment", "117 / 150 / 141 history; treated as a recurring non-cash charge"),
 "capex":           driver(250.0,  "guidance", "management guidance; first-half 2026 ran about 339 annualised"),
 "tax":             driver(0.255,  "judgment", "history 24.8 / 25.2 / 25.7%"),
 "inventory_days":  driver(2135.8/(17999.0-3071.7)*365, "history", "FY2025 inventory over cost of sales"),
 "floorplan_ratio": driver(2027.0/2135.8, "history", "floor-plan notes over inventory"),
 "other_wc_ratio":  driver(0.008,  "judgment", "other working capital, on the change in revenue"),
 "rate_floorplan":  driver(0.0467, "history proxy", "floor-plan interest over opening floor plan"),
 "rate_debt":       driver(0.0544, "history proxy", "term-debt interest over opening debt"),
 "rate_revolver":   driver(0.06,   "judgment", "assumed contract spread"),
 "revolver_cap":    driver(850.0,  "judgment", "assumed capacity, not a filed facility size"),
 "min_cash":        driver(25.0,   "history", "the floor the company has operated at"),
 "paydown":         driver(150.0,  "judgment", "term-debt repayment, explicit years only"),
 "buyback":         driver(150.0,  "judgment", "a distribution; financing, so outside free cash flow to equity"),
 "cost_of_equity":  driver(0.10,   "judgment", "tested in Video 4: 8.15% to 11%"),
 "terminal_growth": driver(0.025,  "judgment", "below the discount rate, checked in code"),
}

# ----------------------------------------------------------------------------- METHOD LAYER (company-agnostic)
def project(opening, d, years=5):
    """Five years of statements. Cash is computed LAST, from the cash-flow statement. Never a plug."""
    v = {k: (x["value"] if isinstance(x, dict) else x) for k, x in d.items()}
    prev = dict(opening); rows = []
    for i in range(years):
        y = opening["year"] + 1 + i
        revenue = prev["revenue"] * (1 + v["growth"])
        gp      = revenue * v["gross_margin"]
        cogs    = revenue - gp
        sga     = gp * v["sga_over_gp"][i]
        da      = prev["ppe"] * v["da_over_ppe"]
        ebit    = gp - sga - da - v["impairment"]
        interest = prev["floorplan"] * v["rate_floorplan"] + prev["debt"] * v["rate_debt"] + prev["revolver"] * v["rate_revolver"]
        pretax  = ebit - interest
        tax     = max(0.0, pretax) * v["tax"]
        ni      = pretax - tax
        inventory = cogs / 365 * v["inventory_days"]
        floorplan = inventory * v["floorplan_ratio"]
        ppe     = prev["ppe"] + v["capex"] - da
        delta_wc = (revenue - prev["revenue"]) * v["other_wc_ratio"]
        # the impairment is a non-cash write-down of the other (intangible) assets; the WC change rolls here too
        other_assets = prev["other_assets"] + delta_wc - v["impairment"]
        paydown = min(v["paydown"], prev["debt"]); debt = prev["debt"] - paydown
        fcfe = (ni + da + v["impairment"] - v["capex"] - (inventory - prev["inventory"])
                - delta_wc + (floorplan - prev["floorplan"]) - paydown)
        cash_before = prev["cash"] + fcfe - v["buyback"]
        draw  = min(max(0.0, v["min_cash"] - cash_before), max(0.0, v["revolver_cap"] - prev["revolver"]))
        repay = min(prev["revolver"], max(0.0, cash_before - v["min_cash"]))
        revolver = prev["revolver"] + draw - repay
        cash = cash_before + draw - repay                      # computed last
        equity = prev["equity"] + ni - v["buyback"]
        assets = cash + inventory + ppe + other_assets
        liabs_equity = floorplan + debt + revolver + prev["other_liabilities"] + equity
        row = dict(year=y, revenue=revenue, gp=gp, sga=sga, da=da, ebit=ebit, interest=interest, pretax=pretax,
                   tax=tax, ni=ni, capex=v["capex"], inventory=inventory, floorplan=floorplan, ppe=ppe, debt=debt,
                   revolver=revolver, cash=cash, equity=equity, fcfe=fcfe, paydown=paydown, assets=assets,
                   other_assets=other_assets, liabs_equity=liabs_equity,
                   balance_residual=assets - liabs_equity,
                   cash_tie=cash - (prev["cash"] + fcfe - v["buyback"] + draw - repay),
                   cash_floor_gap=cash - v["min_cash"] + revolver)
        rows.append(row)
        prev = {**prev, **{k: row[k] for k in ("revenue","inventory","floorplan","ppe","debt","revolver","cash","equity","other_assets")}}
    return rows

def checks(rows, tol=0.05):
    """The check block. Returns True only when every year passes every test."""
    out = {}
    for r in rows:
        out[r["year"]] = {"balance": abs(r["balance_residual"]) < tol,
                          "cash tie": abs(r["cash_tie"]) < tol,
                          "cash floor": r["cash_floor_gap"] > -tol}
    return out

def all_green(rows): return all(all(v.values()) for v in checks(rows).values())

def dcf(rows, d, shares):
    """Equity DCF. Refuses to run on an impossible input or on a model whose checks do not pass."""
    r = d["cost_of_equity"]["value"]; g = d["terminal_growth"]["value"]
    if g >= r: raise ValueError(f"terminal growth g={g:.3%} must be below the discount rate r={r:.3%} — the model refuses to run")
    if not all_green(rows): raise ValueError("check block does not pass — no balance, no valuation")
    n = len(rows)
    pv_explicit = sum(x["fcfe"] / (1 + r) ** i for i, x in enumerate(rows, 1))
    terminal_cf = (rows[-1]["fcfe"] + rows[-1]["paydown"]) * (1 + g)   # the paydown is finite: restore it
    pv_terminal = terminal_cf / (r - g) / (1 + r) ** n
    equity = pv_explicit + pv_terminal
    return dict(pv_explicit=pv_explicit, terminal_cf=terminal_cf, pv_terminal=pv_terminal,
                equity=equity, per_share=equity * 1e6 / shares, terminal_share=pv_terminal / equity)

def revalue(opening, d, shares, **overrides):
    """Rerun the WHOLE model with drivers changed. Never add sensitivities; rerun them."""
    d2 = {k: dict(v) for k, v in d.items()}
    for k, val in overrides.items(): d2[k]["value"] = val
    return dcf(project(opening, d2), d2, shares)["per_share"]

# ----------------------------------------------------------------------------- RUN
print(f"{D['entityName']}  CIK {D['cik']}  ({TICKER})   company facts pulled {pd.Timestamp.today():%Y-%m-%d}\n")

print("== TABLE 1: the filed lines, $m, as first reported in that year's own 10-K ==")
hist = {}
for label, concept in LINES.items():
    vals, filed, accn, rev = [], None, None, []
    for e in ENDS:
        v, f, a = as_first_reported(concept, e); vals.append(v)
        if e == ENDS[-1]: filed, accn = f, a
        if revised_later(concept, e): rev.append(e[:4])
    hist[label] = vals
    cells = "  ".join(f"{(v or 0)/1e6:9,.1f}" if v is not None else f"{'n/a':>9}" for v in vals)
    print(f"{label:22s} {cells}   FY2025 filed {filed}  {'REVISED later: '+','.join(rev) if rev else 'unchanged in later 10-Ks'}")
print(f"capex                  {FROM_THE_FILING['capex']['2023']:9,.1f}  {FROM_THE_FILING['capex']['2024']:9,.1f}  {FROM_THE_FILING['capex']['2025']:9,.1f}   not in the feed after FY2023 — typed from the filing")
print(f"  source: {FROM_THE_FILING['capex']['source']}")

h = pd.DataFrame(hist, index=ENDS).astype(float) / 1e6
print("\n== TABLE 2: history ratios — the 'history' rows of the ledger ==")
ratios = pd.DataFrame({
  "revenue growth %": h["revenue"].pct_change()*100,
  "gross margin %":   h["gross profit"]/h["revenue"]*100,
  "SG&A / GP %":      h["SG&A"]/h["gross profit"]*100,
  "tax rate %":       h["income tax"]/h["pretax income"]*100,
  "inventory days":   h["inventory"]/(h["revenue"]-h["gross profit"])*365,
})
print(ratios.round(1).to_string())

OPENING = dict(year=2025, revenue=float(h.loc["2025-12-31","revenue"]), inventory=float(h.loc["2025-12-31","inventory"]),
               ppe=float(h.loc["2025-12-31","PP&E, net"]), equity=float(h.loc["2025-12-31","shareholders' equity"]) - 0.2,
               cash=FROM_THE_FILING["cash"]["2025"], floorplan=FROM_THE_FILING["floorplan"]["2025"],
               debt=FROM_THE_FILING["debt"]["2025"], revolver=0.0,
               other_assets=float(h.loc["2025-12-31","total assets"]) - FROM_THE_FILING["cash"]["2025"] - float(h.loc["2025-12-31","inventory"]) - float(h.loc["2025-12-31","PP&E, net"]),
               other_liabilities=(float(h.loc["2025-12-31","total assets"]) - (float(h.loc["2025-12-31","shareholders' equity"]) - 0.2))
                                 - FROM_THE_FILING["floorplan"]["2025"] - FROM_THE_FILING["debt"]["2025"])
print("\n== opening balance sheet at 2025-12-31, $m ==")
print("  " + "  ".join(f"{k} {v:,.1f}" for k, v in OPENING.items() if k != "year"))

rows = project(OPENING, DRIVERS)
print("\n== TABLE 3: the five-year pro-forma, $m — cash computed LAST ==")
df = pd.DataFrame(rows).set_index("year")
print(df[["revenue","gp","sga","da","ebit","interest","ni","capex","inventory","floorplan","ppe","debt","revolver","cash","equity","fcfe"]].round(1).to_string())

print("\n== TABLE 4: the check block — no balance, no valuation ==")
for y, c in checks(rows).items():
    r = df.loc[y]
    print(f"  {y}  balance residual {r['balance_residual']:+.4f}  cash tie {r['cash_tie']:+.4f}  cash - floor {r['cash_floor_gap']:+8.1f}   " + ("PASS" if all(c.values()) else "FAIL"))
print(f"  all green: {all_green(rows)}")

SHARES = FROM_THE_FILING["shares"]["count"]
b = dcf(rows, DRIVERS, SHARES)
print("\n== TABLE 5: the base case ==")
print(f"  PV of five years at {DRIVERS['cost_of_equity']['value']:.1%}          {b['pv_explicit']:9,.1f}")
print(f"  terminal cash flow (fcfe + paydown restored) x (1+g)  {b['terminal_cf']:9,.1f}")
print(f"  discounted terminal value                             {b['pv_terminal']:9,.1f}")
print(f"  equity value {b['equity']:,.1f} / {SHARES:,} shares  =  ${b['per_share']:,.2f}   terminal share {b['terminal_share']:.1%}")
print(f"  video: $291.75, 79.8% terminal")

print("\n== TABLE 6: edge cases — the model must refuse ==")
for label, kw in [("terminal growth at the discount rate", dict(terminal_growth=0.10))]:
    try: revalue(OPENING, DRIVERS, SHARES, **kw); print(f"  {label}: DID NOT REFUSE — defect")
    except ValueError as e: print(f"  {label}: refused — {e}")

print("\n== TABLE 7: the discount rate, rerun (Video 4's seven rates) ==")
for r in (0.0815, 0.0857, 0.09, 0.0942, 0.0954, 0.10, 0.11):
    print(f"  r {r:6.2%}   ${revalue(OPENING, DRIVERS, SHARES, cost_of_equity=r):7.0f}")
print("  video: 399, 369, 342, 319, 313, 292, 254")

print("\n== TABLE 8: the tornado, one driver at a time, whole model rerun ==")
base = b["per_share"]
shocks = [("gross margin +/-1pt", "gross_margin", DRIVERS["gross_margin"]["value"]-0.01, DRIVERS["gross_margin"]["value"]+0.01),
          ("capex +/-50",         "capex",        DRIVERS["capex"]["value"]+50,          DRIVERS["capex"]["value"]-50),
          ("growth +/-1pt",       "growth",       DRIVERS["growth"]["value"]-0.01,       DRIVERS["growth"]["value"]+0.01),
          ("tax +/-1pt",          "tax",          DRIVERS["tax"]["value"]+0.01,          DRIVERS["tax"]["value"]-0.01),
          ("impairment +/-40",    "impairment",   DRIVERS["impairment"]["value"]+40,     DRIVERS["impairment"]["value"]-40)]
bars = []
for label, key, lo, hi in shocks:
    a, c = revalue(OPENING, DRIVERS, SHARES, **{key: lo}), revalue(OPENING, DRIVERS, SHARES, **{key: hi})
    bars.append((abs(c-a), label, min(a,c), max(a,c)))
for swing, label, lo, hi in sorted(bars, reverse=True):
    print(f"  {label:22s} ${lo:6.0f} .. ${hi:6.0f}   swing ${swing:5.0f}")
print("  video's biggest bar: gross margin, $256 .. $327")

print("\n== TABLE 9: cases — rerun, never added ==")
cases = {"bear  (SG&A/GP stuck 68%)": dict(sga_over_gp=[0.68]*5),
         "bull  (SG&A/GP to 63%)":    dict(sga_over_gp=[0.655,0.645,0.63,0.63,0.63]),
         "base":                      {}}
for label, kw in cases.items():
    print(f"  {label:28s} ${revalue(OPENING, DRIVERS, SHARES, **kw):7.0f}")
print("  video: bear 236, bull 321")
print("\n  The COMPANY layer above is the only part that changes for a second company.")
print("  The METHOD layer — project, checks, dcf, revalue — does not. That is one pipeline, run twice.")
