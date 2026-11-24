"""Retention Desk: call queue, quick score and planning for the DBA3803 churn model.

Run from this folder:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import calllog
from churn_model import CHOICES, SCRIPTS, SEGMENT_ORDER, RetentionModel, clean

HERE = Path(__file__).parent
st.set_page_config(page_title="Retention Desk", page_icon=":material/support_agent:", layout="wide")

NAVY, NAVY_LINE = "#0E1B3D", "#24345F"
BLUE, BLUE_SOFT = "#2563EB", "#93C5FD"
MUTED, ON_NAVY = "#475569", "#A9B8D9"
SEG_COLOR = {"New high-value": "#2563EB", "High-risk service": "#93C5FD", "Other": "#5A6A90", "Stable": "#CBD5E1"}
STATUS_STYLE = {
    "To call": ("#EAF1FE", "#1D4ED8"), "Accepted": ("#DCFCE7", "#166534"),
    "Declined": ("#F1F5F9", "#334155"), "No answer": ("#FFEDD5", "#9A3412"),
}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=JetBrains+Mono:wght@500;700&display=swap');
[data-testid="stMain"] { background-color:#F4F6FB;
  background-image: radial-gradient(#D9E0EC 1px, transparent 1px); background-size: 18px 18px; }
.block-container { padding-top: 2.2rem; max-width: 1400px; }
h1 { font-weight: 800 !important; letter-spacing: -0.02em; }
[class*="st-key-card"] { background:#FFFFFF; border-radius:16px !important; box-shadow:0 1px 2px rgba(15,23,42,.04); }
.num { font-family:'JetBrains Mono', ui-monospace, monospace; }
.disp { font-family:'Bricolage Grotesque', Inter, sans-serif; }
.hero { background:#0E1B3D; color:#FFFFFF; border-radius:16px; padding:22px 26px; display:flex; flex-wrap:wrap;
        align-items:center; gap:28px; margin-bottom: 8px; }
.hero .lab { font-size:13px; color:#A9B8D9; }
.hero .div { width:1px; align-self:stretch; background:#24345F; }
.chip { display:inline-block; font-size:13px; padding:5px 12px; border-radius:999px; border:1px solid #DCE2EA;
        margin:0 6px 6px 0; background:#FFFFFF; }
.sect { font-size:12px; font-weight:600; color:#475569; text-transform:uppercase; letter-spacing:.08em; margin:4px 0 8px; }
.badge { display:inline-flex; align-items:center; justify-content:center; border-radius:10px; font-weight:700; }
.legend { display:flex; flex-wrap:wrap; gap:12px; font-size:12px; color:#A9B8D9; }
.legend i { display:inline-block; width:8px; height:8px; border-radius:4px; margin-right:5px; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- helpers
def persist(key: str) -> dict:
    """Widget kwargs that keep a value in session state across pages (widget keys are dropped off-page)."""
    def sync():
        ss[key] = ss[f"_w_{key}"]
    ss[f"_w_{key}"] = ss[key]
    return {"key": f"_w_{key}", "on_change": sync}


def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def ring(p: float, size: int, stroke: int, track: str, colour: str, sub_colour: str) -> str:
    r = (size - stroke) / 2
    c = 2 * np.pi * r
    h = size / 2
    return (
        f"<div style='position:relative;width:{size}px;height:{size}px;flex-shrink:0'>"
        f"<svg role='img' aria-label='{p:.0%} chance of leaving' width='{size}' height='{size}' viewBox='0 0 {size} {size}'>"
        f"<circle cx='{h}' cy='{h}' r='{r}' fill='none' stroke='{track}' stroke-width='{stroke}'/>"
        f"<circle cx='{h}' cy='{h}' r='{r}' fill='none' stroke='{colour}' stroke-width='{stroke}' stroke-linecap='round'"
        f" stroke-dasharray='{c * p:.1f} {c:.1f}' transform='rotate(-90 {h} {h})'/></svg>"
        f"<div style='position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center'>"
        f"<span class='disp' style='font-size:{size // 4}px;font-weight:800;line-height:1'>{p:.0%}</span>"
        f"<span style='font-size:11px;color:{sub_colour};margin-top:2px'>leave risk</span></div></div>"
    )


def chips(text: str, border: str = "#DCE2EA", bg: str = "#FFFFFF") -> str:
    return "".join(f"<span class='chip' style='border-color:{border};background:{bg}'>{t}</span>"
                   for t in text.split(" · "))


def money_k(x: float) -> str:
    return f"${x / 1000:.1f}k" if x >= 1000 else f"${x:,.0f}"


@st.cache_resource(show_spinner="Fitting the model…")
def load_model():
    return RetentionModel(HERE)


try:
    model = load_model()
except FileNotFoundError:
    st.error("Put WA_Fn-UseC_-Telco-Customer-Churn.csv in this folder or the one above it, then reload.")
    st.stop()

ss = st.session_state
for key, value in {"frame": model.test_frame, "source": "test_group.csv", "quota": 10, "current": None,
                   "table_key": 0, "cost": 20.0, "keep": 20, "months": 1}.items():
    ss.setdefault(key, value)


@st.cache_data(show_spinner=False)
def score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return model.score(frame)


def build_queue():
    scored = score_frame(ss.frame)
    k = max(1, int(round(len(scored) * ss.quota / 100)))
    queue = scored.head(k).copy()
    queue["holdout"] = calllog.holdout_mask(queue)
    calllog.record_assignments(HERE, queue, ss.source)
    status = calllog.latest_status(calllog.load_log(HERE))
    queue["status"] = queue["customerID"].map(status).fillna("To call")
    return scored, queue


def to_call_order(queue: pd.DataFrame) -> pd.DataFrame:
    calls = queue[~queue["holdout"]]
    pending = calls[calls["status"].isin(["To call", "No answer"])].copy()
    pending["retry"] = pending["status"].eq("No answer")
    return pending.sort_values(["retry", "rank"])


# ================================================================ page: call queue
def queue_page():
    scored, queue = build_queue()
    calls = queue[~queue["holdout"]]
    n_hold = int(queue["holdout"].sum())
    pending = to_call_order(queue)
    done = calls[calls["status"].isin(["Accepted", "Declined"])]
    called = int((calls["status"] != "To call").sum())

    head = st.columns([3, 2.2, 1.3, 1.3], vertical_alignment="center")
    with head[0]:
        t1, t2 = st.columns([0.7, 0.3], vertical_alignment="center", gap="small")
        t1.title("Call queue")
        with t2.popover("", icon=":material/info:"):
            st.markdown("Ranked by **chance of leaving × monthly bill**. The model finds customers who look like "
                        "past leavers; it can't tell whether an offer will keep them. That's what the call log and "
                        "the holdout group are for.")
    head[1].markdown(f"<div style='text-align:right;font-size:13px;color:{MUTED}'>{ss.source} · "
                     f"<span class='num'>{len(scored):,}</span> customers</div>", unsafe_allow_html=True)
    with head[2].popover("Upload CSV", icon=":material/upload:", use_container_width=True):
        up = st.file_uploader("Customer file", type="csv", label_visibility="collapsed")
        if up is not None and up.name != ss.source:
            try:
                ss.frame = clean(pd.read_csv(up))
                ss.source, ss.current = up.name, None
                st.rerun()
            except Exception as exc:
                st.error(f"Couldn't read that file. {exc}")
        if st.button("Use sample test group", use_container_width=True):
            ss.frame, ss.source, ss.current = model.test_frame, "test_group.csv", None
            st.rerun()
        st.caption("Telco column format. `Churn` and `customerID` optional.")
    head[3].selectbox("Quota", [5, 10, 15, 20, 25, 30], **persist("quota"), format_func=lambda q: f"Top {q}%",
                      label_visibility="collapsed")

    # ---- summary card
    mix = queue["segment"].value_counts()
    bar = "".join(f"<div style='width:{mix.get(s, 0) / len(queue) * 100:.1f}%;background:{SEG_COLOR[s]}'></div>"
                  for s in SEGMENT_ORDER if mix.get(s, 0))
    legend = "".join(f"<span><i style='background:{SEG_COLOR[s]}'></i>{s} <span class='num'>{mix.get(s, 0)}</span></span>"
                     for s in SEGMENT_ORDER if mix.get(s, 0))
    lost_block = ""
    if "churn" in scored.columns:
        lost = scored.loc[scored["churn"] == 1, "MonthlyCharges"].sum()
        reached = queue.loc[queue["churn"] == 1, "MonthlyCharges"].sum()
        lost_block = (f"<div><div class='lab'>Lost bills reached</div>"
                      f"<div class='num' style='font-size:26px;font-weight:700;color:{BLUE_SOFT}'>{reached / lost:.0%}</div></div>")
    html(
        f"<div class='hero'>"
        f"<div><div class='lab'>Called</div>"
        f"<div class='disp' style='font-size:48px;font-weight:800;line-height:1;letter-spacing:-.03em'>{called}"
        f"<span style='font-size:22px;color:{ON_NAVY}'> / {len(calls)}</span></div>"
        f"<div style='width:150px;height:6px;border-radius:3px;background:{NAVY_LINE};margin-top:8px;overflow:hidden'>"
        f"<div style='width:{called / max(len(calls), 1) * 100:.1f}%;height:100%;background:#3B82F6'></div></div>"
        f"<div class='lab' style='font-size:12px;margin-top:4px'>+ {n_hold} held back as control</div></div>"
        f"<div class='div'></div>"
        f"<div><div class='lab'>Revenue at risk</div><div class='num' style='font-size:26px;font-weight:700'>"
        f"{money_k(queue['revenue_at_risk'].sum())}<span style='font-size:14px;color:{ON_NAVY}'>/mo</span></div></div>"
        f"{lost_block}"
        f"<div style='flex:1 1 240px;display:flex;flex-direction:column;gap:8px'>"
        f"<div class='lab' style='font-size:12px'>Queue mix · top {ss.quota}%</div>"
        f"<div style='display:flex;height:10px;border-radius:5px;overflow:hidden;gap:2px'>{bar}</div>"
        f"<div class='legend'>{legend}</div></div></div>"
    )

    left, right = st.columns([1.55, 1], gap="medium")

    # ---- list
    with left, st.container(border=True, key="card_queue"):
        counts = {"To call": len(pending), "Done": len(done), "All": len(calls)}
        view = st.segmented_control("View", list(counts), default="To call", key="view",
                                    format_func=lambda v: f"{v} {counts[v]}", label_visibility="collapsed") or "To call"
        segs = ["All"] + [s for s in SEGMENT_ORDER if mix.get(s, 0)]
        seg = st.pills("Segment", segs, default="All", key="seg", label_visibility="collapsed") or "All"

        rows = {"To call": pending, "Done": done, "All": calls}[view]
        if seg != "All":
            rows = rows[rows["segment"] == seg]
        table = pd.DataFrame({
            "#": rows["rank"],
            "Customer": rows["customerID"],
            "Leave risk": (rows["p_leave"] * 100).round(),
            "At risk / mo": rows["revenue_at_risk"],
            "Offer": rows["segment"].map(lambda s: SCRIPTS[s]["short"]),
            "Why": rows["drivers"],
            "Status": rows["status"],
        })
        event = st.dataframe(
            table, hide_index=True, use_container_width=True, height=440,
            on_select="rerun", selection_mode="single-row", key=f"table_{ss.table_key}",
            column_config={
                "#": st.column_config.NumberColumn(width="small"),
                "Leave risk": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100, width="small"),
                "At risk / mo": st.column_config.NumberColumn(format="$%.0f", width="small"),
                "Why": st.column_config.TextColumn(width="medium"),
            },
        )
        if event.selection.rows:
            ss.current = table.iloc[event.selection.rows[0]]["Customer"]
            ss.table_key += 1  # clear the selection so the next click always registers
            st.rerun()
        export = queue.assign(offer=queue["segment"].map(lambda s: SCRIPTS[s]["offer"]),
                              group=np.where(queue["holdout"], "holdout", "call"))
        st.download_button("Export list", export.drop(columns=["holdout"]).to_csv(index=False),
                           f"call_queue_top{ss.quota}pct.csv", "text/csv", icon=":material/download:")

    # ---- customer panel
    with right, st.container(border=True, key="card_customer"):
        cur = calls[calls["customerID"] == ss.current]
        if cur.empty:
            ss.current = None
            html(f"<div class='disp' style='font-size:20px;font-weight:600;margin:6px 0 4px'>Ready to call</div>"
                 f"<div style='color:{MUTED};font-size:14px;margin-bottom:14px'>Pick a customer from the list, or start at the top.</div>")
            if not pending.empty and st.button(f"Start with {pending.iloc[0]['customerID']}", type="primary",
                                               use_container_width=True):
                ss.current = pending.iloc[0]["customerID"]
                st.rerun()
            return
        c = cur.iloc[0]
        script = SCRIPTS[c["segment"]]
        top3 = c["rank"] <= 3
        st_bg, st_fg = STATUS_STYLE[c["status"]]
        html(
            f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:16px'>"
            f"<span class='badge num' style='width:32px;height:32px;font-size:14px;background:{BLUE if top3 else '#EEF1F5'};"
            f"color:{'#FFFFFF' if top3 else '#334155'}'>{c['rank']}</span>"
            f"<div style='flex-grow:1'><div class='num' style='font-size:18px;font-weight:700'>{c['customerID']}</div>"
            f"<div style='font-size:12px;color:{MUTED}'>{c['segment']} · <span class='num'>${c['MonthlyCharges']:.2f}</span>/mo bill</div></div>"
            f"<span style='font-size:12px;font-weight:600;padding:4px 10px;border-radius:999px;white-space:nowrap;"
            f"background:{st_bg};color:{st_fg}'>{c['status']}</span></div>"
            f"<div style='display:flex;align-items:center;gap:22px;padding:16px;border-radius:16px;background:#F5F8FE;margin-bottom:18px'>"
            f"{ring(c['p_leave'], 116, 12, '#DCE5F5', BLUE, MUTED)}"
            f"<div><div style='font-size:12px;color:{MUTED}'>At risk</div>"
            f"<div class='num' style='font-size:28px;font-weight:700'>${c['revenue_at_risk']:.0f}"
            f"<span style='font-size:14px;color:{MUTED}'>/mo</span></div></div></div>"
            f"<div class='sect'>Offer</div>"
            f"<div style='background:{NAVY};color:#FFFFFF;border-radius:14px;padding:16px 18px;margin-bottom:18px'>"
            f"<div class='disp' style='font-size:18px;font-weight:600;line-height:1.3'>{script['offer']}</div>"
            f"<div style='font-size:13px;color:{ON_NAVY};margin-top:6px'>{script['avoid']}</div></div>"
            f"<div class='sect'>Why flagged</div><div style='margin-bottom:10px'>{chips(c['drivers'])}</div>"
            f"<div class='sect'>Log call</div>"
        )
        with st.expander("Add note", icon=":material/edit_note:"):
            note = st.text_input("Note", key=f"note_{c['customerID']}", label_visibility="collapsed",
                                 placeholder="e.g. call back after 6pm")
        b = st.columns(3, gap="small")
        nxt = pending[pending["customerID"] != c["customerID"]]
        for col, outcome, kind in zip(b, calllog.OUTCOMES, ["primary", "secondary", "secondary"]):
            if col.button(outcome, key=f"log_{outcome}", type=kind, use_container_width=True):
                calllog.append_log(HERE, {"customerID": c["customerID"], "rank": int(c["rank"]),
                                          "segment": c["segment"], "offer": script["short"],
                                          "outcome": outcome, "note": note, "source": ss.source})
                ss.current = nxt.iloc[0]["customerID"] if not nxt.empty else None
                ss.table_key += 1
                st.toast(f"{c['customerID']}: {outcome}")
                st.rerun()
        n1, n2 = st.columns([1, 1.3], vertical_alignment="center")
        n1.caption("Saves to call_log.csv")
        if not nxt.empty and n2.button(f"Skip to {nxt.iloc[0]['customerID']}", type="tertiary",
                                       icon=":material/arrow_forward:", use_container_width=True):
            ss.current = nxt.iloc[0]["customerID"]
            st.rerun()


# ================================================================ page: quick score
def quick_page():
    st.title("Quick score")
    left, right = st.columns([1.15, 1], gap="medium")
    with left, st.container(border=True, key="card_form"):
        a, b = st.columns(2)
        tenure = a.number_input("Months with company", 0, 120, 4)
        bill = b.number_input("Monthly bill ($)", 0.0, 200.0, 89.0, 1.0)
        contract = st.segmented_control("Contract", ["Monthly", "1 year", "2 years"], default="Monthly",
                                        width="stretch") or "Monthly"
        internet = st.segmented_control("Internet", ["Fibre", "DSL", "None"], default="Fibre",
                                        width="stretch") or "Fibre"
        services = st.pills("Has these", ["Online security", "Tech support", "Online backup", "Device protection",
                                          "Streaming TV", "Streaming movies", "Phone", "Multiple lines"],
                            selection_mode="multi", default=["Streaming TV", "Phone"])
        with st.expander("More: payment, billing, household"):
            payment = st.selectbox("Payment method", CHOICES["PaymentMethod"])
            m1, m2, m3 = st.columns(3)
            paperless = m1.toggle("Paperless", True)
            partner = m2.toggle("Partner")
            dependents = m3.toggle("Dependents")

    def has(name):
        return "Yes" if name in services else "No"

    net = internet != "None"
    one = pd.DataFrame([{
        "customerID": "INPUT", "tenure": tenure, "MonthlyCharges": bill,
        "Contract": {"Monthly": "Month-to-month", "1 year": "One year", "2 years": "Two year"}[contract],
        "InternetService": {"Fibre": "Fiber optic", "DSL": "DSL", "None": "No"}[internet],
        "PaymentMethod": payment, "PaperlessBilling": "Yes" if paperless else "No",
        **{col: (has(label) if net else "No") for col, label in [
            ("OnlineSecurity", "Online security"), ("TechSupport", "Tech support"), ("OnlineBackup", "Online backup"),
            ("DeviceProtection", "Device protection"), ("StreamingTV", "Streaming TV"),
            ("StreamingMovies", "Streaming movies")]},
        "PhoneService": has("Phone"), "MultipleLines": has("Multiple lines") if "Phone" in services else "No",
        "Partner": "Yes" if partner else "No", "Dependents": "Yes" if dependents else "No",
    }])
    r = model.score(clean(one)).iloc[0]
    on_list = r["revenue_at_risk"] >= model.cutoff_10
    top = max(1 - r["train_percentile"], 0.01)
    script = SCRIPTS[r["segment"]]
    pill = (f"<span style='font-size:12px;font-weight:700;padding:5px 12px;border-radius:999px;"
            f"background:{'#3B82F6' if on_list else NAVY_LINE};color:#FFFFFF'>"
            f"{'On call list' if on_list else 'Not a call target'}</span>")
    with right:
        html(
            f"<div style='background:{NAVY};color:#FFFFFF;border-radius:16px;padding:24px;display:flex;flex-direction:column;gap:20px'>"
            f"<div style='display:flex;align-items:center'><span style='font-size:12px;color:{ON_NAVY};flex-grow:1'>Live result · 10% quota</span>{pill}</div>"
            f"<div style='display:flex;align-items:center;gap:24px'>"
            f"{ring(r['p_leave'], 140, 14, NAVY_LINE, '#3B82F6', ON_NAVY)}"
            f"<div style='display:flex;flex-direction:column;gap:12px'>"
            f"<div><div style='font-size:12px;color:{ON_NAVY}'>At risk</div><div class='num' style='font-size:24px;font-weight:700'>"
            f"${r['revenue_at_risk']:.0f}<span style='font-size:13px;color:{ON_NAVY}'>/mo</span></div></div>"
            f"<div><div style='font-size:12px;color:{ON_NAVY}'>Ranks in</div><div class='num' style='font-size:24px;font-weight:700;color:{BLUE_SOFT}'>"
            f"top {top:.0%}</div></div></div></div>"
            f"<div style='background:#FFFFFF;color:#0F172A;border-radius:12px;padding:14px 16px'>"
            f"<div style='font-size:12px;color:{MUTED}'>Offer · {r['segment']}</div>"
            f"<div class='disp' style='font-size:16px;font-weight:600;margin-top:4px'>{script['offer']}</div></div>"
            f"<div>{chips(r['drivers'], border='#3B4C7A', bg='transparent')}</div></div>"
        )


# ================================================================ page: planning
def planning_page():
    scored, queue = build_queue()
    st.title("Planning")

    top_l, top_r = st.columns([1, 1], gap="medium")
    with top_l, st.container(border=True, key="card_inputs"):
        a, b = st.columns(2)
        a.selectbox("Quota", [5, 10, 15, 20, 25, 30], **persist("quota"), format_func=lambda q: f"Top {q}%")
        b.number_input("Cost per call ($)", 0.0, 500.0, step=5.0, **persist("cost"))
        a.number_input("Leavers kept (%)", 0, 100, step=5, **persist("keep"))
        b.number_input("Months they stay", 1, 36, **persist("months"))

    def plan(q):
        k = max(1, int(round(len(scored) * q / 100)))
        top = scored.head(k).copy()
        top = top[~calllog.holdout_mask(top)]
        expected = (top["p_leave"] * top["MonthlyCharges"]).sum()
        cost = len(top) * ss.cost
        return ss.keep / 100 * expected * ss.months - cost, cost, expected, len(top)

    net, cost, expected, n_call = plan(ss.quota)
    breakeven = cost / (expected * ss.months) if expected else np.nan
    with top_r:
        colour = "#FDBA74" if net < 0 else "#86EFAC"
        html(
            f"<div style='background:{NAVY};color:#FFFFFF;border-radius:16px;padding:22px 24px;min-height:186px;"
            f"box-sizing:border-box;display:flex;flex-direction:column;justify-content:space-between;gap:16px'>"
            f"<div><div style='font-size:13px;color:{ON_NAVY}'>Net this cycle · {n_call} calls</div>"
            f"<div class='disp' style='font-size:52px;font-weight:800;line-height:1.05;letter-spacing:-.03em;color:{colour}'>"
            f"{'−' if net < 0 else ''}${abs(net):,.0f}</div></div>"
            f"<div style='display:flex;gap:28px'>"
            f"<div><div style='font-size:12px;color:{ON_NAVY}'>Programme cost</div><div class='num' style='font-size:20px;font-weight:700'>${cost:,.0f}</div></div>"
            f"<div><div style='font-size:12px;color:{ON_NAVY}'>Break-even keep rate</div><div class='num' style='font-size:20px;font-weight:700;color:{BLUE_SOFT}'>{breakeven:.0%}</div></div>"
            f"</div></div>"
        )

    c_l, c_r = st.columns([2, 1], gap="medium")
    with c_l, st.container(border=True, key="card_chart"):
        st.markdown("**Net value by quota**")
        qs = np.arange(5, 31)
        nets = [plan(q)[0] for q in qs]
        fig = go.Figure(go.Scatter(x=qs, y=nets, mode="lines", line=dict(color=BLUE, width=3), fill="tozeroy",
                                   fillcolor="rgba(37,99,235,0.10)",
                                   hovertemplate="Top %{x}%<br>net $%{y:,.0f}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color="#94A3B8", width=1, dash="dot"))
        fig.add_trace(go.Scatter(x=[ss.quota], y=[nets[ss.quota - 5]], mode="markers+text", text=[f"{ss.quota}%"],
                                 textposition="top right", textfont=dict(color="#C2410C", size=12),
                                 marker=dict(size=12, color="#FFFFFF", line=dict(color="#C2410C", width=3)),
                                 hoverinfo="skip"))
        fig.update_layout(height=230, margin=dict(l=0, r=10, t=10, b=0), showlegend=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Inter, sans-serif", color=MUTED, size=11))
        fig.update_xaxes(ticksuffix="%", gridcolor="#EEF1F5", zeroline=False)
        fig.update_yaxes(tickformat="$,.0f", gridcolor="#EEF1F5", zeroline=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c_r, st.container(border=True, key="card_mix"):
        st.markdown("**Who's in the queue**")
        mix = queue["segment"].value_counts()
        segs = [s for s in SEGMENT_ORDER if mix.get(s, 0)]
        bar = "".join(f"<div style='width:{mix[s] / len(queue) * 100:.1f}%;background:{SEG_COLOR[s]}'></div>" for s in segs)
        rows = "".join(f"<div style='display:flex;align-items:center;gap:8px;font-size:13px;margin-top:10px'>"
                       f"<span style='width:10px;height:10px;border-radius:5px;background:{SEG_COLOR[s]}'></span>"
                       f"<span style='flex-grow:1'>{s}</span><span class='num' style='font-weight:700'>{mix[s]}</span></div>"
                       for s in segs)
        html(f"<div style='display:flex;height:14px;border-radius:7px;overflow:hidden;gap:2px;margin-top:6px'>{bar}</div>{rows}")

    # ---- outcomes
    with st.container(border=True, key="card_outcomes"):
        log = calllog.load_log(HERE)
        html(f"<b>Call outcomes</b> <span style='color:{MUTED};font-size:13px'>· "
             f"<span class='num'>{len(log)}</span> calls logged</span>")
        if log.empty:
            st.caption("No calls logged yet. Results appear here as agents log calls in the queue.")
        else:
            latest = log.sort_values("timestamp").groupby("customerID").last().reset_index()
            summary = latest.groupby("segment").agg(
                Customers=("customerID", "size"),
                Reached=("outcome", lambda o: o.isin(["Accepted", "Declined"]).sum()),
                Accepted=("outcome", lambda o: (o == "Accepted").sum()),
            )
            summary = summary.reindex([s for s in SEGMENT_ORDER if s in summary.index])
            summary["Accept rate"] = (summary["Accepted"] / summary["Reached"].replace(0, np.nan) * 100).round()
            summary.insert(0, "Offer", [SCRIPTS[s]["short"] for s in summary.index])
            summary.index.name = "Segment"
            st.dataframe(summary, use_container_width=True, column_config={
                "Accept rate": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100)})
            reached = summary["Reached"].sum()
            if reached:
                st.caption(f"Overall accept rate {summary['Accepted'].sum() / reached:.0%}. Accepting isn't staying: "
                           "treat it as an upper bound on the keep rate until the next extract shows who left.")
        d1, d2, _ = st.columns([1, 1, 2])
        for col, name in [(d1, "call_log.csv"), (d2, "queue_assignments.csv")]:
            p = HERE / name
            col.download_button(name, p.read_bytes() if p.exists() else b"", name, "text/csv",
                                disabled=not p.exists(), icon=":material/download:", use_container_width=True)
        st.caption(f"Holdout: {int(queue['holdout'].sum())} of {len(queue)} customers in this queue are not called. "
                   "Compare who left in each group at the next extract to measure the effect of calling.")

    with st.expander("About the model"):
        st.markdown(f"""
- **Model:** logistic regression (C = 2.0) fit on the notebook's 80% training sample ({model.n_train:,} customers,
  seed 3803). Random forest, XGBoost and a neural net didn't beat it by the 2-point margin.
- **Score:** chance of leaving × monthly bill. `gender` and `SeniorCitizen` aren't used (no loss in coverage).
- **Segments, in priority order:** Stable (1–2 yr contract, 24+ mo) → New high-value (monthly, ≤ 12 mo,
  bill ≥ $70) → High-risk service (fibre, no security, no tech support) → Other.
- **Limit:** it ranks customers who look like past leavers. Only the call log plus the holdout can show whether an offer works.
""")


# ================================================================ navigation
st.logo(str(HERE / "logo.svg"), size="large")
_, _queue = build_queue()
pg = st.navigation([
    st.Page(queue_page, title=f"Call queue · {len(to_call_order(_queue))}", icon=":material/format_list_bulleted:",
            url_path="queue", default=True),
    st.Page(quick_page, title="Quick score", icon=":material/person_search:", url_path="score"),
    st.Page(planning_page, title="Planning", icon=":material/bar_chart:", url_path="planning"),
])
pg.run()
