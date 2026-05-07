import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind, chi2_contingency, norm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="DataDoctors", page_icon="👁️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'Outfit',sans-serif;}
[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse at 20% 50%,#0f1729 0%,#060b17 40%,#0a0f1e 100%);min-height:100vh;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#060c1a 0%,#0a1228 50%,#060c1a 100%);border-right:1px solid rgba(56,189,248,0.15);}
[data-testid="stAppViewContainer"]::before{content:'';position:fixed;top:-200px;left:-200px;width:600px;height:600px;background:radial-gradient(circle,rgba(56,189,248,0.06) 0%,transparent 70%);border-radius:50%;animation:orbFloat 8s ease-in-out infinite;pointer-events:none;z-index:0;}
[data-testid="stAppViewContainer"]::after{content:'';position:fixed;bottom:-200px;right:-200px;width:700px;height:700px;background:radial-gradient(circle,rgba(129,140,248,0.05) 0%,transparent 70%);border-radius:50%;animation:orbFloat 10s ease-in-out infinite reverse;pointer-events:none;z-index:0;}
@keyframes orbFloat{0%,100%{transform:translate(0,0) scale(1);}33%{transform:translate(30px,-30px) scale(1.05);}66%{transform:translate(-20px,20px) scale(0.97);}}
.hero-wrap{text-align:center;padding:48px 20px 32px;position:relative;}
.hero-eyeball{font-size:4rem;animation:pulse 2.5s ease-in-out infinite;display:block;margin-bottom:12px;}
@keyframes pulse{0%,100%{transform:scale(1);filter:drop-shadow(0 0 8px rgba(56,189,248,0.5));}50%{transform:scale(1.12);filter:drop-shadow(0 0 24px rgba(56,189,248,0.9));}}
.hero-title{font-family:'Outfit',sans-serif;font-size:clamp(2.8rem,6vw,4.5rem);font-weight:900;letter-spacing:-0.03em;background:linear-gradient(135deg,#38bdf8 0%,#818cf8 40%,#34d399 80%,#38bdf8 100%);background-size:300% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:gradShift 4s linear infinite;line-height:1;}
@keyframes gradShift{to{background-position:300% center;}}
.hero-badge{display:inline-block;margin-top:14px;background:rgba(56,189,248,0.1);border:1px solid rgba(56,189,248,0.3);border-radius:999px;padding:6px 20px;font-size:0.82rem;font-weight:500;color:#7dd3fc;letter-spacing:0.12em;text-transform:uppercase;}
.gcard{background:linear-gradient(135deg,rgba(255,255,255,0.04) 0%,rgba(255,255,255,0.015) 100%);border:1px solid rgba(56,189,248,0.18);border-radius:20px;padding:28px 20px;text-align:center;position:relative;overflow:hidden;transition:all 0.35s cubic-bezier(.25,.8,.25,1);}
.gcard:hover{transform:translateY(-6px);border-color:rgba(56,189,248,0.5);box-shadow:0 16px 48px rgba(56,189,248,0.12);}
.gcard-val{font-family:'Outfit',sans-serif;font-size:2.4rem;font-weight:800;background:linear-gradient(135deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1;}
.gcard-label{font-size:0.75rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.12em;margin-top:8px;}
.gcard-sub{font-size:0.8rem;color:#7dd3fc;margin-top:4px;}
.sec-hdr{display:flex;align-items:center;gap:12px;margin:28px 0 16px;}
.sec-hdr-line{flex:1;height:1px;background:linear-gradient(90deg,rgba(56,189,248,0.5),transparent);}
.sec-hdr-text{font-family:'Outfit',sans-serif;font-size:1.4rem;font-weight:800;color:#e2e8f0;white-space:nowrap;}
.ibox{background:rgba(56,189,248,0.06);border-left:3px solid #38bdf8;border-radius:0 12px 12px 0;padding:14px 18px;color:#cbd5e1;font-size:0.9rem;line-height:1.7;margin:10px 0;}
.pred-box{border-radius:24px;padding:40px 32px;text-align:center;animation:fadeUp 0.5s ease forwards;}
@keyframes fadeUp{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}
.pred-box-dr{background:linear-gradient(135deg,rgba(239,68,68,0.18),rgba(239,68,68,0.05));border:2px solid rgba(239,68,68,0.5);}
.pred-box-ok{background:linear-gradient(135deg,rgba(52,211,153,0.18),rgba(52,211,153,0.05));border:2px solid rgba(52,211,153,0.5);}
.pred-icon{font-size:3.5rem;animation:pulse 2s infinite;}
.pred-title{font-family:'Outfit',sans-serif;font-size:2rem;font-weight:900;margin:10px 0 6px;}
.pred-prob{font-size:1.2rem;font-weight:600;opacity:0.85;}
.pred-sub{font-size:0.85rem;opacity:0.6;margin-top:8px;}
.badge{display:inline-block;padding:6px 18px;border-radius:999px;font-size:0.82rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-top:10px;}
.badge-low{background:rgba(52,211,153,0.15);border:1px solid #34d399;color:#34d399;}
.badge-med{background:rgba(251,191,36,0.15);border:1px solid #fbbf24;color:#fbbf24;}
.badge-high{background:rgba(239,68,68,0.15);border:1px solid #ef4444;color:#ef4444;}
.sb-logo{text-align:center;padding:24px 0 8px;font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:900;background:linear-gradient(135deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sb-sub{text-align:center;font-size:0.7rem;color:#334155;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:18px;}
.sb-stat{background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.12);border-radius:12px;padding:10px 14px;margin:6px 0;font-size:0.82rem;color:#94a3b8;}
.sb-stat b{color:#7dd3fc;}
h1,h2,h3,h4,h5{color:#e2e8f0 !important;font-family:'Outfit',sans-serif !important;}
p,li{color:#94a3b8;}
[data-testid="stMetricValue"]{color:#38bdf8 !important;font-family:'Outfit',sans-serif !important;}
[data-testid="stMetricLabel"]{color:#64748b !important;}
[data-testid="stMetric"]{background:rgba(56,189,248,0.05) !important;border:1px solid rgba(56,189,248,0.15) !important;border-radius:14px !important;padding:14px !important;}
div[data-testid="stTabs"] button{color:#64748b !important;font-family:'Outfit',sans-serif !important;font-weight:600 !important;}
div[data-testid="stTabs"] button[aria-selected="true"]{color:#38bdf8 !important;border-bottom:2px solid #38bdf8 !important;}
.stSlider label,.stSelectbox label,.stNumberInput label{color:#94a3b8 !important;}
.stButton button{background:linear-gradient(135deg,#38bdf8,#818cf8) !important;color:#000 !important;border:none !important;border-radius:12px !important;font-family:'Outfit',sans-serif !important;font-weight:700 !important;font-size:1rem !important;padding:14px !important;letter-spacing:0.05em !important;transition:all 0.3s !important;}
.stButton button p, .stButton button span{color:#000 !important;}
.stButton button:hover{opacity:0.88 !important;transform:translateY(-2px) !important;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv('Retinopathy_Debrecen.csv')

@st.cache_resource
def train_models(df):
    X = df.drop('class', axis=1)
    y = df['class']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc  = StandardScaler()
    Xtr_sc = sc.fit_transform(Xtr)
    Xte_sc = sc.transform(Xte)
    lr = LogisticRegression(max_iter=5000, random_state=42).fit(Xtr_sc, ytr)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr_sc, ytr)
    return lr, rf, sc, Xte_sc, yte, X.columns.tolist()

df = load_data()
lr, rf, sc, Xte, yte, feat_cols = train_models(df)

num_cols = df.select_dtypes(include=np.number).columns.drop('class').tolist()
cat_cols = [c for c in df.columns if c != 'class' and df[c].nunique() <= 5]
dr    = df[df['class'] == 1]
no_dr = df[df['class'] == 0]
N     = len(df)
N_dr  = len(dr)
N_ndr = len(no_dr)
PCT   = N_dr / N * 100

BG  = 'rgba(0,0,0,0)'
GRD = 'rgba(56,189,248,0.08)'
TXT = '#e2e8f0'
C1,C2,C3,C4 = '#38bdf8','#34d399','#818cf8','#fb923c'

def themed(fig, h=400, title='', xt='', yt=''):
    fig.update_layout(height=h, paper_bgcolor=BG, plot_bgcolor=BG,
                      font=dict(color=TXT,family='Outfit'),
                      title=dict(text=title,font=dict(size=15,color=TXT)),
                      xaxis=dict(gridcolor=GRD,linecolor=GRD,title=xt,tickfont=dict(color='#64748b')),
                      yaxis=dict(gridcolor=GRD,linecolor=GRD,title=yt,tickfont=dict(color='#64748b')),
                      legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#94a3b8')),
                      margin=dict(l=14,r=14,t=44,b=14))
    return fig

def sec(label):
    st.markdown(f'<div class="sec-hdr"><span class="sec-hdr-text">{label}</span><div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)

def ibox(html):
    st.markdown(f'<div class="ibox">{html}</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sb-logo">👁️ DataDoctors</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Retinopathy AI · DataDoctors</div>', unsafe_allow_html=True)
    page = st.selectbox("", ["🏠  Home","📊  Visualization","📐  Statistical Analysis",
                              "🎲  Probability","🧪  Hypothesis Testing","🔍  Outlier Detection",
                              "🤖  Live Prediction","📈  Model Evaluation"],
                        label_visibility='collapsed')
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="sb-stat">🧬 <b>{N}</b> total patients</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-stat">📋 <b>{len(feat_cols)}</b> features</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-stat">🔴 <b>{N_dr}</b> DR positive ({PCT:.1f}%)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-stat">🟢 <b>{N_ndr}</b> DR negative</div>', unsafe_allow_html=True)

# ── HOME ────────────────────────────────────────────────────────────────────────
if "Home" in page:
    st.markdown(f'<div class="hero-wrap"><span class="hero-eyeball">👁️</span><div class="hero-title">DataDoctors</div><div class="hero-badge">Diabetic Retinopathy · AI Prediction · DataDoctors · </div></div>', unsafe_allow_html=True)

    rf_p_home  = rf.predict_proba(Xte)[:,1]
    auc_home   = roc_auc_score(yte, rf_p_home)
    rec_home   = recall_score(yte, (rf_p_home>=0.3).astype(int))

    cards = [(str(N),"Total Patients","UCI Debrecen Dataset"),
             (str(len(feat_cols)),"Features","Retinal measurements"),
             (f"{PCT:.1f}%","DR Positive",f"{N_dr} patients"),
             (f"{auc_home:.3f}","Best AUC","Random Forest"),
             (f"{rec_home*100:.1f}%","Recall @ 0.3","Medical threshold")]
    cols = st.columns(5)
    for col,(v,l,s) in zip(cols,cards):
        with col:
            st.markdown(f'<div class="gcard"><div class="gcard-val">{v}</div><div class="gcard-label">{l}</div><div class="gcard-sub">{s}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns([1.3,1])
    with cl:
        sec("About DiabEye")
        ibox(f"<b>DiabEye</b> uses AI to detect <b>Diabetic Retinopathy (DR)</b> — a leading cause of blindness in diabetic patients.<br><br>Trained on <b>{N} patients</b> with <b>{len(feat_cols)} retinal features</b> including microaneurysms, exudates and optic disc measurements.<br><br><b>Models:</b> Logistic Regression &amp; Random Forest<br><b>Threshold:</b> 0.3 — maximizes recall (WHO guideline aligned)")
        sec("Feature Overview")
        col_info = pd.DataFrame({'Feature': feat_cols,
                                  'Min':  [round(df[c].min(),4) for c in feat_cols],
                                  'Max':  [round(df[c].max(),4) for c in feat_cols],
                                  'Mean': [round(df[c].mean(),4) for c in feat_cols],
                                  'Std':  [round(df[c].std(),4)  for c in feat_cols]})
        st.dataframe(col_info, use_container_width=True, hide_index=True)
    with cr:
        counts = df['class'].value_counts().sort_index()
        fig = go.Figure(go.Pie(labels=['No DR','Has DR'], values=counts.values, hole=0.6,
                               pull=[0.03,0.06], marker=dict(colors=[C2,'#ef4444'],
                               line=dict(color='rgba(0,0,0,0)',width=0))))
        fig = themed(fig, h=300, title="Dataset Distribution")
        fig.update_layout(annotations=[dict(text=f'{PCT:.1f}%<br>DR',x=0.5,y=0.5,font_size=18,showarrow=False,font_color=TXT)])
        st.plotly_chart(fig, use_container_width=True)
        sec("Sample Data")
        st.dataframe(df.sample(6,random_state=42), use_container_width=True, height=240)

# ── VISUALIZATION ────────────────────────────────────────────────────────────────
elif "Visualization" in page:
    sec("📊 Data Visualization")
    t1,t2,t3,t4,t5 = st.tabs(["Distribution","Histograms","Scatter Plots","Correlation","Boxplots"])

    with t1:
        counts = df['class'].value_counts().sort_index()
        friendly = ['No DR','Has DR']
        c1,c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(x=friendly, y=counts.values, text=counts.values,
                                   textposition='outside', textfont=dict(color=TXT,size=14),
                                   marker=dict(color=[C2,'#ef4444'])))
            fig = themed(fig, title='Class Distribution', xt='', yt='Count')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            pct_val = counts.values[1]/counts.values.sum()*100
            fig = go.Figure(go.Pie(labels=friendly, values=counts.values, hole=0.55,
                                   pull=[0.03,0.06], marker=dict(colors=[C2,'#ef4444'],
                                   line=dict(color='rgba(0,0,0,0)',width=0))))
            fig = themed(fig, title='Percentage Split')
            fig.update_layout(annotations=[dict(text=f'{pct_val:.1f}%<br>DR',x=0.5,y=0.5,font_size=18,showarrow=False,font_color=TXT)])
            st.plotly_chart(fig, use_container_width=True)
        freq_df = pd.DataFrame({'Class':friendly,'Count':counts.values,
                                 'Percentage':[f"{v/N*100:.2f}%" for v in counts.values]})
        st.dataframe(freq_df, use_container_width=True, hide_index=True)

    with t2:
        col_h = st.selectbox("Select Feature", num_cols, key='h_sel')
        bins  = st.slider("Bins", 10, 80, 30, key='h_bins')
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Overall","DR vs No DR"])
        fig.add_trace(go.Histogram(x=df[col_h], nbinsx=bins, marker_color=C1, opacity=0.8, name='All'), row=1,col=1)
        fig.add_trace(go.Histogram(x=dr[col_h], nbinsx=bins, marker_color='#ef4444', opacity=0.7, name='Has DR'), row=1,col=2)
        fig.add_trace(go.Histogram(x=no_dr[col_h], nbinsx=bins, marker_color=C2, opacity=0.7, name='No DR'), row=1,col=2)
        fig = themed(fig, h=400, title=f'Distribution of {col_h}')
        fig.update_layout(barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Mean",   f"{df[col_h].mean():.3f}")
        m2.metric("Median", f"{df[col_h].median():.3f}")
        m3.metric("Std",    f"{df[col_h].std():.3f}")
        m4.metric("Min",    f"{df[col_h].min():.3f}")
        m5.metric("Max",    f"{df[col_h].max():.3f}")

    with t3:
        c1,c2 = st.columns(2)
        x_col = c1.selectbox("X Axis", num_cols, index=0, key='sc_x')
        y_col = c2.selectbox("Y Axis", num_cols, index=min(6,len(num_cols)-1), key='sc_y')
        show_trend = st.checkbox("Show Trendline", value=True)
        color_map = df['class'].map({0:'No DR',1:'Has DR'})
        fig = px.scatter(df, x=x_col, y=y_col, color=color_map,
                         color_discrete_map={'No DR':C2,'Has DR':'#ef4444'},
                         opacity=0.55, trendline='ols' if show_trend else None)
        fig = themed(fig, h=450, title=f'Scatter — {x_col} vs {y_col}', xt=x_col, yt=y_col)
        fig.update_layout(legend_title='')
        st.plotly_chart(fig, use_container_width=True)
        corr_val = df[[x_col,y_col]].corr().iloc[0,1]
        lin = ("Strong Positive 📈" if corr_val>0.7 else "Strong Negative 📉" if corr_val<-0.7 else
               "Moderate" if abs(corr_val)>0.4 else "Non-Linear / Weak")
        ibox(f"<b>Pearson Correlation:</b> {corr_val:.4f} → <b>{lin}</b>")

    with t4:
        sel_h = st.multiselect("Columns", num_cols+['class'], default=num_cols+['class'])
        if sel_h:
            fig = px.imshow(df[sel_h].corr().round(2), text_auto=True,
                            color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect='auto')
            fig = themed(fig, h=520, title='Correlation Heatmap')
            st.plotly_chart(fig, use_container_width=True)

    with t5:
        col_b = st.selectbox("Select Feature", num_cols, key='b_sel')
        fig = go.Figure()
        fig.add_trace(go.Box(y=dr[col_b],    name='Has DR', marker_color='#ef4444', boxmean='sd'))
        fig.add_trace(go.Box(y=no_dr[col_b], name='No DR',  marker_color=C2,       boxmean='sd'))
        fig = themed(fig, h=440, title=f'Boxplot — {col_b}', yt=col_b)
        st.plotly_chart(fig, use_container_width=True)

# ── STATISTICAL ANALYSIS ────────────────────────────────────────────────────────
elif "Statistical" in page:
    sec("📐 Statistical Analysis")
    t1,t2,t3 = st.tabs(["Descriptive Stats","Group Comparison","Confidence Intervals"])

    with t1:
        desc = df[num_cols].describe().round(4)
        st.dataframe(desc.style.background_gradient(cmap='Blues', axis=1), use_container_width=True)

    with t2:
        sel = st.multiselect("Select Features", num_cols, default=num_cols[:6])
        if sel:
            comp = pd.DataFrame({'Feature':sel,
                                  'DR Mean':   [round(dr[c].mean(),4)    for c in sel],
                                  'No DR Mean':[round(no_dr[c].mean(),4) for c in sel],
                                  'Difference':[round(dr[c].mean()-no_dr[c].mean(),4) for c in sel]})
            st.dataframe(comp, use_container_width=True, hide_index=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Has DR', x=comp['Feature'], y=comp['DR Mean'],
                                 marker_color='#ef4444', text=comp['DR Mean'].round(2), textposition='outside'))
            fig.add_trace(go.Bar(name='No DR',  x=comp['Feature'], y=comp['No DR Mean'],
                                 marker_color=C2, text=comp['No DR Mean'].round(2), textposition='outside'))
            fig = themed(fig, h=380, title='Mean Comparison: DR vs No DR', yt='Mean Value')
            fig.update_layout(barmode='group')
            st.plotly_chart(fig, use_container_width=True)

    with t3:
        ci_sel = st.multiselect("Select Features", num_cols, default=num_cols[:6], key='ci_sel')
        if ci_sel:
            def ci95(d):
                m=d.mean(); mg=1.96*d.std(ddof=1)/np.sqrt(len(d))
                return m, m-mg, m+mg, mg
            rows=[]
            for c in ci_sel:
                m,lo,hi,mg = ci95(df[c])
                rows.append({'Feature':c,'Mean':round(m,4),'Lower 95%CI':round(lo,4),'Upper 95%CI':round(hi,4),'±Margin':round(mg,4)})
            ci_df = pd.DataFrame(rows)
            st.dataframe(ci_df, use_container_width=True, hide_index=True)
            fig = go.Figure()
            for _,r in ci_df.iterrows():
                fig.add_trace(go.Scatter(x=[r['Feature']], y=[r['Mean']],
                                         error_y=dict(type='data',array=[r['±Margin']],color=C1,thickness=2.5,width=8),
                                         mode='markers', marker=dict(color=C1,size=14,symbol='diamond'),
                                         name=r['Feature'], showlegend=False))
            fig = themed(fig, h=360, title='95% Confidence Intervals', yt='Value')
            st.plotly_chart(fig, use_container_width=True)

# ── PROBABILITY ──────────────────────────────────────────────────────────────────
elif "Probability" in page:
    sec("🎲 Probability & Distribution")
    p_dr  = N_dr  / N
    p_ndr = N_ndr / N
    cond_feat = num_cols[0]
    q75 = df[cond_feat].quantile(0.75)
    sub  = df[df[cond_feat] > q75]
    p_cond = len(sub[sub['class']==1]) / max(len(sub),1)

    m1,m2,m3 = st.columns(3)
    m1.metric("P(Diabetic Retinopathy)",    f"{p_dr*100:.2f}%",   f"{N_dr} patients")
    m2.metric("P(No Diabetic Retinopathy)", f"{p_ndr*100:.2f}%",  f"{N_ndr} patients")
    m3.metric(f"P(DR | {cond_feat} > Q75)", f"{p_cond*100:.2f}%", "Conditional")

    st.markdown("<br>", unsafe_allow_html=True)
    feat_d = st.selectbox("Select Feature for Distribution", num_cols)
    mu_d = df[feat_d].mean(); sig_d = df[feat_d].std()
    x_d = np.linspace(df[feat_d].min(), df[feat_d].max(), 300)

    cl, cr = st.columns([2,1])
    with cl:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[feat_d], histnorm='probability density',
                                   marker_color=C1, opacity=0.65, name='Actual Data', nbinsx=35))
        fig.add_trace(go.Scatter(x=x_d, y=norm.pdf(x_d,mu_d,sig_d),
                                 line=dict(color='#ef4444',width=3), name='Normal Curve'))
        fig.add_vline(x=mu_d, line_dash='dash', line_color=C3,
                      annotation_text=f'Mean={mu_d:.2f}', annotation_font_color=C3)
        fig = themed(fig, h=360, title=f'Distribution — {feat_d}', xt=feat_d, yt='Density')
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        sk = df[feat_d].skew(); ku = df[feat_d].kurt()
        dt = 'Normal' if abs(sk)<0.5 else ('Right Skewed' if sk>0 else 'Left Skewed')
        ibox(f"<b>{feat_d}</b><br><br><b>Mean:</b> {mu_d:.4f}<br><b>Std:</b> {sig_d:.4f}<br><b>Skewness:</b> {sk:.4f}<br><b>Kurtosis:</b> {ku:.4f}<br><b>Type:</b> {dt}")

    sec("Skewness Summary — All Features")
    sk_df = pd.DataFrame({'Feature':num_cols,'Skewness':[round(df[c].skew(),4) for c in num_cols],
                           'Type':['Normal' if abs(df[c].skew())<0.5 else ('Right Skewed' if df[c].skew()>0 else 'Left Skewed') for c in num_cols]})
    st.dataframe(sk_df, use_container_width=True, hide_index=True)
    fig = go.Figure(go.Bar(x=sk_df['Feature'], y=sk_df['Skewness'],
                           marker=dict(color=sk_df['Skewness'], colorscale='RdBu_r'),
                           text=sk_df['Skewness'].round(3), textposition='outside'))
    fig.add_hline(y=0.5,  line_dash='dash', line_color=C4, annotation_text='+0.5')
    fig.add_hline(y=-0.5, line_dash='dash', line_color=C4, annotation_text='-0.5')
    fig = themed(fig, h=320, title='Skewness — All Features', yt='Skewness')
    st.plotly_chart(fig, use_container_width=True)

# ── HYPOTHESIS TESTING ───────────────────────────────────────────────────────────
elif "Hypothesis" in page:
    sec("🧪 Hypothesis Testing")
    ibox("<b>H₀:</b> No significant difference between groups &nbsp;|&nbsp; <b>H₁:</b> Significant difference (p &lt; α)")
    alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
    t_sel = st.multiselect("Select Features for T-Test", num_cols, default=num_cols[:7])
    if t_sel:
        rows=[]
        for c in t_sel:
            t,p = ttest_ind(dr[c], no_dr[c])
            rows.append({'Feature':c,'DR Mean':round(dr[c].mean(),4),'No DR Mean':round(no_dr[c].mean(),4),
                         'p-value':round(p,6),'Result':'✅ Reject H₀' if p<alpha else '❌ Fail to Reject'})
        tt_df = pd.DataFrame(rows)
        st.dataframe(tt_df, use_container_width=True, hide_index=True)
        sig = sum(1 for r in rows if '✅' in r['Result'])
        st.metric("Significant Features", f"{sig}/{len(t_sel)}")
        fig = go.Figure(go.Bar(x=tt_df['Feature'], y=tt_df['p-value'],
                               marker=dict(color=tt_df['p-value'], colorscale='RdYlGn_r'),
                               text=tt_df['p-value'].round(5), textposition='outside'))
        fig.add_hline(y=alpha, line_dash='dash', line_color=C4, annotation_text=f'α={alpha}')
        fig = themed(fig, h=340, title='T-Test p-values', yt='p-value')
        st.plotly_chart(fig, use_container_width=True)

    if cat_cols:
        sec("Chi-Square Test — Categorical Features")
        ibox("<b>H₀:</b> Feature and class are independent &nbsp;|&nbsp; <b>H₁:</b> Feature and class are related")
        chi_rows=[]
        for c in cat_cols:
            ct = pd.crosstab(df[c], df['class'])
            chi2,p,dof,_ = chi2_contingency(ct)
            chi_rows.append({'Feature':c,'Chi²':round(chi2,4),'p-value':round(p,6),'DoF':dof,
                             'Result':'✅ Related' if p<alpha else '❌ Independent'})
        st.dataframe(pd.DataFrame(chi_rows), use_container_width=True, hide_index=True)

# ── OUTLIER DETECTION ────────────────────────────────────────────────────────────
elif "Outlier" in page:
    sec("🔍 Outlier Detection — IQR Method")
    iqr_m  = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)
    o_sel  = st.multiselect("Select Features", num_cols, default=num_cols[:8])
    if o_sel:
        rows=[]
        for c in o_sel:
            Q1,Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            IQR=Q3-Q1; lo,hi=Q1-iqr_m*IQR, Q3+iqr_m*IQR
            n_out = df[(df[c]<lo)|(df[c]>hi)].shape[0]
            rows.append({'Feature':c,'Q1':round(Q1,3),'Q3':round(Q3,3),'IQR':round(IQR,3),
                         'Lower':round(lo,3),'Upper':round(hi,3),'Outliers':n_out,'%':round(n_out/N*100,2)})
        out_df = pd.DataFrame(rows)
        st.dataframe(out_df, use_container_width=True, hide_index=True)
        fig = go.Figure(go.Bar(x=out_df['Feature'], y=out_df['Outliers'],
                               marker_color=C4, text=out_df['Outliers'], textposition='outside'))
        fig = themed(fig, h=300, title='Outlier Count per Feature', yt='Count')
        st.plotly_chart(fig, use_container_width=True)
    col_b2 = st.selectbox("Feature Boxplot", num_cols)
    fig = go.Figure()
    fig.add_trace(go.Box(y=dr[col_b2],    name='Has DR', marker_color='#ef4444', boxmean='sd'))
    fig.add_trace(go.Box(y=no_dr[col_b2], name='No DR',  marker_color=C2,       boxmean='sd'))
    fig = themed(fig, h=400, title=f'Boxplot — {col_b2}', yt=col_b2)
    st.plotly_chart(fig, use_container_width=True)
    ibox("<b>Medical Decision:</b> Outliers in exudate columns are retained — severe DR patients naturally exhibit high values. Removing them would discard real clinical cases.")

# ── LIVE PREDICTION ──────────────────────────────────────────────────────────────
elif "Prediction" in page:
    sec("🤖 Live DR Risk Prediction")
    t1, t2 = st.tabs(["🎛️ Manual Input", "🎲 Random Patient Demo"])

    with t1:
        model_choice = st.selectbox("Select Model", ["Logistic Regression","Random Forest"])
        threshold    = st.slider("Decision Threshold", 0.10, 0.90, 0.30, 0.05,
                                 help="Lower = higher sensitivity — fewer missed DR cases")
        cols3 = st.columns(3)
        patient_vals = {}
        for i,feat in enumerate(feat_cols):
            with cols3[i % 3]:
                mn = float(df[feat].min()); mx = float(df[feat].max()); me = float(df[feat].mean())
                if df[feat].nunique() <= 5:
                    opts = sorted(df[feat].unique().tolist())
                    def_idx = min(range(len(opts)), key=lambda j: abs(opts[j]-me))
                    patient_vals[feat] = st.selectbox(feat, opts, index=def_idx)
                else:
                    patient_vals[feat] = st.slider(feat, mn, mx, me, step=round((mx-mn)/200, 6))

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔬 Run Prediction", use_container_width=True):
            inp    = np.array([[patient_vals[f] for f in feat_cols]])
            inp_sc = sc.transform(inp)
            model  = lr if model_choice=="Logistic Regression" else rf
            prob   = model.predict_proba(inp_sc)[0][1]
            pred   = 1 if prob >= threshold else 0
            risk,badge = (("Low Risk","badge-low") if prob<0.3 else
                          ("Medium Risk","badge-med") if prob<0.6 else ("High Risk","badge-high"))
            if pred==1:
                st.markdown(f'<div class="pred-box pred-box-dr"><div class="pred-icon">⚠️</div><div class="pred-title" style="color:#ef4444">Diabetic Retinopathy Detected</div><div class="pred-prob" style="color:#fca5a5">Probability: {prob*100:.1f}%</div><div><span class="badge {badge}">{risk}</span></div><div class="pred-sub">Immediate ophthalmological consultation recommended.</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pred-box pred-box-ok"><div class="pred-icon">✅</div><div class="pred-title" style="color:#34d399">No Diabetic Retinopathy</div><div class="pred-prob" style="color:#6ee7b7">Probability: {prob*100:.1f}%</div><div><span class="badge {badge}">{risk}</span></div><div class="pred-sub">Regular annual retinal screening advised.</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob*100,
                number={'suffix':'%','font':{'color':TXT,'size':32,'family':'Outfit'}},
                gauge=dict(axis=dict(range=[0,100],tickcolor='#64748b'),
                           bar=dict(color='#ef4444' if prob>0.5 else C2,thickness=0.25),
                           bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)',
                           steps=[dict(range=[0,30],color='rgba(52,211,153,0.1)'),
                                  dict(range=[30,60],color='rgba(251,191,36,0.1)'),
                                  dict(range=[60,100],color='rgba(239,68,68,0.1)')],
                           threshold=dict(line=dict(color='#94a3b8',width=3),value=threshold*100)),
                title=dict(text='DR Risk Gauge',font=dict(color='#64748b',size=13))))
            fig.update_layout(paper_bgcolor=BG, font=dict(color=TXT,family='Outfit'),
                              height=270, margin=dict(l=30,r=30,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

    with t2:
        n_demo   = st.slider("Number of Random Patients", 1, 10, 3)
        thresh2  = st.slider("Threshold", 0.10, 0.90, 0.30, 0.05, key='demo_t')
        if st.button("🎲 Pick Random Patients & Predict", use_container_width=True):
            idxs = np.random.choice(len(df), n_demo, replace=False)
            results=[]
            for idx in idxs:
                row    = df.iloc[idx]
                actual = row['class']
                feats  = row.drop('class').values.reshape(1,-1)
                lr_p   = lr.predict_proba(sc.transform(feats))[0][1]
                rf_p   = rf.predict_proba(sc.transform(feats))[0][1]
                lr_pred = 1 if lr_p>=thresh2 else 0
                rf_pred = 1 if rf_p>=thresh2 else 0
                risk_lbl = "🟢 Low" if rf_p<0.3 else ("🟡 Medium" if rf_p<0.6 else "🔴 High")
                results.append({'Patient #':idx,'Actual':'Has DR ⚠️' if actual==1 else 'No DR ✅',
                                 'LR Pred':'Has DR' if lr_pred==1 else 'No DR','LR Prob':f"{lr_p*100:.1f}%",
                                 'RF Pred':'Has DR' if rf_pred==1 else 'No DR','RF Prob':f"{rf_p*100:.1f}%",
                                 'Risk':risk_lbl,'RF Correct?':'✅' if rf_pred==actual else '❌'})
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            correct = sum(1 for r in results if r['RF Correct?']=='✅')
            st.metric("RF Correct Predictions", f"{correct}/{n_demo}", f"{correct/n_demo*100:.0f}%")

# ── MODEL EVALUATION ─────────────────────────────────────────────────────────────
elif "Evaluation" in page:
    sec("📈 Model Evaluation")
    thresh_e = st.slider("Threshold", 0.10, 0.90, 0.30, 0.05, key='ev_t')
    lr_probs = lr.predict_proba(Xte)[:,1]
    rf_probs = rf.predict_proba(Xte)[:,1]
    lr_preds = (lr_probs>=thresh_e).astype(int)
    rf_preds = (rf_probs>=thresh_e).astype(int)
    models_e = {'Logistic Regression':(lr_preds,lr_probs),'Random Forest':(rf_preds,rf_probs)}

    rows=[]
    for name,(preds,probs) in models_e.items():
        rows.append({'Model':name,
                     'Accuracy':f"{accuracy_score(yte,preds)*100:.2f}%",
                     'Precision':f"{precision_score(yte,preds)*100:.2f}%",
                     'Recall':f"{recall_score(yte,preds)*100:.2f}%",
                     'F1 Score':f"{f1_score(yte,preds)*100:.2f}%",
                     'AUC':f"{roc_auc_score(yte,probs):.4f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("LR Recall",f"{recall_score(yte,lr_preds)*100:.1f}%")
    m2.metric("RF Recall",f"{recall_score(yte,rf_preds)*100:.1f}%")
    m3.metric("LR AUC",f"{roc_auc_score(yte,lr_probs):.4f}")
    m4.metric("RF AUC",f"{roc_auc_score(yte,rf_probs):.4f}")

    t1,t2,t3 = st.tabs(["🔲 Confusion Matrix","📉 ROC Curve","🏆 Feature Importance"])
    with t1:
        c1,c2 = st.columns(2)
        for col,(name,(preds,_)) in zip([c1,c2],models_e.items()):
            with col:
                cm = confusion_matrix(yte,preds)
                fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                x=['No DR','Has DR'], y=['No DR','Has DR'], aspect='auto')
                fig = themed(fig, h=300, title=f'Confusion Matrix — {name}')
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
                tn,fp,fn,tp = cm.ravel()
                ibox(f"TP={tp} | TN={tn} | FP={fp} | <b>FN={fn}</b> ← missed DR cases")

    with t2:
        fig = go.Figure()
        for (name,(_,probs)),clr in zip(models_e.items(),['#ef4444',C2]):
            fpr,tpr,_ = roc_curve(yte,probs)
            auc = roc_auc_score(yte,probs)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{name} (AUC={auc:.3f})',
                                     line=dict(color=clr,width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],line=dict(dash='dash',color='#475569'),
                                 name='Random Guessing'))
        fig = themed(fig, h=440, title='ROC Curve', xt='False Positive Rate', yt='True Positive Rate')
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation='h',
                               marker=dict(color=imp.values, colorscale='Blues'),
                               text=imp.round(4).values, textposition='outside'))
        fig = themed(fig, h=480, title='Feature Importance — Random Forest', xt='Score')
        st.plotly_chart(fig, use_container_width=True)
        top3 = imp.sort_values(ascending=False).head(3)
        ibox(f"<b>Top 3 Features:</b> {', '.join([f'{k} ({v:.4f})' for k,v in top3.items()])}")
