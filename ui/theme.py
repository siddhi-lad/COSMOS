"""Visual identity: deep-forest sidebar, warm cream canvas, serif headlines."""
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');
:root{--forest:#1F4D3A;--forest-d:#14352A;--sage:#E6EFE8;--cream:#FAF8F4;--blush:#F7E6E2;--gold:#B08D57;
--ink:#1B2A24;--muted:#66756D;--line:#E3E8E2;--low:#2E7D5B;--med:#B7791F;--high:#B83A4B;--unk:#7A857F;}
html,body,.stApp,[class*="css"]{font-family:'Inter',system-ui,sans-serif;}
.stApp{background:var(--cream);}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:2.2rem;max-width:1180px;}
h1,h2,h3{font-family:'Fraunces',Georgia,serif!important;color:var(--forest-d);letter-spacing:-.01em;}
h3{font-size:1.25rem!important;}

/* sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,var(--forest-d),var(--forest));}
[data-testid="stSidebar"] *{color:#E8F0EA;}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a{border-radius:10px;padding:.35rem .6rem;}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"]{background:rgba(255,255,255,.14);font-weight:600;}
[data-testid="stSidebar"] label p,[data-testid="stSidebar"] .stMarkdown p{font-size:.82rem;}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.15);}
[data-testid="stSidebar"] input{color:#1B2A24!important;}
[data-testid="stSidebar"] [data-baseweb="select"] *{color:#1B2A24!important;}
[data-testid="stSidebar"] button{color:#1B2A24!important;}
[data-testid="stSidebar"] [data-testid="stExpander"],[data-testid="stSidebar"] [data-testid="stExpander"] details{background:rgba(255,255,255,.07)!important;border:1px solid rgba(255,255,255,.18);}
[data-testid="stSidebar"] [data-testid="stExpander"] summary,[data-testid="stSidebar"] [data-testid="stExpander"] summary *{color:#E8F0EA!important;}
[data-testid="stSidebar"] [data-testid="stExpander"] button{background:#F4EFE6;}
[data-testid="stSidebar"] [data-testid="stExpander"] button *{color:#1B2A24!important;}

/* hero */
.hero{background:linear-gradient(135deg,#E6EFE8 0%,#F4EDE3 60%,#F7E6E2 100%);border:1px solid var(--line);
border-radius:20px;padding:1.6rem 1.8rem;margin-bottom:1.2rem;}
.hero h1{margin:0 0 .25rem 0;font-size:2.05rem;}
.hero p{margin:0;color:var(--muted);font-size:1rem;}
.eyebrow{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--gold);font-weight:600;margin-bottom:.35rem;}

/* cards & pills */
.card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:1rem 1.2rem;margin-bottom:.7rem;
box-shadow:0 1px 2px rgba(20,53,42,.04);}
.card h4{margin:0 0 .25rem 0;font-size:.98rem;color:var(--ink);}
.card p{margin:.15rem 0;color:#33443B;font-size:.9rem;line-height:1.5;}
.meta{color:var(--muted);font-size:.78rem;}
.pill{display:inline-block;padding:.12rem .6rem;border-radius:999px;font-size:.74rem;font-weight:600;margin-right:.35rem;}
.pill.Low{background:#E2F2EA;color:var(--low);} .pill.Medium{background:#FBEFD5;color:var(--med);}
.pill.High{background:#F8DDE1;color:var(--high);} .pill.Unknown{background:#ECEFED;color:var(--unk);}
.pill.src{background:var(--sage);color:var(--forest);} .pill.cur{background:#F3ECDF;color:#8A6A35;}

/* metrics */
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:14px;padding:.8rem 1rem;}
[data-testid="stMetricValue"]{font-family:'Fraunces',serif;color:var(--forest-d);}
[data-testid="stMetricLabel"] p{color:var(--muted);}

/* buttons */
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:var(--forest);border:0;border-radius:12px;
padding:.55rem 1.3rem;font-weight:600;}
.stButton>button[kind="primary"]:hover{background:var(--forest-d);}
.stButton>button,.stLinkButton>a{border-radius:12px;}
.stButton>button:disabled{background:#DCE5DF!important;color:#7A8A80!important;border:0;}
[data-testid="stTabs"] [role="tab"]{font-weight:500;}
[data-testid="stFileUploaderDropzone"]{border:1.5px dashed #9DB7A8;border-radius:16px;background:#fff;}
[data-testid="stChatMessage"]{background:#fff;border:1px solid var(--line);border-radius:16px;}
[data-testid="stExpander"]{background:#fff;border-radius:14px;border:1px solid var(--line);}
.empty{background:#fff;border:1px dashed #B9C9BE;border-radius:18px;padding:2rem;text-align:center;color:var(--muted);}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def pill(text: str, cls: str = "") -> str:
    return f'<span class="pill {cls}">{text}</span>'


def empty(message: str) -> None:
    st.markdown(f'<div class="empty">{message}</div>', unsafe_allow_html=True)
