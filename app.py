"""Banaras Ghat Bot - Streamlit + LangChain + FAISS RAG over Wikipedia + local notes."""

import html
import json
import os
import re
import threading
import urllib.parse
from functools import lru_cache
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import wiki

DATA = Path(__file__).parent / "data" / "places.json"
CATEGORIES = [ # key, icon, title, subtitle, question for the chat
    ("ghat", "🛕", "Famous Ghats", "Explore the most popular ghats", "Tell me about the famous ghats of Banaras"),
    ("temple", "🛕", "Temples", "Spiritual places near ghats", "Which temples should I visit in Varanasi?"),
    ("boat", "🛶", "Boat Rides", "Experience Varanasi from the river", "Tell me about boat rides and Ganga Aarti"),
    ("food", "🍽️", "Local Food", "Taste the flavours of Banaras", "What local food should I try in Varanasi?"),
    ("reach", "📍", "How to Reach", "Transport & travel guidance", "How do I reach Varanasi?"),
]
CAT_TITLE = {c[0]: c[2] for c in CATEGORIES} | {"house": "Local Houses & Stays"}
SUGGESTIONS = ["Aarti ka time kya hai?", "Best ghat for sunrise?", "Tell me about Bhelpur haveli", "Nearby food options?"]
FAQS = [
    ("What is the best time to visit Varanasi?", "October to March has pleasant weather. Summers are very hot and monsoon floods can close some ghats."),
    ("Is the Ganga Aarti free?", "Yes, the evening Ganga Aarti at Dashashwamedh Ghat is free. Arrive 45-60 minutes early for a seat."),
    ("How much does a boat ride cost?", "Roughly Rs 300-500 per person on a shared boat, or Rs 1,500-2,500 for a private boat. Always fix the price first."),
    ("Is it safe for solo travellers?", "Generally yes. Avoid unofficial guides and touts near the ghats, and keep valuables safe in crowded lanes."),
    ("Can I take photos at Manikarnika Ghat?", "Photography of cremations is not permitted and is disrespectful. View quietly from a boat."),
]

# ------------------------------ Data ------------------------------
@st.cache_data
def load_data() -> list[dict]:
    places = json.loads(DATA.read_text(encoding="utf-8"))
    cache = wiki.build_cache(places)
    fallback = cache.get("about_varanasi", {}).get("image", "")
    for p in places:
        w = cache.get(p["id"], {})
        p["wiki_text"] = w.get("text", "")
        p["wiki_url"] = w.get("url", "")
        p["image"] = w.get("image") or fallback
        p["map"] = "https://www.google.com/maps/search/?api=1&query=" + urllib.parse.quote(p["name"] + " Varanasi")
    return places

@lru_cache(maxsize=1)
def _index_by_id() -> dict:
    return {p["id"]: p for p in load_data()}

def by_id(pid: str) -> dict:
    return _index_by_id()[pid]

def thumb(url: str, width: int) -> str:
    return re.sub(r"/\d+px-", f"/{width}px-", url) if "wikimedia.org" in url else url

# ------------------------------ RAG ------------------------------
def chunk(text: str, size: int = 700) -> list[str]:
    out, cur = [], ""
    for para in text.split("\n\n"):
        if len(cur) + len(para) > size and cur:
            out.append(cur)
            cur = ""
        cur += para + "\n\n"
    if cur.strip():
        out.append(cur.strip())
    return out

_INDEX = {}
_LOCK = threading.Lock()

def build_index():
    with _LOCK:
        if "r" not in _INDEX:
            _INDEX["r"] = _make_index()
        return _INDEX["r"]

@st.cache_resource
def warm_up():
    threading.Thread(target=build_index, daemon=True).start()
    return True

def _make_index():
    from langchain_core.documents import Document
    docs = []
    for p in load_data():
        docs.append(Document(page_content=f"{p['name']}. {p['details']}", metadata={"id": p["id"], "src": "notes"}))
        for c in chunk(p["wiki_text"]):
            docs.append(Document(page_content=f"{p['name']}. {c}", metadata={"id": p["id"], "src": "wikipedia"}))
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_openai import OpenAIEmbeddings
            return FAISS.from_documents(docs, OpenAIEmbeddings(model="text-embedding-3-small"))
        except Exception:
            pass
    from langchain_community.retrievers import TFIDFRetriever
    return TFIDFRetriever.from_documents(docs, k=8, tfidf_params={"ngram_range": (1, 2), "sublinear_tf": True, "stop_words": "english"})

def search(query: str, k: int = 6) -> list:
    r = build_index()
    docs = r.similarity_search(query, k=k) if hasattr(r, "similarity_search") else r.invoke(query)
    return list(docs)[:k]

STOP_WORDS = {"ghat", "temple", "mandir", "the", "of", "on", "ganga", "by", "varanasi", "bhavan", "niwas", "air", "train", "boat", "rides", "and", "about"}

def match_place(q: str):
    words = set(re.findall(r"[a-z]+", q.lower()))
    best, score = None, 0
    for p in load_data():
        toks = {t for t in re.findall(r"[a-z]+", p["name"].lower()) if t not in STOP_WORDS and len(t) > 3}
        s = len(toks & words)
        if s > score:
            best, score = p, s
    return best

def pick_place(q: str, hits: list) -> dict:
    named = match_place(q)
    if named:
        return named
    if "aarti" in q:
        return by_id("ganga_aarti")
    note = next((h for h in hits if h.metadata["src"] == "notes"), None)
    if note or hits:
        return by_id((note or hits[0]).metadata["id"])
    return by_id("about_varanasi")

def answer(query: str) -> dict:
    q = query.lower().strip(" !?")
    if q in {"hi", "hello", "hey", "namaste", "hii"}:
        return {"role": "bot", "text": "🙏 **Namaste!** Ask me about any ghat, temple, food, boat ride or locality in Banaras."}

    hits = search(query)
    top = pick_place(q, hits)
    others = [by_id(h.metadata["id"]) for h in hits if h.metadata["id"] != top["id"]]
    context = f"{top['name']}. {top['details']}\n\n" + "\n\n".join(h.page_content for h in hits[:3])

    text = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            text = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).invoke(
                f"You are Banaras Ghat Bot, a friendly Varanasi guide. Answer briefly in Markdown using only this context. If unsure, say so.\n\nContext:\n{context}\n\nQuestion: {query}").content
        except Exception:
            text = None

    if not text:
        text = f"**{top['name']}**\n\n{top['details']}"
        wp = next((c for c in chunk(top["wiki_text"], 450)), "")
        if wp:
            text += f"\n\n**From Wikipedia:** {wp[:450].strip()}..."
        if others:
            text += f"\n\n_Also see: {others[0]['name']}._"

    if top["wiki_url"]:
        text += f"\n\n[Read more on Wikipedia]({top['wiki_url']})"
    img = by_id("ganga_aarti")["image"] if "aarti" in q else top["image"]
    return {"role": "bot", "text": text, "image": img, "map": top["map"], "place": top["name"]}

# ------------------------------ State / callbacks ------------------------------
def init_state():
    st.session_state.setdefault("view", ("home",))
    st.session_state.setdefault("contacts", [])
    st.session_state.setdefault("chat_open", True)
    st.session_state.setdefault("messages", [
        {"role": "bot", "text": "🙏 **Namaste!** Ask me about any ghat, temple, food or locality in Banaras."}
    ])

def ask(q: str):
    q = (q or "").strip()
    if q:
        st.session_state.chat_open = True
        st.session_state.messages.append({"role": "user", "text": q})
        st.session_state.messages.append(answer(q))

def set_chat(is_open: bool):
    st.session_state.chat_open = is_open

def ask_from(key: str):
    ask(st.session_state.get(key, ""))
    st.session_state[key] = ""

def go(*view):
    st.session_state.view = view

def open_category(cat: str, q: str):
    go("category", cat)
    places = [p for p in load_data() if p["category"] == cat]
    lines = "\n".join(f"- **{p['name']}**: {p['short']}" for p in places)
    st.session_state.chat_open = True
    st.session_state.messages.append({"role": "user", "text": q})
    st.session_state.messages.append({"role": "bot", "text": f"Here are the **{CAT_TITLE[cat]}**:\n\n{lines}\n\nTap any card on the page, or ask me about one of them!", "image": places[0]["image"] if places else None})

def open_place(pid: str):
    go("place", pid)
    ask(f"Tell me about {by_id(pid)['name']}")

# ------------------------------ UI ------------------------------
CSS = """
<style>
/* Force white background & black text for main page */
.stApp, [data-testid="stAppViewContainer"], .main { background-color: #ffffff !important; }
.st-key-body, .st-key-body p, .st-key-body span, .st-key-body div { color: #111111 !important; }

/* Hide Streamlit Defaults */
#MainMenu, footer, header[data-testid="stHeader"] {display:none;}
.block-container {padding:0 !important; max-width:100% !important;}

/* Navigation */
.st-key-nav {background:#fdf6ee; padding:10px 40px;}
.st-key-nav button {background:transparent; border:none; color:#333 !important; box-shadow:none;}
.st-key-nav button:hover {color:#e8731a !important;}
.brand b {font-size:22px; color:#14305a !important; font-family:Georgia,serif;} 
.brand small {display:block; color:#14305a !important;}

/* Hero Section (Keep Text White) */
.st-key-hero {background-size:cover !important; background-position:center !important; padding:50px 40px 40px; min-height:250px;}
.st-key-hero h1 {color:#fff !important; font-family:Georgia,serif; font-size:2.8rem; margin:0; text-shadow:0 2px 6px #0008;}
.st-key-hero p, .st-key-hero small {color:#fff !important; text-shadow:0 1px 4px #0008;}
.st-key-hero input {border-radius:30px; padding:14px 20px; color: #111111 !important;}

/* Body */
.st-key-body {padding:10px 40px; background-color: #ffffff !important;}
h3.t {font-family:Georgia,serif; color:#14305a !important; margin:20px 0 10px;}

/* Category Cards */
div[class*="st-key-cat"] button {width:100%; height:120px; border:none; border-radius:12px; white-space:pre-line; color:#222 !important;}
.st-key-cat0 button {background:#fbe5dc;} .st-key-cat1 button {background:#f0ebd3;}
.st-key-cat2 button {background:#e1ecf8;} .st-key-cat3 button {background:#dff1e5;} .st-key-cat4 button {background:#e9e4f6;}

/* Place Cards */
.card {background:#fff; border-radius:10px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,.1); border: 1px solid #eee;}
.card img {width:100%; height:120px; object-fit:cover;}
.card div {padding:10px; min-height:95px; font-size:.85rem; color:#555 !important; background:#fff !important;}
.card b {display:block; color:#14305a !important; font-size:1rem;}

/* Buttons (Fix for invisible text) */
div[class*="st-key-view"] button, .st-key-send button {background:#14305a !important; color:#fff !important; border-radius:20px;}
div[class*="st-key-view"] button p, .st-key-send button p, .st-key-hero_go button p { color: #ffffff !important; }

/* Chat Window */
.st-key-chat {position:fixed; top:70px; right:20px; width:370px; max-width:calc(100vw - 20px); background:#fff;
    border-radius:16px; box-shadow:0 8px 30px rgba(0,0,0,.20); padding:0 !important; z-index:999; gap:0 !important;
    max-height:calc(100vh - 90px); overflow-y:auto; overflow-x:hidden; overscroll-behavior:contain;}
div:has(> .st-key-chat_head), .st-key-chat_head {position:sticky; top:0; z-index:5;}
div:has(> .st-key-chat_foot), .st-key-chat_foot {position:sticky; bottom:0; z-index:5; background:#fff !important;}

/* Chat Header */
.st-key-chat_head {background:#14305a !important; padding:10px 14px; border-radius:16px 16px 0 0;}
.st-key-chat_head, .st-key-chat_head * {color:#fff !important;}
.st-key-chat_head p {margin:0; line-height:1.3;}
.st-key-chat_head button {background:rgba(255,255,255,.15); border:none; border-radius:50%; width:34px; height:34px; min-height:0; padding:0; color:#fff !important;}
.st-key-chat_head button:hover {background:#e8731a;}

/* Chat Messages */
.st-key-msgs {min-height:150px; padding:12px; background:#fdf8f2;}
.st-key-msgs img {max-height:220px; object-fit:cover; border-radius:10px;}
.st-key-msgs [data-testid="stChatMessage"] {background:#fff; border:1px solid #eee; border-radius:14px; padding:8px 10px; margin-bottom:8px; font-size:.9rem;}
.st-key-msgs [data-testid="stChatMessage"] p, .st-key-msgs li {font-size:.9rem; line-height:1.45; color: #111111 !important;}

.st-key-chat_foot {padding:8px 12px 12px; border-top:1px solid #eee;}
.online {color:#7df0b0 !important; font-size:.75rem;}
.me {scroll-margin-top:70px; background:#14305a !important; color:#fff !important; padding:8px 12px; border-radius:14px 14px 2px 14px; margin:6px 0 8px 36px; font-size:.9rem;}

/* Chat Launcher */
.st-key-launcher {position:fixed !important; right:20px !important; bottom:20px !important; left:auto !important; top:auto !important;
    width:68px !important; height:68px !important; z-index:9999; overflow:visible;}
.st-key-launcher [data-testid="stButton"] {width:68px !important; height:68px;}
.st-key-launcher button {width:68px !important; height:68px !important; min-height:68px !important; border-radius:50% !important; padding:0 !important;
    background:radial-gradient(circle at 30% 30%, #ff9a3c, #e8731a 60%, #b8520c) !important; border:3px solid #fff !important;
    box-shadow:0 6px 20px rgba(20,48,90,.5) !important; animation:pulse 2.2s infinite;}
.st-key-launcher button p {font-size:34px !important; line-height:1 !important; margin:0; color:#fff !important;}
.st-key-launcher button:hover {transform:scale(1.1); background:#14305a !important;}
@keyframes pulse {0%{box-shadow:0 0 0 0 rgba(232,115,26,.6);} 70%{box-shadow:0 0 0 16px rgba(232,115,26,0);} 100%{box-shadow:0 0 0 0 rgba(232,115,26,0);}}

/* Layout Fixes */
.st-key-chat [data-testid="stHorizontalBlock"], .st-key-nav [data-testid="stHorizontalBlock"], .st-key-hero [data-testid="stHorizontalBlock"]
    {flex-wrap:wrap; gap:.4rem;}
.st-key-chat [data-testid="stColumn"], .st-key-hero [data-testid="stColumn"] {min-width:0 !important;}
.st-key-nav [data-testid="stColumn"] {min-width:auto !important; flex:1 1 auto !important; width:auto !important;}

/* Tablet / Phone */
@media (max-width: 900px) {
    .st-key-chat {top:0; right:0; left:0; bottom:0; width:100%; max-width:100%; max-height:100vh; height:100dvh; border-radius:0;}
    .st-key-chat_head {border-radius:0;}
    .st-key-nav {padding:8px 12px;}
    .st-key-hero {padding:30px 16px 24px; min-height:200px;}
    .st-key-hero h1 {font-size:1.9rem;}
    .st-key-body {padding:8px 14px;}
    .st-key-body [data-testid="stColumn"] {min-width:45% !important;}
    .foot {flex-direction:column; gap:8px; text-align:center; padding:18px;}
    .st-key-launcher {right:14px !important; bottom:14px !important;}
}

/* Suggestion Pills */
div[class*="st-key-pill"] button {width:100%; min-height:44px; line-height:1.2; border-radius:16px; font-size:.75rem; background:#eaf1fb !important; color:#14305a !important; border:none; padding:8px 10px;}
div[class*="st-key-pill"] button p { color: #14305a !important; }

/* Footer */
.foot {background:#14305a !important; color:#fff !important; display:flex; justify-content:space-between; padding:22px 40px; margin-top:30px;}
.foot a, .foot b, .foot i, .foot span {color:#fff !important; margin:0 6px; text-decoration: none;}
</style>
"""

def header():
    with st.container(key="nav"):
        c = st.columns([5, 1, 1, 1, 1, 1])
        c[0].markdown("<div class='brand'><b>🚩 Banaras Ghat Bot</b><small>Explore • Know • Experience</small></div>", unsafe_allow_html=True)
        c[1].button("Home", key="n_home", on_click=go, args=("home",))
        c[2].button("About", key="n_about", on_click=go, args=("place", "about_varanasi"))
        c[3].button("Ghats", key="n_ghats", on_click=go, args=("category", "ghat"))
        c[4].button("FAQs", key="n_faq", on_click=go, args=("faqs",))
        c[5].button("Contact", key="n_contact", on_click=go, args=("contact",))

def hero():
    img = by_id("dashashwamedh")["image"].replace("/330px-", "/1280px-")
    st.markdown(f"<style>.st-key-hero{{background:linear-gradient(rgba(0,0,0,.35),rgba(0,0,0,.2)),url('{img}');}}</style>", unsafe_allow_html=True)
    with st.container(key="hero"):
        st.markdown("<small>WELCOME TO</small><h1>Banaras Ghat Bot</h1><p>Your friendly guide to Varanasi's ghats, temples, culture, food and more.<br>Ask me anything!</p>", unsafe_allow_html=True)
        c1, c2 = st.columns([8, 1])
        c1.text_input("s", key="hero_q", placeholder="Ask me about any ghat, history, timings, places, food...", label_visibility="collapsed", on_change=ask_from, args=("hero_q",))
        c2.button("➤", key="hero_go", on_click=ask_from, args=("hero_q",))

def place_cards(places: list[dict], per_row: int = 4, prefix: str = "view"):
    for i in range(0, len(places), per_row):
        cols = st.columns(per_row)
        for col, p in zip(cols, places[i:i + per_row]):
            with col:
                st.markdown(f"<div class='card'><img src='{thumb(p['image'], 330)}' loading='lazy'><div><b>{html.escape(p['name'])}</b><br>{html.escape(p['short'])}</div></div>", unsafe_allow_html=True)
                st.button("View Details →", key=f"{prefix}_{p['id']}", on_click=open_place, args=(p['id'],))

def home_view():
    cols = st.columns(5)
    # Bug fix for original code
    for i, (col, cat) in enumerate(zip(cols, CATEGORIES)):
        key, icon, title, sub, q = cat
        col.button(f"{icon}\n**{title}**\n{sub}", key=f"cat{i}", on_click=open_category, args=(key, q))
    c1, c2 = st.columns([6, 1])
    c1.markdown("<h3 class='t'>Popular Ghats</h3>", unsafe_allow_html=True)
    c2.button("View All →", key="viewall", on_click=open_category, args=("ghat", CATEGORIES[0][4]))
    place_cards([by_id(i) for i in ("dashashwamedh", "assi", "manikarnika", "panchganga")])

def category_view(cat: str):
    st.button("← Back to Home", key="back", on_click=go, args=("home",))
    st.markdown(f"<h3 class='t'>{CAT_TITLE[cat]}</h3>", unsafe_allow_html=True)
    place_cards([p for p in load_data() if p["category"] == cat], per_row=3, prefix=f"v{cat}")

def place_view(pid: str):
    p = by_id(pid)
    st.button("← Back to Home", key="back", on_click=go, args=("home",))
    c1, c2 = st.columns([1, 1.4])
    c1.image(p["image"])
    with c2:
        st.markdown(f"<h3 class='t'>{p['name']}</h3>", unsafe_allow_html=True)
        st.markdown(p["details"])
        st.markdown(f"📍 [View on Map]({p['map']})" + (f" &nbsp;| &nbsp; 📖 [Wikipedia]({p['wiki_url']})" if p["wiki_url"] else ""))
        st.button("🤖 Ask the bot about this", key="askbot", on_click=ask, args=(f"What should I know before visiting {p['name']}?",))
    if p["wiki_text"]:
        st.markdown("#### From Wikipedia")
        paras = p["wiki_text"].split("\n\n")
        st.markdown("\n\n".join(paras[:3]))
        if len(paras) > 3:
            with st.expander("Read more"):
                st.markdown("\n\n".join(paras[3:]))
    same = [x for x in load_data() if x["category"] == p["category"] and x["id"] != pid][:3]
    if same:
        st.markdown("<h3 class='t'>You may also like</h3>", unsafe_allow_html=True)
        place_cards(same, per_row=3, prefix="alike")

def faqs_view():
    st.button("← Back to Home", key="back", on_click=go, args=("home",))
    st.markdown("<h3 class='t'>Frequently Asked Questions</h3>", unsafe_allow_html=True)
    for q, a in FAQS:
        with st.expander(q):
            st.write(a)
            st.button("Ask the bot", key=f"faq_{q[:12]}", on_click=ask, args=(q,))

def contact_view():
    st.button("← Back to Home", key="back", on_click=go, args=("home",))
    st.markdown("<h3 class='t'>Contact Us</h3>", unsafe_allow_html=True)
    with st.form("contact"):
        name, email = st.text_input("Name"), st.text_input("Email")
        msg = st.text_area("Message")
        if st.form_submit_button("Send"):
            if name and email and msg:
                st.session_state.contacts.append({"name": name, "email": email, "message": msg})
                st.success("Thank you! We'll get back to you soon.")
            else:
                st.warning("Please fill all fields.")

def chatbot_interface():
    if not st.session_state.chat_open:
        with st.container(key="launcher"):
            st.button("🤖", key="open_chat", on_click=set_chat, args=(True,), help="Chat with Banaras Ghat Bot")
        return
    with st.container(key="chat"):
        with st.container(key="chat_head"):
            h1, h2 = st.columns([6, 1], vertical_alignment="center")
            h1.markdown("🤖 **Banaras Ghat Bot**<br><span class='online'>● Online</span>", unsafe_allow_html=True)
            h2.button("✕", key="close_chat", on_click=set_chat, args=(False,), help="Close chat")
        with st.container(key="msgs"):
            for m in st.session_state.messages:
                if m["role"] == "user":
                    st.markdown(f"<div class='me'>{html.escape(m['text'])}</div>", unsafe_allow_html=True)
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(m["text"])
                        if m.get("image"):
                            st.image(thumb(m["image"], 500))
                        if m.get("map"):
                            st.markdown(f"📍 [{m['place']}]({m['map']})")
        with st.container(key="chat_foot"):
            st.caption("You can also ask:")
            c = st.columns(2)
            for i, s in enumerate(SUGGESTIONS):
                c[i % 2].button(s, key=f"pill{i}", on_click=ask, args=(s,))
            c1, c2 = st.columns([5, 1])
            c1.text_input("m", key="chat_q", placeholder="Type your message...", label_visibility="collapsed", on_change=ask_from, args=("chat_q",))
            c2.button("➤", key="send", on_click=ask_from, args=("chat_q",))
    components.html(
        f"<script>/* {len(st.session_state.messages)} */"
        "function go(){const d=window.parent.document;const chat=d.querySelector('.st-key-chat');"
        "const me=d.querySelectorAll('.st-key-msgs .me');if(!chat||!me.length)return;"
        "const last=me[me.length-1];"
        "chat.scrollTop+=last.getBoundingClientRect().top-chat.getBoundingClientRect().top-80;}"
        "go();setTimeout(go,150);setTimeout(go,600);setTimeout(go,1500);</script>", height=0)

def footer():
    st.markdown("<div class='foot'><b>🚩 Banaras Ghat Bot</b><i>\"Kashi sirf ek shehar nahi, ek anubhav hai...\"</i>"
        "<span><a href='https://instagram.com' target='_blank'>Instagram</a>"
        "<a href='https://youtube.com' target='_blank'>YouTube</a>"
        "<a href='https://x.com' target='_blank'>X</a></span>"
        "<span>Explore • Know • Experience</span></div>", unsafe_allow_html=True)

def render_ui():
    st.markdown(CSS, unsafe_allow_html=True)
    if st.session_state.chat_open:
        st.markdown("<style>@media (min-width:901px){.st-key-body{margin-right:400px;}}</style>", unsafe_allow_html=True)
    header()
    hero()
    with st.container(key="body"):
        view = st.session_state.view
        if view[0] == "category":
            category_view(view[1])
        elif view[0] == "place":
            place_view(view[1])
        elif view[0] == "faqs":
            faqs_view()
        elif view[0] == "contact":
            contact_view()
        else:
            home_view()
    chatbot_interface()
    footer()

def main():
    st.set_page_config(page_title="Banaras Ghat Bot", page_icon="🚩", layout="wide")
    init_state()
    warm_up()
    render_ui()

if __name__ == "__main__":
    main()