"""
🚀 TITAN V27 - NVIDIA AI EDITION [SINGLE FILE - NO IMPORTS]
Version: 2.9.0-STANDALONE
"""
import streamlit as st
import pandas as pd
from collections import Counter
from openai import OpenAI
import google.generativeai as genai
import json
import re
import itertools
from pathlib import Path

# ================= ⚙️ CONFIGURATION =================
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
DB_FILE = Path("titan_v27_permanent.json")

PAIR_RULES = [
    "178", "034", "458", "578", "019", "679", "235", "456", "124", "245", 
    "247", "248", "246", "340", "349", "348", "015", "236", "028", "026", 
    "047", "046", "056", "136", "138", "378"
]

AI_MODELS = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "meta/llama-3.1-70b-instruct",
        "temperature": 0.2
    },
    "gemini": {
        "model": "gemini-1.5-flash",
        "temperature": 0.3
    }
}

THEME = {
    "bg_primary": "#05050a",
    "bg_secondary": "#0d1117",
    "accent": "#76b900",
    "accent_secondary": "#00d4ff",
    "danger": "#ff4444",
    "text_primary": "#ffffff",
    "text_secondary": "#888888"
}

# ================= 🎨 CSS =================
accent_color = THEME["accent"]
accent_sec = THEME["accent_secondary"]
bg_primary = THEME["bg_primary"]
bg_secondary = THEME["bg_secondary"]
text_primary = THEME["text_primary"]
text_secondary = THEME["text_secondary"]
danger_color = THEME["danger"]

css_html = f"""
<style>
.stApp {{ background-color: {bg_primary}; color: {text_primary}; font-family: 'Segoe UI', sans-serif; }}
.main-box {{ background: {bg_secondary}; border: 2px solid {accent_color}; border-radius: 20px; padding: 30px; box-shadow: 0 0 25px rgba(118,185,0,0.3); }}
.big-num {{ font-size: 70px; font-weight: 900; color: {accent_color}; text-align: center; letter-spacing: 10px; text-shadow: 0 0 20px {accent_color}; font-family: monospace; }}
.status-bar {{ padding: 15px 25px; border-radius: 50px; text-align: center; font-weight: 800; margin: 20px 0; background: {accent_color}; color: #000; }}
.combo-card {{ background: rgba(255,255,255,0.05); border-radius: 10px; padding: 15px; margin: 5px 0; border-left: 3px solid {accent_sec}; }}
.win-badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; }}
.win-yes {{ background: {accent_color}; color: #000; }}
.win-no {{ background: {danger_color}; color: #fff; }}
.stButton>button {{ background: linear-gradient(135deg, {accent_color}, #5a9e00); color: #000; font-weight: 700; border: none; border-radius: 12px; padding: 12px 30px; }}
</style>
"""

# ================= 🔧 UTILITY FUNCTIONS =================
def extract_lottery_numbers(text):
    """Extract 5-digit numbers from text"""
    if not text:
        return []
    return re.findall(r"\b\d{5}\b", text)

def check_win_2so(bet_numbers, draw_number):
    """Check win for 2-number bet"""
    if len(bet_numbers) != 2 or len(draw_number) != 5:
        return False
    draw_digits = set(draw_number)
    return all(digit in draw_digits for digit in bet_numbers)

def check_win_3so(bet_numbers, draw_number):
    """Check win for 3-number bet"""
    if len(bet_numbers) != 3 or len(draw_number) != 5:
        return False
    draw_digits = set(draw_number)
    return all(digit in draw_digits for digit in bet_numbers)

def generate_combos(numbers_7):
    """Generate 2-so and 3-so combinations from 7 numbers"""
    digits = list(set(numbers_7))
    if len(digits) < 7:
        remaining = [str(i) for i in range(10) if str(i) not in digits]
        digits += remaining[:7 - len(digits)]
    digits = digits[:7]
    
    pairs = ["".join(p) for p in itertools.combinations(sorted(digits), 2)]
    triples = ["".join(t) for t in itertools.combinations(sorted(digits), 3)]
    
    return {
        "base_7": "".join(sorted(digits)),
        "pairs": pairs,
        "triples": triples
    }

def load_db():
    """Load database from file"""
    try:
        if DB_FILE.exists():
            with open(DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except:
        pass
    return []

def save_db(data, max_entries=5000):
    """Save database to file"""
    try:
        unique_data = list(dict.fromkeys(data))[-max_entries:]
        DB_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(unique_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Save error: {e}")

def calc_scores(db, pair_rules, last_draw):
    """Calculate frequency scores"""
    if not db:
        return {str(i): 0 for i in range(10)}
    
    all_digits = "".join(db[-60:]) if db else ""
    scores = {str(i): all_digits.count(str(i)) for i in range(10)}
    
    if last_draw:
        for rule in pair_rules:
            if sum(1 for d in last_draw if d in rule) >= 2:
                for digit in rule:
                    scores[digit] = scores.get(digit, 0) + 15
    return scores

def fallback_prediction(db, pair_rules):
    """Fallback prediction when AI fails"""
    if not db:
        return {
            "base_7": "0123456",
            "main_3": "012",
            "pairs_sample": ["01", "23", "45"],
            "triples_sample": ["012", "234", "456"],
            "adv": "DỪNG",
            "logic": "Chưa đủ dữ liệu",
            "conf": 50
        }
    
    try:
        last_draw = db[-1] if db else ""
        scores = calc_scores(db, pair_rules, last_draw)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_7 = [x[0] for x in sorted_scores[:7]]
        base_7 = "".join(sorted(top_7))
        combos = generate_combos(base_7)
        
        return {
            "base_7": base_7,
            "main_3": combos["triples"][0] if combos["triples"] else "012",
            "pairs_sample": combos["pairs"][:5],
            "triples_sample": combos["triples"][:5],
            "adv": "ĐÁNH",
            "logic": "Phân tích tần suất + pair rules",
            "conf": 85
        }
    except:
        return {
            "base_7": "0123456",
            "main_3": "012",
            "pairs_sample": ["01", "23", "45"],
            "triples_sample": ["012", "234", "456"],
            "adv": "DỪNG",
            "logic": "Lỗi tính toán",
            "conf": 50
        }

def format_prompt(db, pair_rules, last_n=50):
    """Format prompt for AI"""
    recent = db[-last_n:] if len(db) >= last_n else db
    return f"""
[DATA] {recent}
[PAIR_RULES] {pair_rules}
[TASK] Chọn ra 7 số có xác suất cao nhất.
[OUTPUT] JSON: {{"base_7":"7 digits","main_3":"3 digits","pairs_sample":["12","34"],"triples_sample":["123","345"],"adv":"ĐÁNH/DỪNG","logic":"explanation","conf":85}}
"""

# ================= 🎨 PAGE CONFIG =================
st.set_page_config(page_title="TITAN V27", page_icon="🎲", layout="wide")
st.markdown(css_html, unsafe_allow_html=True)

# ================= 🧠 INIT SESSION =================
if "db" not in st.session_state:
    st.session_state.db = load_db()
if "pred" not in st.session_state:
    st.session_state.pred = None
if "test_result" not in st.session_state:
    st.session_state.test_result = None

# ================= 🤖 SETUP AI =================
@st.cache_resource
def setup_ai():
    try:
        nv = OpenAI(base_url=AI_MODELS["nvidia"]["base_url"], api_key=NVIDIA_API_KEY)
        genai.configure(api_key=GEMINI_API_KEY)
        gm = genai.GenerativeModel(AI_MODELS["gemini"]["model"])
        return nv, gm
    except:
        return None, None

nv_ai, gm_ai = setup_ai()

# ================= 🎲 SIDEBAR =================
with st.sidebar:
    st.title("🎮 TITAN V27")
    st.image("https://api.dicebear.com/7.x/bottts/svg?seed=TITAN27", width=70)
    st.markdown("---")
    
    with st.expander("📖 Luật chơi", expanded=True):
        st.markdown("""
        **2 số 5 tinh**: Chọn 2 số, thắng nếu cả 2 xuất hiện trong 5 chữ số kết quả  
        **3 số 5 tinh**: Chọn 3 số, thắng nếu cả 3 xuất hiện trong 5 chữ số kết quả  
        """)
    
    st.markdown("### 🧪 Test")
    mode = st.radio("Chế độ", ["2 số", "3 số"], horizontal=True)
    c1, c2 = st.columns(2)
    with c1:
        bet_len = 2 if mode == "2 số" else 3
        test_bet = st.text_input("Số cược", "12" if mode=="2 số" else "126", max_chars=bet_len)
    with c2:
        test_draw = st.text_input("Kết quả", "12864", max_chars=5)
    
    if st.button("🔍 Kiểm tra", use_container_width=True):
        if len(test_bet) == bet_len and len(test_draw) == 5 and test_bet.isdigit() and test_draw.isdigit():
            if mode == "2 số":
                win = check_win_2so(list(test_bet), test_draw)
            else:
                win = check_win_3so(list(test_bet), test_draw)
            st.session_state.test_result = {"bet": test_bet, "draw": test_draw, "win": win, "mode": mode}
    
    if st.session_state.test_result:
        r = st.session_state.test_result
        icon = "🎉" if r["win"] else "❌"
        status = "THẮNG" if r["win"] else "THUA"
        badge = "win-yes" if r["win"] else "win-no"
        st.markdown(f'<div class="combo-card"><strong>{icon} {r["mode"]}: {r["bet"]} vs {r["draw"]}</strong><br><span class="win-badge {badge}">{status}</span></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.metric("📊 Database", f"{len(st.session_state.db)} kỳ")
    
    if st.button("🗑️ Reset", use_container_width=True):
        st.session_state.db = []
        st.session_state.pred = None
        save_db([])
        st.rerun()

# ================= 🖥️ MAIN =================
st.markdown(f'<h1 style="text-align:center;color:{accent_color}">🚀 TITAN V27</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="text-align:center;color:{text_secondary}">Dàn 7 số → 21 cặp 2-tinh + 35 tổ hợp 3-tinh</p>', unsafe_allow_html=True)

col_input, col_info = st.columns([2.5, 1])
with col_input:
    raw_input = st.text_area("📡 DÁN DỮ LIỆU:", height=100, placeholder="32457\n83465\n...")

with col_info:
    st.metric("📦 Tổng kỳ", len(st.session_state.db))
    last = st.session_state.db[-1] if st.session_state.db else "-"
    st.metric("🎯 Kỳ cuối", last)
    
    if st.button("⚡ CHỐT SỐ", use_container_width=True, type="primary"):
        clean_data = extract_lottery_numbers(raw_input)
        if clean_data:
            st.session_state.db.extend(clean_data)
            save_db(st.session_state.db)
            
            fallback = fallback_prediction(st.session_state.db, PAIR_RULES)
            prompt = format_prompt(st.session_state.db, PAIR_RULES)
            
            pred_result = fallback
            
            if nv_ai:
                try:
                    comp = nv_ai.chat.completions.create(
                        model=AI_MODELS["nvidia"]["model"],
                        messages=[{"role": "user", "content": prompt}],
                        temperature=AI_MODELS["nvidia"]["temperature"],
                        response_format={"type": "json_object"}
                    )
                    pred_result = json.loads(comp.choices[0].message.content)
                except:
                    if gm_ai:
                        try:
                            res = gm_ai.generate_content(prompt)
                            m = re.search(r'\{[\s\S]*\}', res.text)
                            if m:
                                pred_result = json.loads(m.group())
                        except:
                            pass
                    st.warning("⚠️ Dùng thuật toán dự phòng")
            
            st.session_state.pred = pred_result
            st.rerun()
        else:
            st.error("❌ Không tìm thấy số 5 chữ số!")

# ================= 📊 DISPLAY =================
if st.session_state.pred:
    p = st.session_state.pred
    is_go = p.get("adv", "").upper() == "ĐÁNH"
    status_bg = accent_color if is_go else danger_color
    status_txt = "🔥 KHUYÊN ĐÁNH" if is_go else "⏸️ NÊN DỪNG"
    
    st.markdown(f'<div class="status-bar">{status_txt} | Tin cậy: {p.get("conf",0)}%</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-box">', unsafe_allow_html=True)
    
    base7 = p.get("base_7", "0123456")
    st.markdown(f'<div style="text-align:center;margin-bottom:20px;"><span style="color:{text_secondary}">🎲 DÀN 7 SỐ</span><br><span style="font-size:50px;font-weight:bold;color:{accent_color};font-family:monospace;">{base7}</span></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 2 SỐ", "🎯 3 SỐ", "📋 COPY"])
    
    with tab1:
        pairs = p.get("pairs_sample", [])
        if not pairs and "base_7" in p:
            combos = generate_combos(p["base_7"])
            pairs = combos["pairs"]
        
        cols = st.columns(7)
        for i, pair in enumerate(pairs[:21]):
            with cols[i % 7]:
                st.markdown(f'<div class="combo-card" style="text-align:center;padding:10px;"><strong style="font-size:20px;color:{accent_sec};font-family:monospace;">{pair}</strong></div>', unsafe_allow_html=True)
    
    with tab2:
        triples = p.get("triples_sample", [])
        main3 = p.get("main_3", "")
        if not triples and "base_7" in p:
            combos = generate_combos(p["base_7"])
            triples = combos["triples"]
        
        cols = st.columns(5)
        for i, triple in enumerate(triples[:35]):
            with cols[i % 5]:
                is_main = triple == main3
                color = accent_color if is_main else text_primary
                size = "22px" if is_main else "18px"
                star = " ⭐" if is_main else ""
                st.markdown(f'<div class="combo-card" style="text-align:center;padding:10px;"><strong style="font-size:{size};color:{color};font-family:monospace;">{triple}{star}</strong></div>', unsafe_allow_html=True)
        
        if main3:
            st.success(f"🌟 Số chủ lực: **{main3}**")
    
    with tab3:
        st.text_input("🎲 Dàn 7 số:", base7, disabled=True)
        if "base_7" in p:
            all_combos = generate_combos(base7)
            st.text_area("🎯 21 cặp 2-số:", "\n".join(all_combos["pairs"]), height=150)
            st.text_area("🎯 35 tổ hợp 3-số:", "\n".join(all_combos["triples"]), height=200)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(f"🧠 **AI:** {p.get('logic', '...')}")

# ================= 📈 ANALYTICS =================
if st.session_state.db:
    with st.expander("📊 Thống kê 100 kỳ"):
        freq = Counter("".join(st.session_state.db[-100:]))
        df = pd.DataFrame([{"Số": str(i), "Tần suất": freq.get(str(i), 0)} for i in range(10)])
        st.bar_chart(df.set_index("Số"), color=accent_sec)

# ================= 🦶 FOOTER =================
st.markdown("---")
st.markdown(f'<div style="text-align:center;color:{text_secondary};font-size:13px;">🔐 TITAN V27 v2.9.0-STANDALONE</div>', unsafe_allow_html=True)