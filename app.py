"""
🚀 TITAN V27 - NVIDIA AI EDITION [2-SO & 3-SO SUPPORT]
"""
import streamlit as st
import pandas as pd
from collections import Counter
from openai import OpenAI
import google.generativeai as genai
import json
import re

from config import NVIDIA_API_KEY, GEMINI_API_KEY, PAIR_RULES, AI_MODELS, THEME, LOTTERY_CONFIG
from utils import (
    extract_lottery_numbers, 
    check_win_2so5tinh, check_win_3so5tinh, 
    generate_combinations_from_7, calculate_win_examples,
    load_database, save_database, generate_fallback_prediction, format_for_ai
)

# ================= 🎨 CONFIG =================
st.set_page_config(page_title="TITAN V27 | 2-SO & 3-SO", page_icon="🎲", layout="wide")

# 💡 FIX: Gán biến theme để tránh lỗi quote
accent = THEME["accent"]
accent_sec = THEME["accent_secondary"]
bg_pri = THEME["bg_primary"]
bg_sec = THEME["bg_secondary"]
txt_pri = THEME["text_primary"]
txt_sec = THEME["text_secondary"]
danger = THEME["danger"]

custom_css = f"""
<style>
.stApp {{ background-color: {bg_pri}; color: {txt_pri}; font-family: 'Segoe UI', sans-serif; }}
.main-box {{ background: {bg_sec}; border: 2px solid {accent}; border-radius: 20px; padding: 30px; box-shadow: 0 0 25px rgba(118,185,0,0.3); }}
.big-num {{ font-size: 70px; font-weight: 900; color: {accent}; text-align: center; letter-spacing: 10px; text-shadow: 0 0 20px {accent}; font-family: monospace; }}
.lot-num {{ font-size: 35px; font-weight: 700; color: {accent_sec}; text-align: center; letter-spacing: 5px; font-family: monospace; }}
.status-bar {{ padding: 15px 25px; border-radius: 50px; text-align: center; font-weight: 800; margin: 20px 0; background: {accent}; color: #000; animation: pulse 2s infinite; }}
@keyframes pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(118,185,0,0.4); }} 70% {{ box-shadow: 0 0 0 15px rgba(118,185,0,0); }} }}
.combo-card {{ background: rgba(255,255,255,0.05); border-radius: 10px; padding: 15px; margin: 5px 0; border-left: 3px solid {accent_sec}; }}
.win-badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; }}
.win-yes {{ background: {accent}; color: #000; }}
.win-no {{ background: {danger}; color: #fff; }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ================= 🧠 INIT =================
if "db" not in st.session_state: st.session_state.db = load_database()
if "pred" not in st.session_state: st.session_state.pred = None
if "test_result" not in st.session_state: st.session_state.test_result = None

@st.cache_resource
def setup_ai():
    nv = OpenAI(base_url=AI_MODELS["nvidia"]["base_url"], api_key=NVIDIA_API_KEY)
    genai.configure(api_key=GEMINI_API_KEY)
    gm = genai.GenerativeModel(AI_MODELS["gemini"]["model"])
    return nv, gm

nv_ai, gm_ai = setup_ai()

# ================= 🎲 SIDEBAR =================
with st.sidebar:
    st.title("🎮 TITAN V27")
    st.image("https://api.dicebear.com/7.x/bottts/svg?seed=TITAN27", width=70)
    st.markdown("---")
    
    # 📜 Luật chơi
    with st.expander("📖 Luật chơi", expanded=True):
        st.markdown("""
        ### 🎯 2 số 5 tinh
        - Chọn **2 số** từ 0-9
        - Thắng nếu **cả 2 số** xuất hiện trong kết quả 5 chữ số
        - Ví dụ: Cược `[1,2]` + Kết quả `12121` → 🎉 THẮNG
        
        ### 🎯 3 số 5 tinh  
        - Chọn **3 số** từ 0-9
        - Thắng nếu **cả 3 số** xuất hiện trong kết quả 5 chữ số
        - Ví dụ: Cược `[1,2,6]` + Kết quả `12864` → 🎉 THẮNG
        
        > 💡 Chiến lược: AI chọn **7 số nóng** → Sinh **21 cặp 2so** + **35 tổ hợp 3so** → Bạn chọn đánh loại nào!
        """)
    
    # 🧪 Test tool
    st.markdown("### 🧪 Test nhanh")
    mode = st.radio("Chế độ", ["2 số", "3 số"], horizontal=True)
    c1, c2 = st.columns(2)
    with c1:
        bet_len = 2 if mode == "2 số" else 3
        test_bet = st.text_input(f"Số cược ({bet_len} số)", "12" if mode=="2 số" else "126", max_chars=bet_len)
    with c2:
        test_draw = st.text_input("Kết quả (5 số)", "12864", max_chars=5)
    
    if st.button("🔍 Kiểm tra", use_container_width=True):
        if len(test_bet) == bet_len and len(test_draw)==5 and test_bet.isdigit() and test_draw.isdigit():
            if mode == "2 số":
                win = check_win_2so5tinh(list(test_bet), test_draw)
            else:
                win = check_win_3so5tinh(list(test_bet), test_draw)
            st.session_state.test_result = {"bet":test_bet, "draw":test_draw, "win":win, "mode":mode}
    
    if st.session_state.test_result:
        r = st.session_state.test_result
        icon = "🎉" if r["win"] else "❌"
        badge_class = "win-yes" if r["win"] else "win-no"
        st.markdown(f'''
        <div class="combo-card">
        <strong>{icon} {r["mode"]}: {r["bet"]} vs {r["draw"]}</strong><br>
        <span class="win-badge {badge_class}">{"THẮNG" if r["win"] else "THUA"}</span>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    st.metric("📊 Database", f"{len(st.session_state.db)} kỳ")
    if st.button("🗑️ Reset DB", use_container_width=True):
        st.session_state.db = []
        st.session_state.pred = None
        save_database([])
        st.rerun()

# ================= 🖥️ MAIN =================
st.markdown(f'<h1 style="text-align:center;color:{accent};text-shadow:0 0 15px {accent}">🚀 TITAN V27 - 2SO & 3SO MODE</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="text-align:center;color:{txt_sec}">Chiến lược dàn 7 số → 21 cặp 2-tinh + 35 tổ hợp 3-tinh</p>', unsafe_allow_html=True)

# Input
col_in, col_info = st.columns([2.5, 1])
with col_in:
    raw = st.text_area("📡 DÁN DỮ LIỆU (mỗi dòng 1 số 5 chữ số):", height=100, placeholder="32457\n83465\n...")
with col_info:
    st.metric("📦 Tổng kỳ", len(st.session_state.db))
    st.metric("🎯 Kỳ cuối", st.session_state.db[-1] if st.session_state.db else "-")
    if st.button("⚡ CHỐT SỐ AI", use_container_width=True, type="primary"):
        clean = extract_lottery_numbers(raw)
        if clean:
            st.session_state.db.extend(clean)
            save_database(st.session_state.db)
            
            fallback = generate_fallback_prediction(st.session_state.db, PAIR_RULES)
            prompt = format_for_ai(st.session_state.db, PAIR_RULES)
            
            try:
                comp = nv_ai.chat.completions.create(
                    model=AI_MODELS["nvidia"]["model"],
                    messages=[{"role":"user","content":prompt}],
                    temperature=AI_MODELS["nvidia"]["temperature"],
                    response_format={"type":"json_object"}
                )
                st.session_state.pred = json.loads(comp.choices[0].message.content)
            except:
                try:
                    res = gm_ai.generate_content(prompt)
                    m = re.search(r'\{[\s\S]*\}', res.text)
                    st.session_state.pred = json.loads(m.group()) if m else fallback
                except:
                    st.warning("⚠️ AI lỗi → Dùng thuật toán dự phòng")
                    st.session_state.pred = fallback
            st.rerun()
        else:
            st.error("❌ Không tìm thấy số 5 chữ số hợp lệ!")

# ================= 📊 DISPLAY =================
if st.session_state.pred:
    p = st.session_state.pred
    is_go = p.get("adv","").upper()=="ĐÁNH"
    status_bg = accent if is_go else danger
    status_txt = "🔥 KHUYÊN ĐÁNH" if is_go else "⏸️ NÊN DỪNG"
    
    st.markdown(f'<div class="status-bar">{status_txt} | Tin cậy: {p.get("conf",0)}%</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-box">', unsafe_allow_html=True)
    
    # 🎯 Dàn 7 số gốc
    base7 = p.get("base_7", "0123456")
    st.markdown(f'''
    <div style="text-align:center;margin-bottom:20px;">
    <span style="color:{txt_sec};font-size:18px;">🎲 DÀN 7 SỐ NỀN (AI CHỌN)</span><br>
    <span style="font-size:50px;font-weight:bold;color:{accent};letter-spacing:8px;font-family:monospace;">{base7}</span>
    </div>
    ''', unsafe_allow_html=True)
    
    # Tabs: 2-so vs 3-so
    tab1, tab2, tab3 = st.tabs(["🎯 2 SỐ 5 TINH", "🎯 3 SỐ 5 TINH", "📋 COPY DÀN"])
    
    with tab1:
        st.markdown(f'<p style="color:{txt_sec};text-align:center">21 cặp từ dàn 7 số • Xác suất trúng cao hơn</p>', unsafe_allow_html=True)
        pairs = p.get("pairs_sample", [])
        if not pairs and "base_7" in p:
            combos = generate_combinations_from_7(p["base_7"])
            pairs = combos["pairs"]
        
        # Hiển thị dạng grid
        cols = st.columns(7)
        for i, pair in enumerate(pairs[:21]):
            with cols[i%7]:
                st.markdown(f'''
                <div class="combo-card" style="text-align:center;padding:10px;">
                <strong style="font-size:20px;color:{accent_sec};font-family:monospace;">{pair}</strong>
                </div>
                ''', unsafe_allow_html=True)
        
        st.info("💡 Mẹo: Đánh rải 5-10 cặp có cảm giác tốt, hoặc đánh all 21 cặp nếu vốn mạnh")
    
    with tab2:
        st.markdown(f'<p style="color:{txt_sec};text-align:center">35 tổ hợp từ dàn 7 số • Thưởng cao hơn</p>', unsafe_allow_html=True)
        triples = p.get("triples_sample", [])
        if not triples and "base_7" in p:
            combos = generate_combinations_from_7(p["base_7"])
            triples = combos["triples"]
        
        # Highlight main_3
        main3 = p.get("main_3", "")
        cols = st.columns(5)
        for i, triple in enumerate(triples[:35]):
            with cols[i%5]:
                is_main = triple == main3
                color = accent if is_main else txt_pri
                size = "22px" if is_main else "18px"
                st.markdown(f'''
                <div class="combo-card" style="text-align:center;padding:10px;border-left-color:{accent if is_main else accent_sec}">
                <strong style="font-size:{size};color:{color};font-family:monospace;">{triple}{" ⭐" if is_main else ""}</strong>
                </div>
                ''', unsafe_allow_html=True)
        
        if main3:
            st.success(f"🌟 Số chủ lực AI đề xuất: **{main3}**")
    
    with tab3:
        st.markdown("### 📋 Copy nhanh cho app cược")
        # Dàn 7 số
        st.text_input("🎲 Dàn 7 số nền:", base7, disabled=True)
        # All pairs
        if "base_7" in p:
            all_combos = generate_combinations_from_7(base7)
            st.text_area("🎯 21 cặp 2-số (mỗi dòng 1 cặp):", "\n".join(all_combos["pairs"]), height=150)
            st.text_area("🎯 35 tổ hợp 3-số (mỗi dòng 1 tổ hợp):", "\n".join(all_combos["triples"]), height=200)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 🧠 Logic + Test result
    st.markdown(f"🧠 **AI phân tích:** {p.get('logic', 'Đang xử lý...')}")
    
    with st.expander("🔍 Test với kết quả thực tế"):
        test_res = st.text_input("Nhập kết quả 5 số mới:", max_chars=5)
        if test_res and len(test_res)==5 and test_res.isdigit() and "base_7" in p:
            draw_set = set(test_res)
            base_digits = set(p["base_7"])
            matched = base_digits.intersection(draw_set)
            st.markdown(f'''
            <div class="combo-card">
            <strong>Kết quả:</strong> {test_res}<br>
            <strong>Số trong dàn 7 trùng:</strong> <span style="color:{accent};font-size:18px">{"".join(sorted(matched)) or "Không có"}</span><br>
            <strong>Tỷ lệ:</strong> {len(matched)}/7 số trúng
            </div>
            ''', unsafe_allow_html=True)
            
            # Auto-check main_3 if exists
            if p.get("main_3") and len(p["main_3])==3:
                is_win = check_win_3so5tinh(list(p["main_3"]), test_res)
                icon = "🎉" if is_win else "❌"
                msg = "THẮNG 3-TINH!" if is_win else "Trượt 3-tinh"
                st.success(f"{icon} Số chủ lực `{p['main_3']}` → **{msg}**") if is_win else st.warning(f"{icon} Số chủ lực `{p['main_3']}` → **{msg}**")

# ================= 📈 ANALYTICS =================
if st.session_state.db:
    with st.expander("📊 Thống kê 100 kỳ gần nhất"):
        freq = Counter("".join(st.session_state.db[-100:]))
        df = pd.DataFrame([{"Số":str(i),"Tần suất":freq.get(str(i),0)} for i in range(10)])
        st.bar_chart(df.set_index("Số"), color=accent_sec)
        
        ch, cc = st.columns(2)
        with ch:
            st.markdown("### 🔥 Top 3 nóng")
            for _,r in df.nlargest(3,"Tần suất").iterrows():
                st.markdown(f"`{r['Số']}` → {r['Tần suất']} lần")
        with cc:
            st.markdown("### ❄️ Top 3 lạnh")
            for _,r in df.nsmallest(3,"Tần suất").iterrows():
                st.markdown(f"`{r['Số']}` → {r['Tần suất']} lần")

# ================= ⚙️ DEBUG =================
with st.expander("⚙️ Dev Tools"):
    if st.button("🧪 Test examples"):
        for r in calculate_win_examples():
            icon = "✅" if r["correct"] else "❌"
            st.text(f"{icon} {r['type'].upper()}|{r['bet']} vs {r['draw']} → {'WIN' if r['predicted'] else 'LOSE'}")
    if st.button("📋 Xem 10 kỳ cuối"):
        for i,n in enumerate(reversed(st.session_state.db[-10:]),1):
            st.text(f"{i}. {n}")

# ================= 🦶 FOOTER =================
st.markdown("---")
st.markdown(f'''
<div style="text-align:center;color:{txt_sec};font-size:13px;">
🔐 TITAN V27 • Micro-SaaS Edition • 
<a href="https://github.com/yourusername/titan-v27" style="color:{accent_sec};text-decoration:none">GitHub</a> • 
<span style="color:{accent}">v2.8.0 [2SO+3SO]</span>
</div>
''', unsafe_allow_html=True)