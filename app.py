"""
🚀 TITAN V27 - NVIDIA AI EDITION [2-SO & 3-SO SUPPORT]
Version: 2.8.1-FIXED
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

# ================= 🎨 CONFIG PAGE =================
st.set_page_config(
    page_title="TITAN V27 | 2-SO & 3-SO",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 🎨 CUSTOM CSS =================
# 💡 FIX: Gán tất cả theme ra biến, không dùng trực tiếp trong f-string phức tạp
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
.lot-num {{ font-size: 35px; font-weight: 700; color: {accent_sec}; text-align: center; letter-spacing: 5px; font-family: monospace; }}
.status-bar {{ padding: 15px 25px; border-radius: 50px; text-align: center; font-weight: 800; margin: 20px 0; background: {accent_color}; color: #000; }}
.combo-card {{ background: rgba(255,255,255,0.05); border-radius: 10px; padding: 15px; margin: 5px 0; border-left: 3px solid {accent_sec}; }}
.win-badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; }}
.win-yes {{ background: {accent_color}; color: #000; }}
.win-no {{ background: {danger_color}; color: #fff; }}
.stButton>button {{ background: linear-gradient(135deg, {accent_color}, #5a9e00); color: #000; font-weight: 700; border: none; border-radius: 12px; padding: 12px 30px; }}
</style>
"""
st.markdown(css_html, unsafe_allow_html=True)

# ================= 🧠 INIT SESSION =================
if "db" not in st.session_state:
    st.session_state.db = load_database()
if "pred" not in st.session_state:
    st.session_state.pred = None
if "test_result" not in st.session_state:
    st.session_state.test_result = None

# ================= 🤖 SETUP AI CLIENTS =================
@st.cache_resource
def setup_ai_clients():
    nv_client = OpenAI(
        base_url=AI_MODELS["nvidia"]["base_url"], 
        api_key=NVIDIA_API_KEY
    )
    genai.configure(api_key=GEMINI_API_KEY)
    gm_model = genai.GenerativeModel(AI_MODELS["gemini"]["model"])
    return nv_client, gm_model

nv_ai, gm_ai = setup_ai_clients()

# ================= 🎲 SIDEBAR =================
with st.sidebar:
    st.title("🎮 TITAN V27")
    st.image("https://api.dicebear.com/7.x/bottts/svg?seed=TITAN27", width=70)
    st.markdown("---")
    
    with st.expander("📖 Luật chơi", expanded=True):
        st.markdown("""
        ### 🎯 2 số 5 tinh
        - Chọn **2 số** từ 0-9
        - Thắng nếu **cả 2 số** xuất hiện trong kết quả 5 chữ số
        - Ví dụ: `[1,2]` + `12121` → 🎉 THẮNG
        
        ### 🎯 3 số 5 tinh  
        - Chọn **3 số** từ 0-9
        - Thắng nếu **cả 3 số** xuất hiện trong kết quả 5 chữ số
        - Ví dụ: `[1,2,6]` + `12864` → 🎉 THẮNG
        """)
    
    st.markdown("### 🧪 Test nhanh")
    mode = st.radio("Chế độ", ["2 số", "3 số"], horizontal=True)
    c1, c2 = st.columns(2)
    with c1:
        bet_len = 2 if mode == "2 số" else 3
        default_bet = "12" if mode == "2 số" else "126"
        test_bet = st.text_input(f"Số cược ({bet_len} số)", default_bet, max_chars=bet_len)
    with c2:
        test_draw = st.text_input("Kết quả (5 số)", "12864", max_chars=5)
    
    if st.button("🔍 Kiểm tra", use_container_width=True):
        if len(test_bet) == bet_len and len(test_draw) == 5 and test_bet.isdigit() and test_draw.isdigit():
            if mode == "2 số":
                win = check_win_2so5tinh(list(test_bet), test_draw)
            else:
                win = check_win_3so5tinh(list(test_bet), test_draw)
            st.session_state.test_result = {
                "bet": test_bet, 
                "draw": test_draw, 
                "win": win, 
                "mode": mode
            }
    
    if st.session_state.test_result:
        r = st.session_state.test_result
        icon = "🎉" if r["win"] else "❌"
        badge_class = "win-yes" if r["win"] else "win-no"
        status_text = "THẮNG" if r["win"] else "THUA"
        
        result_html = f"""
        <div class="combo-card">
        <strong>{icon} {r["mode"]}: {r["bet"]} vs {r["draw"]}</strong><br>
        <span class="win-badge {badge_class}">{status_text}</span>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)
    
    st.markdown("---")
    st.metric("📊 Database", f"{len(st.session_state.db)} kỳ")
    
    if st.button("🗑️ Reset DB", use_container_width=True):
        st.session_state.db = []
        st.session_state.pred = None
        save_database([])
        st.rerun()

# ================= 🖥️ MAIN INTERFACE =================
# 💡 FIX: Dùng single quotes cho f-string đơn giản
st.markdown(
    f'<h1 style="text-align:center;color:{accent_color};text-shadow:0 0 15px {accent_color}">🚀 TITAN V27 - 2SO & 3SO MODE</h1>', 
    unsafe_allow_html=True
)
st.markdown(
    f'<p style="text-align:center;color:{text_secondary}">Chiến lược dàn 7 số → 21 cặp 2-tinh + 35 tổ hợp 3-tinh</p>', 
    unsafe_allow_html=True
)

# 📥 Input Section
col_input, col_info = st.columns([2.5, 1])
with col_input:
    raw_input = st.text_area(
        "📡 DÁN DỮ LIỆU (mỗi dòng 1 số 5 chữ số):", 
        height=100, 
        placeholder="32457\n83465\n19283\n..."
    )
    
with col_info:
    st.metric("📦 Tổng kỳ", len(st.session_state.db))
    last_draw = st.session_state.db[-1] if st.session_state.db else "-"
    st.metric("🎯 Kỳ cuối", last_draw)
    
    if st.button("⚡ CHỐT SỐ AI", use_container_width=True, type="primary"):
        clean_data = extract_lottery_numbers(raw_input)
        if clean_data:
            st.session_state.db.extend(clean_data)
            save_database(st.session_state.db)
            
            fallback_pred = generate_fallback_prediction(st.session_state.db, PAIR_RULES)
            prompt = format_for_ai(st.session_state.db, PAIR_RULES)
            
            try:
                completion = nv_ai.chat.completions.create(
                    model=AI_MODELS["nvidia"]["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=AI_MODELS["nvidia"]["temperature"],
                    response_format={"type": "json_object"}
                )
                st.session_state.pred = json.loads(completion.choices[0].message.content)
            except Exception as e1:
                try:
                    res = gm_ai.generate_content(prompt)
                    json_match = re.search(r'\{[\s\S]*\}', res.text)
                    if json_match:
                        st.session_state.pred = json.loads(json_match.group())
                    else:
                        st.session_state.pred = fallback_pred
                except Exception as e2:
                    st.warning("⚠️ AI lỗi → Dùng thuật toán dự phòng")
                    st.session_state.pred = fallback_pred
            
            st.rerun()
        else:
            st.error("❌ Không tìm thấy số 5 chữ số hợp lệ!")

# ================= 📊 DISPLAY PREDICTION =================
if st.session_state.pred:
    p = st.session_state.pred
    is_go = p.get("adv", "").upper() == "ĐÁNH"
    status_bg = accent_color if is_go else danger_color
    status_txt = "🔥 KHUYÊN ĐÁNH" if is_go else "⏸️ NÊN DỪNG"
    confidence = p.get("conf", 0)
    
    status_html = f'<div class="status-bar">{status_txt} | Tin cậy: {confidence}%</div>'
    st.markdown(status_html, unsafe_allow_html=True)
    st.markdown('<div class="main-box">', unsafe_allow_html=True)
    
    # 🎯 Dàn 7 số gốc
    base7 = p.get("base_7", "0123456")
    header_html = f"""
    <div style="text-align:center;margin-bottom:20px;">
    <span style="color:{text_secondary};font-size:18px;">🎲 DÀN 7 SỐ NỀN (AI CHỌN)</span><br>
    <span style="font-size:50px;font-weight:bold;color:{accent_color};letter-spacing:8px;font-family:monospace;">{base7}</span>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 2 SỐ 5 TINH", "🎯 3 SỐ 5 TINH", "📋 COPY DÀN"])
    
    with tab1:
        st.markdown(f'<p style="color:{text_secondary};text-align:center">21 cặp từ dàn 7 số • Xác suất trúng cao hơn</p>', unsafe_allow_html=True)
        pairs = p.get("pairs_sample", [])
        if not pairs and "base_7" in p:
            combos = generate_combinations_from_7(p["base_7"])
            pairs = combos["pairs"]
        
        cols = st.columns(7)
        for i, pair in enumerate(pairs[:21]):
            with cols[i % 7]:
                pair_html = f"""
                <div class="combo-card" style="text-align:center;padding:10px;">
                <strong style="font-size:20px;color:{accent_sec};font-family:monospace;">{pair}</strong>
                </div>
                """
                st.markdown(pair_html, unsafe_allow_html=True)
        
        st.info("💡 Mẹo: Đánh rải 5-10 cặp có cảm giác tốt")
    
    with tab2:
        st.markdown(f'<p style="color:{text_secondary};text-align:center">35 tổ hợp từ dàn 7 số • Thưởng cao hơn</p>', unsafe_allow_html=True)
        triples = p.get("triples_sample", [])
        main3 = p.get("main_3", "")
        
        if not triples and "base_7" in p:
            combos = generate_combinations_from_7(p["base_7"])
            triples = combos["triples"]
        
        cols = st.columns(5)
        for i, triple in enumerate(triples[:35]):
            with cols[i % 5]:
                is_main = triple == main3
                color = accent_color if is_main else text_primary
                size = "22px" if is_main else "18px"
                star = " ⭐" if is_main else ""
                
                triple_html = f"""
                <div class="combo-card" style="text-align:center;padding:10px;border-left-color:{accent_color if is_main else accent_sec}">
                <strong style="font-size:{size};color:{color};font-family:monospace;">{triple}{star}</strong>
                </div>
                """
                st.markdown(triple_html, unsafe_allow_html=True)
        
        if main3:
            st.success(f"🌟 Số chủ lực AI đề xuất: **{main3}**")
    
    with tab3:
        st.markdown("### 📋 Copy nhanh cho app cược")
        st.text_input("🎲 Dàn 7 số nền:", base7, disabled=True)
        
        if "base_7" in p:
            all_combos = generate_combinations_from_7(base7)
            st.text_area(
                "🎯 21 cặp 2-số (mỗi dòng 1 cặp):", 
                "\n".join(all_combos["pairs"]), 
                height=150
            )
            st.text_area(
                "🎯 35 tổ hợp 3-số (mỗi dòng 1 tổ hợp):", 
                "\n".join(all_combos["triples"]), 
                height=200
            )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 🧠 Logic
    logic_text = p.get("logic", "Đang xử lý...")
    st.markdown(f"🧠 **AI phân tích:** {logic_text}")
    
    # ✅ Test với kết quả thực tế
    with st.expander("🔍 Test với kết quả thực tế"):
        test_res_input = st.text_input("Nhập kết quả 5 số mới:", max_chars=5)
        
        if test_res_input and len(test_res_input) == 5 and test_res_input.isdigit():
            if "base_7" in p:
                draw_set = set(test_res_input)
                base_digits = set(p["base_7"])
                matched = base_digits.intersection(draw_set)
                matched_str = "".join(sorted(matched)) if matched else "Không có"
                
                test_html = f"""
                <div class="combo-card">
                <strong>Kết quả:</strong> {test_res_input}<br>
                <strong>Số trong dàn 7 trùng:</strong> <span style="color:{accent_color};font-size:18px">{matched_str}</span><br>
                <strong>Tỷ lệ:</strong> {len(matched)}/7 số trúng
                </div>
                """
                st.markdown(test_html, unsafe_allow_html=True)
                
                # 💡 FIX: Kiểm tra main_3 an toàn, không gây lỗi syntax
                main_3_value = p.get("main_3", "")
                if main_3_value and len(main_3_value) == 3:
                    is_win = check_win_3so5tinh(list(main_3_value), test_res_input)
                    icon = "🎉" if is_win else "❌"
                    msg = "THẮNG 3-TINH!" if is_win else "Trượt 3-tinh"
                    
                    if is_win:
                        st.success(f"{icon} Số chủ lực `{main_3_value}` → **{msg}**")
                    else:
                        st.warning(f"{icon} Số chủ lực `{main_3_value}` → **{msg}**")

# ================= 📈 ANALYTICS =================
if st.session_state.db:
    with st.expander("📊 Thống kê 100 kỳ gần nhất"):
        freq_data = Counter("".join(st.session_state.db[-100:]))
        df_freq = pd.DataFrame([
            {"Số": str(i), "Tần suất": freq_data.get(str(i), 0)} 
            for i in range(10)
        ])
        st.bar_chart(df_freq.set_index("Số"), color=accent_sec)
        
        col_hot, col_cold = st.columns(2)
        with col_hot:
            st.markdown("### 🔥 Top 3 nóng")
            top_hot = df_freq.nlargest(3, "Tần suất")
            for _, row in top_hot.iterrows():
                st.markdown(f"`{row['Số']}` → {row['Tần suất']} lần")
        with col_cold:
            st.markdown("### ❄️ Top 3 lạnh")
            top_cold = df_freq.nsmallest(3, "Tần suất")
            for _, row in top_cold.iterrows():
                st.markdown(f"`{row['Số']}` → {row['Tần suất']} lần")

# ================= ⚙️ DEBUG =================
with st.expander("⚙️ Dev Tools"):
    if st.button("🧪 Test examples"):
        results = calculate_win_examples()
        for r in results:
            icon = "✅" if r["correct"] else "❌"
            result_type = "WIN" if r["predicted"] else "LOSE"
            st.text(f"{icon} {r['type'].upper()} | {r['bet']} vs {r['draw']} → {result_type}")
    
    if st.button("📋 Xem 10 kỳ cuối"):
        for i, num in enumerate(reversed(st.session_state.db[-10:]), 1):
            st.text(f"{i}. {num}")
    
    debug_info = {
        "db_entries": len(st.session_state.db),
        "pair_rules_count": len(PAIR_RULES),
        "ai_models": list(AI_MODELS.keys())
    }
    st.json({"config": debug_info})

# ================= 🦶 FOOTER =================
st.markdown("---")
footer_html = f"""
<div style="text-align:center;color:{text_secondary};font-size:13px;">
🔐 TITAN V27 • Micro-SaaS Edition • 
<a href="https://github.com/yourusername/titan-v27" style="color:{accent_sec};text-decoration:none">GitHub</a> • 
<span style="color:{accent_color}">v2.8.1-FIXED</span>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)