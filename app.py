"""
🚀 TITAN V27 - NVIDIA AI EDITION
Lottery Prediction App with Streamlit + AI Integration
[FIXED VERSION - No quote escaping issues]
"""
import streamlit as st
import pandas as pd
from collections import Counter
from openai import OpenAI
import google.generativeai as genai
import json
import re

from config import (
    NVIDIA_API_KEY, GEMINI_API_KEY, PAIR_RULES, 
    AI_MODELS, THEME, LOTTERY_CONFIG
)
from utils import (
    extract_lottery_numbers, check_win_3so5tinh, calculate_win_examples,
    load_database, save_database, generate_fallback_prediction, format_for_ai
)

# ================= 🎨 CONFIG PAGE =================
st.set_page_config(
    page_title="TITAN V27 | NVIDIA AI PRO",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 🎨 CUSTOM CSS =================
# 💡 FIX: Dùng single quotes cho f-string, hoặc gán CSS ra biến riêng
accent_color = THEME["accent"]
accent_secondary = THEME["accent_secondary"]
bg_primary = THEME["bg_primary"]
bg_secondary = THEME["bg_secondary"]
text_primary = THEME["text_primary"]
text_secondary = THEME["text_secondary"]
danger_color = THEME["danger"]

custom_css = f"""
    <style>
    .stApp {{ 
        background-color: {bg_primary}; 
        color: {text_primary}; 
        font-family: 'Segoe UI', system-ui, sans-serif;
    }}
    .main-box {{ 
        background: {bg_secondary}; 
        border: 2px solid {accent_color}; 
        border-radius: 20px; 
        padding: 30px; 
        box-shadow: 0 0 25px rgba(118,185,0,0.3);
        margin: 10px 0;
    }}
    .big-num {{ 
        font-size: 90px; 
        font-weight: 900; 
        color: {accent_color}; 
        text-align: center; 
        letter-spacing: 15px; 
        text-shadow: 0 0 20px {accent_color};
        font-family: 'Courier New', monospace;
    }}
    .lot-num {{ 
        font-size: 45px; 
        font-weight: 700; 
        color: {accent_secondary}; 
        text-align: center; 
        letter-spacing: 10px;
        font-family: 'Courier New', monospace;
    }}
    .status-bar {{ 
        padding: 15px 25px; 
        border-radius: 50px; 
        text-align: center; 
        font-weight: 800; 
        margin: 20px 0; 
        text-transform: uppercase; 
        letter-spacing: 1px;
        animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(118,185,0,0.4); }}
        70% {{ box-shadow: 0 0 0 15px rgba(118,185,0,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(118,185,0,0); }}
    }}
    .stButton>button {{
        background: linear-gradient(135deg, {accent_color}, #5a9e00);
        color: #000;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        transition: all 0.3s;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(118,185,0,0.4);
    }}
    .win-check {{ 
        background: rgba(0,212,255,0.1); 
        border-left: 4px solid {accent_secondary}; 
        padding: 15px; 
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
    }}
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ================= 🧠 INIT SESSION & ENGINES =================
if "db" not in st.session_state:
    st.session_state.db = load_database()
if "pred" not in st.session_state:
    st.session_state.pred = None
if "win_result" not in st.session_state:
    st.session_state.win_result = None

# Setup AI Clients
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

# ================= 🎲 SIDEBAR: LUẬT CHƠI & TOOL =================
with st.sidebar:
    st.image("https://api.dicebear.com/7.x/bottts/svg?seed=TITAN27", width=80)
    st.title("🎮 TITAN V27")
    st.markdown("---")
    
    # 📜 Luật chơi 3 số 5 tinh
    with st.expander("📖 Luật chơi 3 số 5 tinh", expanded=True):
        st.markdown("""
        ✅ **Cách cược**: Chọn 3 số từ `0-9`  
        ✅ **Điều kiện thắng**: 3 số bạn chọn phải **xuất hiện** trong kết quả 5 chữ số (bất kỳ vị trí nào)  
        ✅ **Thứ tự**: Không quan trọng  
        ✅ **Trùng lặp**: Dù số xuất hiện bao nhiêu lần, vẫn chỉ tính 1 lần  
        
        **Ví dụ**:  
        • Cược `[1,2,6]` + Kết quả `12864` → 🎉 **THẮNG**  
        • Cược `[1,3,6]` + Kết quả `12662` → ❌ **THUA**
        """)
    
    # 🧪 Test nhanh logic trúng thưởng
    st.markdown("### 🧪 Test Logic")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        test_bet = st.text_input("Số cược (3 số)", "126", max_chars=3)
    with col_t2:
        test_draw = st.text_input("Kết quả (5 số)", "12864", max_chars=5)
    
    if st.button("🔍 Kiểm tra", use_container_width=True):
        if len(test_bet) == 3 and len(test_draw) == 5 and test_bet.isdigit() and test_draw.isdigit():
            is_win = check_win_3so5tinh(list(test_bet), test_draw)
            st.session_state.win_result = {
                "bet": test_bet,
                "draw": test_draw,
                "win": is_win
            }
        else:
            st.error("Nhập đúng: 3 số cược + 5 số kết quả")
    
    if st.session_state.win_result:
        res = st.session_state.win_result
        icon = "🎉" if res["win"] else "❌"
        status = "THẮNG" if res["win"] else "THUA"
        color = accent_color if res["win"] else danger_color
        st.markdown(f"""
        <div class="win-check">
        <strong>{icon} {res['bet']} vs {res['draw']}</strong><br>
        Kết quả: <span style="color:{color}; font-weight:bold">{status}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption(f"💾 Database: **{len(st.session_state.db)} kỳ**")
    if st.button("🗑️ Xóa dữ liệu", use_container_width=True):
        st.session_state.db = []
        st.session_state.pred = None
        save_database([])
        st.rerun()

# ================= 🖥️ MAIN INTERFACE =================
# 💡 FIX: Dùng single quotes cho f-string khi cần access dict với double quotes
st.markdown(f'<h1 style="text-align: center; color: {accent_color}; text-shadow: 0 0 15px {accent_color};">🚀 TITAN V27 - NVIDIA AI EDITION</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="text-align: center; color: {text_secondary};">Hệ thống dự đoán lô đề 3 số 5 tinh • Powered by NVIDIA Llama-3.1 & Gemini-1.5</p>', unsafe_allow_html=True)

# 📥 Input Section
col_input, col_info = st.columns([2.5, 1])
with col_input:
    raw_input = st.text_area(
        "📡 DÁN DỮ LIỆU KỲ QUAY (mỗi dòng 1 số 5 chữ số):", 
        height=120, 
        placeholder="32457\n83465\n19283\n..."
    )
    
with col_info:
    st.metric("📊 Tổng kỳ lưu trữ", len(st.session_state.db))
    st.metric("🎯 Kỳ gần nhất", st.session_state.db[-1] if st.session_state.db else "Chưa có")
    
    if st.button("⚡ CHỐT SỐ (NVIDIA AI)", use_container_width=True, type="primary"):
        clean_data = extract_lottery_numbers(raw_input)
        if clean_data:
            # Update DB
            st.session_state.db.extend(clean_data)
            save_database(st.session_state.db)
            
            # 🔄 Fallback Algorithm
            fallback_pred = generate_fallback_prediction(st.session_state.db, PAIR_RULES)
            
            # 🤖 AI Prediction Pipeline
            prompt = format_for_ai(st.session_state.db, PAIR_RULES)
            
            try:
                # Priority 1: NVIDIA API
                completion = nv_ai.chat.completions.create(
                    model=AI_MODELS["nvidia"]["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=AI_MODELS["nvidia"]["temperature"],
                    response_format={"type": "json_object"}
                )
                ai_response = json.loads(completion.choices[0].message.content)
                st.session_state.pred = ai_response
                
            except Exception as e:
                try:
                    # Priority 2: Gemini fallback
                    res = gm_ai.generate_content(prompt)
                    json_match = re.search(r'\{[\s\S]*\}', res.text)
                    if json_match:
                        st.session_state.pred = json.loads(json_match.group())
                    else:
                        raise ValueError("No JSON found")
                        
                except Exception as e2:
                    # Priority 3: Local algorithm
                    st.warning(f"⚠️ AI error. Dùng thuật toán dự phòng.")
                    st.session_state.pred = fallback_pred
            
            st.rerun()
        else:
            st.error("❌ Không tìm thấy số 5 chữ số hợp lệ!")

# ================= 📊 DISPLAY PREDICTION =================
if st.session_state.pred:
    p = st.session_state.pred
    is_recommended = p.get('adv', '').upper() == "ĐÁNH"
    status_color = accent_color if is_recommended else danger_color
    status_text = "🔥 KHUYÊN ĐÁNH" if is_recommended else "⏸️ NÊN DỪNG"
    
    st.markdown(f'''
    <div class='status-bar' style='background:{status_color}; color:#000'>
    {status_text} | Độ tin cậy: {p.get('conf', 0)}%
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("<div class='main-box'>", unsafe_allow_html=True)
    
    c_main, c_sub = st.columns([1.5, 1])
    with c_main:
        st.markdown(f"<p style='text-align:center; color:{text_secondary}'>🎯 3 SỐ CHỦ LỰC</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-num'>{p.get('main', '---')}</div>", unsafe_allow_html=True)
        st.caption("Chọn 3 số này để cược chính")
        
    with c_sub:
        st.markdown(f"<p style='text-align:center; color:{text_secondary}'>🛡️ 4 SỐ LÓT BẢO VỆ</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-num'>{p.get('sub', '---')}</div>", unsafe_allow_html=True)
        st.caption("Dùng để lót/rải thêm nếu muốn an toàn")
    
    st.divider()
    
    # 🧠 AI Logic Explanation
    st.markdown(f"🧠 **PHÂN TÍCH CỦA AI:** {p.get('logic', 'Đang xử lý...')}")
    
    # 📋 Copy-ready 7-number combo
    dan_7 = "".join(sorted(set(p.get('main', '') + p.get('sub', ''))))[:7]
    st.text_input("📋 DÀN 7 SỐ COPY CHO KUBET/APP:", dan_7, disabled=True)
    
    # ✅ Kiểm tra nhanh với kết quả mới
    with st.expander("🔍 Test với kết quả thực tế"):
        test_result = st.text_input("Nhập kết quả 5 chữ số mới để kiểm tra:", max_chars=5)
        if test_result and len(test_result) == 5 and test_result.isdigit():
            main_nums = list(p.get('main', ''))
            if len(main_nums) == 3:
                is_win = check_win_3so5tinh(main_nums, test_result)
                icon = "🎉" if is_win else "❌"
                msg = "THẮNG CƯỢC!" if is_win else "Chưa trúng - thử lại kỳ sau"
                if is_win:
                    st.success(f"{icon} Với số chính `{p['main']}` + kết quả `{test_result}` → **{msg}**")
                else:
                    st.warning(f"{icon} Với số chính `{p['main']}` + kết quả `{test_result}` → **{msg}**")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ================= 📈 ANALYTICS SECTION =================
if st.session_state.db:
    with st.expander("📊 PHÂN TÍCH TẦN SUẤT 100 KỲ GẦN NHẤT", expanded=False):
        freq_data = Counter("".join(st.session_state.db[-100:]))
        df_freq = pd.DataFrame([
            {"Số": str(i), "Tần suất": freq_data.get(str(i), 0)} 
            for i in range(10)
        ])
        st.bar_chart(df_freq.set_index("Số"), color=accent_secondary)
        
        col_hot, col_cold = st.columns(2)
        with col_hot:
            st.markdown("### 🔥 Top 3 số 'NÓNG'")
            top_hot = df_freq.nlargest(3, "Tần suất")
            for _, row in top_hot.iterrows():
                st.markdown(f"`{row['Số']}` → {row['Tần suất']} lần")
        with col_cold:
            st.markdown("### ❄️ Top 3 số 'LẠNH'")
            top_cold = df_freq.nsmallest(3, "Tần suất")
            for _, row in top_cold.iterrows():
                st.markdown(f"`{row['Số']}` → {row['Tần suất']} lần")

# ================= ⚙️ DEV / DEBUG SECTION =================
with st.expander("⚙️ Dev Tools / Debug"):
    if st.button("🧪 Test ví dụ luật chơi"):
        results = calculate_win_examples()
        for r in results:
            icon = "✅" if r["correct"] else "❌"
            st.text(f"{icon} Bet:{r['bet']} Draw:{r['draw']} → {'WIN' if r['predicted'] else 'LOSE'}")
    
    if st.button("📋 Xem 10 kỳ gần nhất"):
        for i, num in enumerate(reversed(st.session_state.db[-10:]), 1):
            st.text(f"{i}. {num}")
    
    st.json({
        "config": {
            "db_entries": len(st.session_state.db),
            "pair_rules_count": len(PAIR_RULES),
            "ai_models": list(AI_MODELS.keys())
        }
    })

# ================= 🦶 FOOTER =================
st.markdown("---")
footer_html = f'''
<div style="text-align: center; color: {text_secondary}; font-size: 14px;">
🔐 TITAN V27 • Build for Micro-SaaS • 
<a href="https://github.com/yourusername/titan-v27" style="color: {accent_secondary}; text-decoration: none;">GitHub Repo</a> • 
<span style="color: {accent_color};">v2.7.1-FIXED</span>
</div>
'''
st.markdown(footer_html, unsafe_allow_html=True)