import streamlit as st
import google.generativeai as genai
from openai import OpenAI
import re, json, os, pandas as pd, numpy as np
from collections import Counter
from datetime import datetime

# ================= ⚙️ CẤU HÌNH (BẢO MẬT) =================
NV_KEY = "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"
GM_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
DB_FILE = "titan_v31_hybrid.json"

# Bộ quy tắc cặp số xương máu của anh Đạt
PAIR_RULES = ["178", "034", "458", "578", "019", "679", "235", "456", "124", "245", 
              "247", "248", "246", "340", "349", "348", "015", "236", "028", "026", 
              "047", "046", "056", "136", "138", "378"]

# ================= 🧠 LOGIC THUẬT TOÁN (TỬ HUYỆT NHÀ CÁI) =================

def check_win_strict(result_5d, prediction_3_str):
    """Kiểm tra trúng thưởng chuẩn luật 3 số 5 tinh"""
    if not prediction_3_str or len(prediction_3_str) != 3: return False
    pred_set = set([int(d) for d in str(prediction_3_str)])
    result_set = set([int(d) for d in str(result_5d) if d.isdigit()])
    return pred_set.issubset(result_set)

def titan_algo_core(db_data):
    """Thuật toán Cặp Sâu + Số Khan (Bản nâng cấp)"""
    if len(db_data) < 10: return "012", "3456"
    
    matrix = []
    for val in db_data[-100:]:
        digits = [int(d) for d in str(val) if d.isdigit()]
        if len(digits) == 5: matrix.append(digits)
    matrix = np.array(matrix)

    # 1. Tìm cặp số hay đi cùng nhau nhất
    pairs_count = {}
    for i in range(10):
        for j in range(i + 1, 10):
            count = np.sum(np.all(np.any(matrix == np.array([i, j])[:, None, None], axis=0), axis=1))
            pairs_count[(i, j)] = count
    best_pair = sorted(pairs_count.items(), key=lambda x: x[1], reverse=True)[0][0]
    
    # 2. Tìm số khan (lâu chưa ra) để ghép bộ 3
    last_seen = {i: 99 for i in range(10)}
    for idx, row in enumerate(matrix):
        for d in row: last_seen[d] = len(matrix) - 1 - idx
    khan_numbers = sorted(last_seen.items(), key=lambda x: x[1], reverse=True)
    
    best_third = next(n for n, g in khan_numbers if n not in best_pair)
    main_3 = "".join(map(str, sorted([best_pair[0], best_pair[1], best_third])))
    
    # 3. Tạo dàn lót 4 số
    sub_nums = [n for n, g in khan_numbers if str(n) not in main_3]
    sub_4 = "".join(map(str, sub_nums[:4]))
    
    return main_3, sub_4

# ================= 🎨 GIAO DIỆN (PHONG CÁCH NVIDIA V27) =================

st.set_page_config(page_title="TITAN V31 | HYBRID ELITE", layout="wide")

st.markdown(f"""
    <style>
    .stApp {{ background-color: #05050a; color: #ffffff; }}
    .main-box {{ background: #0d1117; border: 2px solid #76b900; border-radius: 20px; padding: 25px; box-shadow: 0 0 20px rgba(118,185,0,0.2); }}
    .big-num {{ font-size: 90px; font-weight: 900; color: #76b900; text-align: center; letter-spacing: 15px; text-shadow: 0 0 20px #76b900; font-family: monospace; }}
    .status-bar {{ padding: 12px; border-radius: 50px; text-align: center; font-weight: bold; margin-bottom: 20px; color: #000; }}
    </style>
""", unsafe_allow_html=True)

# Khởi tạo Session
if "db" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.db = json.load(f)
    else: st.session_state.db = []
if "pred" not in st.session_state: st.session_state.pred = None
if "history" not in st.session_state: st.session_state.history = []

# --- SIDEBAR LUẬT CHƠI ---
with st.sidebar:
    st.title("🎮 TITAN V31")
    st.info("Luật: Thắng khi kết quả có ĐỦ 3 số đã chọn (không tính thứ tự).")
    if st.button("🗑️ XÓA TOÀN BỘ DỮ LIỆU"):
        st.session_state.db, st.session_state.pred, st.session_state.history = [], None, []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

# --- MAIN UI ---
st.markdown("<h1 style='text-align: center; color: #76b900;'>🚀 TITAN V31 - HYBRID ELITE</h1>", unsafe_allow_html=True)

col_in, col_res = st.columns([1, 1.2])

with col_in:
    st.subheader("📥 Nhập Dữ Liệu")
    raw_input = st.text_area("Dán dãy số 5 chữ số (mỗi kỳ 1 dòng):", height=200)
    
    if st.button("⚡ PHÂN TÍCH & ĐỐI SOÁT", use_container_width=True):
        clean = re.findall(r"\d{5}", raw_input)
        if clean:
            # Đối soát ván trước
            if st.session_state.pred:
                is_win = check_win_strict(clean[0], st.session_state.pred['main'])
                st.session_state.history.insert(0, {
                    "Kỳ": clean[0], "Cược": st.session_state.pred['main'], 
                    "KQ": "🔥 WIN" if is_win else "❌ LOSS"
                })

            st.session_state.db.extend(clean)
            st.session_state.db = list(dict.fromkeys(st.session_state.db))[-2000:]
            with open(DB_FILE, "w") as f: json.dump(st.session_state.db, f)
            
            # Gọi AI & Thuật toán
            m3, s4 = titan_algo_core(st.session_state.db)
            prompt = f"Data 5D: {st.session_state.db[-40:]}. Luật: Tìm 3 số phân biệt xuất hiện cùng lúc. Trả JSON: {{'main': '3 số', 'sub': '4 số', 'adv': 'ĐÁNH/DỪNG', 'conf': 95}}"
            
            try:
                client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NV_KEY)
                res = client.chat.completions.create(model="meta/llama-3.1-70b-instruct", messages=[{"role":"user","content":prompt}], temperature=0.1, response_format={"type":"json_object"})
                st.session_state.pred = json.loads(res.choices[0].message.content)
            except:
                st.session_state.pred = {"main": m3, "sub": s4, "adv": "ĐÁNH", "conf": 80}
            st.rerun()

with col_res:
    if st.session_state.pred:
        p = st.session_state.pred
        bg = "#76b900" if p['adv'] == "ĐÁNH" else "#ff4444"
        st.markdown(f"<div class='status-bar' style='background:{bg};'>{p['adv']} | ĐỘ TIN CẬY: {p['conf']}%</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='main-box'>", unsafe_allow_html=True)
        st.write("🎯 **3 SỐ CHÍNH (ĐÁNH THEO BỘ):**")
        st.markdown(f"<div class='big-num'>{p['main']}</div>", unsafe_allow_html=True)
        st.write(f"🛡️ **DÀN LÓT AN TOÀN:** {p['sub']}")
        
        dan_7 = "".join(sorted(set(p['main'] + p['sub'])))[:7]
        st.text_input("📋 DÀN 7 SỐ KUBET:", dan_7)
        st.markdown("</div>", unsafe_allow_html=True)

# Bảng lịch sử trúng thưởng thực tế
if st.session_state.history:
    st.divider()
    st.subheader("📋 Nhật Ký Đối Soát Thực Tế")
    st.table(pd.DataFrame(st.session_state.history).head(10))
