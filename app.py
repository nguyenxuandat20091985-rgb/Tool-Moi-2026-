import streamlit as st
import google.generativeai as genai
from openai import OpenAI
import re, json, os, pandas as pd, numpy as np
from collections import Counter
from datetime import datetime

# --- CẤU HÌNH GỐC (GIỮ NGUYÊN API CỦA ANH ĐẠT) ---
NV_KEY = "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"
GM_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
DB_FILE = "titan_v30_elite_db.json"

# --- BỘ N NÃO CÔNG CỤ AI ---
def get_neural_engines():
    nv_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NV_KEY)
    genai.configure(api_key=GM_KEY)
    gm_model = genai.GenerativeModel('gemini-1.5-flash')
    return nv_client, gm_model

# --- HÀM KIỂM TRA WIN/LOSS CHUẨN LUẬT (TÌM ĐỦ 3 SỐ KHÁC NHAU) ---
def check_win_strict(result_5d, prediction_3_set):
    """
    Win khi và chỉ khi result_5d chứa đủ cả 3 số trong tập hợp dự đoán.
    """
    if not prediction_3_set or len(prediction_3_set) != 3: return False
    result_set = set([int(d) for d in str(result_5d) if d.isdigit()])
    return prediction_3_set.issubset(result_set)

# --- THUẬT TOÁN CORE TITAN V30 (TÌM CẶP SÂU) ---
class TitanV30Algo:
    def __init__(self, db_data):
        self.matrix = []
        for val in db_data[-100:]: 
            digits = [int(d) for d in str(val) if d.isdigit()]
            if len(digits) == 5: self.matrix.append(digits)
        self.matrix = np.array(self.matrix)

    def analyze(self):
        if len(self.matrix) < 10: return None, None
        
        # 1. Thuật toán "Tìm Cặp Sâu": Tìm 2 số thường xuyên đi cùng nhau
        pairs_count = {}
        for i in range(10):
            for j in range(i + 1, 10):
                count = np.sum(np.all(np.any(self.matrix == np.array([i, j])[:, None, None], axis=0), axis=1))
                pairs_count[(i, j)] = count
        
        best_pair = sorted(pairs_count.items(), key=lambda x: x[1], reverse=True)[0][0]
        
        # 2. Thuật toán "Ghép Số Khan": Tìm số lâu chưa ra nhất
        last_seen = {i: 99 for i in range(10)}
        for idx, row in enumerate(self.matrix):
            for d in row: last_seen[d] = len(self.matrix) - 1 - idx
            
        khan_numbers = sorted(last_seen.items(), key=lambda x: x[1], reverse=True)
        best_third = None
        for num, gap in khan_numbers:
            if num not in best_pair:
                best_third = num
                break
        
        final_3 = sorted([best_pair[0], best_pair[1], best_third])
        main_3_str = "".join([str(d) for d in final_3])
        
        support_nums = [n for n, g in khan_numbers if n not in final_3]
        support_4_str = "".join([str(d) for d in support_nums[:4]])
        
        return main_3_str, support_4_str

# --- GIAO DIỆN VIP v30 ---
st.set_page_config(page_title="TITAN v30 ELITE", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #000; color: #fff; }
    .main-box { background: #080808; border: 2px solid #ff4444; border-radius: 15px; padding: 20px; box-shadow: 0 0 15px #ff4444; }
    .big-num { font-size: 80px; font-weight: 900; color: #ff0000; text-align: center; letter-spacing: 20px; text-shadow: 0 0 20px #ff0000; }
    .stButton>button { background: linear-gradient(45deg, #ff0000, #990000) !important; color: white !important; font-weight: bold; border-radius: 30px; border: none; }
    .status-bar { padding: 10px; border-radius: 10px; text-align: center; font-weight: bold; margin-bottom: 10px; color: black; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #ff0000;'>⚡ TITAN v30 ELITE - FIX UNSHIFT ⚡</h1>", unsafe_allow_html=True)

# Khởi tạo Engines và DB
nv_ai, gm_ai = get_neural_engines()
if "db" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.db = json.load(f)
    else: st.session_state.db = []
if "last_pred_v30" not in st.session_state: st.session_state.last_pred_v30 = None
if "history_log" not in st.session_state: st.session_state.history_log = []

# --- BỐ CỤC ---
col_in, col_res = st.columns([1, 1])

with col_in:
    st.subheader("📥 Dữ liệu & Đối soát")
    raw_input = st.text_area("Dán số mở thưởng (5 chữ số):", height=150, placeholder="Ví dụ: 12864\n27524...")
    
    if st.button("LƯU & CHỐT SỐ MỚI"):
        clean = re.findall(r"\d{5}", raw_input)
        if clean:
            # Đối soát win/loss thực tế kỳ trước
            if st.session_state.last_pred_v30:
                pred_set = set([int(d) for d in str(st.session_state.last_pred_v30['main'])])
                real_result = clean[0]
                is_win = check_win_strict(real_result, pred_set)
                
                # SỬA LỖI TẠI ĐÂY: Dùng .insert(0, ...) thay cho .unshift
                st.session_state.history_log.insert(0, {
                    "Time": datetime.now().strftime("%H:%M"),
                    "Dự đoán": st.session_state.last_pred_v30['main'],
                    "Kết quả": real_result,
                    "Trạng thái": "🔥 WIN" if is_win else "❌ LOSS"
                })

            st.session_state.db.extend(clean)
            st.session_state.db = list(dict.fromkeys(st.session_state.db))[-3000:]
            with open(DB_FILE, "w") as f: json.dump(st.session_state.db, f)

            # CHỐT SỐ MỚI DÙNG DUAL-ENGINE
            algo = TitanV30Algo(st.session_state.db)
            main3, sup4 = algo.analyze()
            
            if main3:
                prompt = f"Data: {st.session_state.db[-50:]}. Find 3 distinct numbers appearing together in 5-digit result. Return JSON: {{'main': '3 digits', 'sub': '4 digits', 'adv': 'BET/WAIT'}}"
                try:
                    res = nv_ai.chat.completions.create(model="meta/llama-3.1-70b-instruct", messages=[{"role": "user", "content": prompt}], temperature=0.1, response_format={"type": "json_object"})
                    st.session_state.last_pred_v30 = json.loads(res.choices[0].message.content)
                except:
                    try:
                        res = gm_ai.generate_content(prompt)
                        st.session_state.last_pred_v30 = json.loads(re.search(r'\{.*\}', res.text).group())
                    except:
                        st.session_state.last_pred_v30 = {"main": main3, "sub": sup4, "adv": "ĐÁNH"}
            st.rerun()

with col_res:
    if st.session_state.last_pred_v30:
        p = st.session_state.last_pred_v30
        st.markdown(f"<div class='status-bar' style='background:#ff4444;'>📢 KHUYẾN NGHỊ: {p.get('adv', 'ĐÁNH')}</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='main-box'>", unsafe_allow_html=True)
        st.write("🔥 **BỘ 3 SỐ CHÍNH (ĐÁNH ĐỦ BỘ)**")
        st.markdown(f<div class='big-num'>{p['main']}</div>", unsafe_allow_html=True)
        
        st.write("🛡️ **4 SỐ LÓT (GIỮ VỐN)**")
        st.write(f"<p style='color:#ccc; text-align:center; font-size:20px;'>{p['sub']}</p>", unsafe_allow_html=True)
        
        st.divider()
        st.write("🧠 **LOGIC TITAN:** Tìm cặp số có tần suất đi cùng nhau cao nhất + Ghép số khan nhất.")
        st.markdown("</div>", unsafe_allow_html=True)

# LỊCH SỬ ĐỐI SOÁT
st.divider()
st.subheader("📋 Nhật ký WIN/LOSS thực tế")
if st.session_state.history_log:
    wins = sum(1 for x in st.session_state.history_log if x['Trạng thái'] == "🔥 WIN")
    rate = (wins / len(st.session_state.history_log)) * 100
    st.metric("TỶ LỆ THẮNG THỰC TẾ", f"{rate:.1f}%")
    st.table(pd.DataFrame(st.session_state.history_log))
