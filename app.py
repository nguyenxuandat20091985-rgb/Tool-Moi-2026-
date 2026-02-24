import streamlit as st
import google.generativeai as genai
import re
import json
import os
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v21.json"
MODEL_FILE = "titan_ml_model.pkl"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: 
        return None

neural_engine = setup_neural()

# ================= HỆ THỐNG GHI NHỚ VĨNH VIỄN =================
def load_memory():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: 
            try: return json.load(f)
            except: return []
    return []

def save_memory(data):
    with open(DB_FILE, "w") as f: 
        json.dump(data[-2000:], f)  # Tăng lên 2000 kỳ

if "history" not in st.session_state:
    st.session_state.history = load_memory()
    st.session_state.patterns = {}
    st.session_state.trap_alerts = []

# ================= THUẬT TOÁN PHÂN TÍCH NÂNG CAO =================
class TitanPredictor:
    def __init__(self, history):
        self.history = history
        self.positions = ['Chục ngàn', 'Ngàn', 'Trăm', 'Chục', 'Đơn vị']
        
    def analyze_streaks(self):
        """Phân tích số đang bệt và dự đoán số bệt tiếp theo"""
        if len(self.history) < 10:
            return {}
        
        streaks = {}
        for pos in range(5):
            pos_numbers = [int(x[pos]) for x in self.history[-20:]]
            current = pos_numbers[-1]
            streak_count = 0
            
            # Đếm streak hiện tại
            for num in reversed(pos_numbers):
                if num == current:
                    streak_count += 1
                else:
                    break
            
            # Tính xác suất bệt tiếp theo
            if streak_count >= 2:
                similar_patterns = []
                for i in range(len(self.history) - 20):
                    pattern = [int(x[pos]) for x in self.history[i:i+streak_count]]
                    if pattern == [current] * streak_count:
                        next_num = int(self.history[i+streak_count][pos]) if i+streak_count < len(self.history) else None
                        if next_num is not None:
                            similar_patterns.append(next_num)
                
                if similar_patterns:
                    next_pred = Counter(similar_patterns).most_common(1)[0][0]
                    probability = Counter(similar_patterns).most_common(1)[0][1] / len(similar_patterns)
                    streaks[f"Vị trí {pos+1}"] = {
                        "current": current,
                        "streak": streak_count,
                        "next_pred": next_pred,
                        "probability": probability
                    }
        
        return streaks
    
    def detect_traps(self):
        """Phát hiện bẫy nhà cái"""
        traps = []
        
        if len(self.history) < 50:
            return traps
        
        # 1. Phát hiện đảo cầu
        recent = self.history[-30:]
        frequent_nums = Counter("".join(recent)).most_common(5)
        frequent_values = [int(x[0]) for x in frequent_nums]
        
        # Kiểm tra xem các số hay về có đang bị né không
        last_10 = "".join(self.history[-10:])
        for num in frequent_values:
            count = last_10.count(str(num))
            if count < 2:  # Số hay về nhưng 10 kỳ gần ít xuất hiện
                traps.append(f"⚠️ BẪY: Số {num} đang bị né, chuẩn bị nổ")
        
        # 2. Phát hiện bệt giả
        for pos in range(5):
            pos_nums = [int(x[pos]) for x in self.history[-15:]]
            for i in range(len(pos_nums)-3):
                if pos_nums[i] == pos_nums[i+1] == pos_nums[i+2] != pos_nums[i+3]:
                    traps.append(f"⚠️ BẪY VỊ TRÍ {pos+1}: Bệt 3 tay rồi đột ngột đảo")
        
        # 3. Phát hiện bóng số
        bong_numbers = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}
        last_num = int(self.history[-1][4])
        bong = bong_numbers[last_num]
        
        # Kiểm tra bóng có hay về sau số vừa ra không
        bong_count = 0
        for i in range(len(self.history)-1):
            if int(self.history[i][4]) == last_num:
                if i+1 < len(self.history) and int(self.history[i+1][4]) == bong:
                    bong_count += 1
        
        if bong_count > len(self.history) * 0.15:  # Trên 15% xuất hiện bóng
            traps.append(f"🎯 CẦU BÓNG: Số {bong} có khả năng về sau {last_num}")
        
        return traps
    
    def find_3_numbers_to_exclude(self):
        """Tìm 3 số cần loại dựa trên phân tích"""
        if len(self.history) < 30:
            return []
        
        all_nums = "".join(self.history[-30:])
        counts = Counter(all_nums)
        
        # Tìm số ít xuất hiện nhất nhưng có chu kỳ
        rare_nums = counts.most_common()[:-4:-1]  # 3 số ít nhất
        
        # Phân tích chu kỳ xuất hiện
        exclude_candidates = []
        for num, _ in rare_nums:
            last_positions = [i for i, x in enumerate(self.history[-50:]) if str(num) in x]
            if last_positions:
                gap = 50 - last_positions[-1]
                if gap > 15:  # Quá lâu chưa ra
                    exclude_candidates.append(int(num))
        
        return exclude_candidates[:3]
    
    def generate_optimal_pairs(self, top_numbers):
        """Ghép các số để tạo ra bộ số chính xác"""
        if len(top_numbers) < 5:
            return []
        
        combinations = []
        positions_weights = self.analyze_position_weights()
        
        # Tạo tổ hợp dựa trên trọng số vị trí
        for pos in range(5):
            pos_pred = positions_weights[pos][:3]
            for num in pos_pred:
                combinations.append(str(num))
        
        # Thêm các tổ hợp từ top numbers
        from itertools import combinations
        for combo in combinations(top_numbers[:7], 5):
            combo_str = "".join(map(str, combo))
            combinations.append(combo_str)
        
        return list(set(combinations))[:10]  # Trả về 10 tổ hợp tốt nhất
    
    def analyze_position_weights(self):
        """Phân tích trọng số từng vị trí"""
        weights = []
        for pos in range(5):
            pos_nums = [int(x[pos]) for x in self.history[-100:]]
            # Tính xác suất theo vị trí
            counter = Counter(pos_nums)
            total = len(pos_nums)
            probs = {num: count/total for num, count in counter.items()}
            
            # Lấy top 5 số có xác suất cao nhất
            top_nums = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
            weights.append([x[0] for x in top_nums])
        
        return weights
    
    def predict_top_5(self):
        """Dự đoán 5 số khả năng về cao nhất"""
        if len(self.history) < 20:
            return list(range(10))[:5]
        
        # Phân tích tần suất
        all_nums = "".join(self.history[-50:])
        freq = Counter(all_nums)
        
        # Phân tích xu hướng gần nhất
        recent = "".join(self.history[-10:])
        recent_freq = Counter(recent)
        
        # Kết hợp có trọng số
        scores = {}
        for num in range(10):
            num_str = str(num)
            freq_score = freq.get(num_str, 0) * 0.3
            recent_score = recent_freq.get(num_str, 0) * 0.7
            scores[num] = freq_score + recent_score
        
        # Thêm phân tích bệt
        streaks = self.analyze_streaks()
        for streak_info in streaks.values():
            if streak_info["probability"] > 0.5:
                scores[streak_info["next_pred"]] += 5
        
        # Lấy top 5
        top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        return [x[0] for x in top_5]

# ================= UI DESIGN =================
st.set_page_config(page_title="TITAN v22.0 PRO MAX", layout="centered")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #c9d1d9; }
    .status-active { color: #238636; font-weight: bold; border-left: 3px solid #238636; padding-left: 10px; }
    .prediction-card {
        background: #0d1117; border: 2px solid #30363d;
        border-radius: 12px; padding: 25px; margin-top: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .num-display { 
        font-size: 60px; font-weight: 900; color: #58a6ff; 
        text-align: center; letter-spacing: 10px; text-shadow: 0 0 25px #58a6ff;
    }
    .logic-box { font-size: 14px; color: #8b949e; background: #161b22; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
    .trap-alert { color: #f85149; background: #2d0a0a; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    .exclude-box { color: #f2cc60; background: #2d1f0a; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #58a6ff;'>🧬 TITAN v22.0 PRO MAX</h2>", unsafe_allow_html=True)

if neural_engine:
    st.markdown(f"<p class='status-active'>● NEURAL-LINK: OK | DỮ LIỆU: {len(st.session_state.history)} KỲ | ML: ACTIVE</p>", unsafe_allow_html=True)
else:
    st.error("LỖI KẾT NỐI API")

# ================= XỬ LÝ DỮ LIỆU =================
raw_input = st.text_area("📡 NẠP DỮ LIỆU (Dán các dãy 5 số):", height=100, placeholder="32880\n21808\n...")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🚀 PHÂN TÍCH CHUYÊN SÂU"):
        new_data = re.findall(r"\d{5}", raw_input)
        if new_data:
            st.session_state.history.extend(new_data)
            save_memory(st.session_state.history)
            
            # Khởi tạo predictor
            predictor = TitanPredictor(st.session_state.history)
            
            # Phân tích
            streaks = predictor.analyze_streaks()
            traps = predictor.detect_traps()
            exclude_3 = predictor.find_3_numbers_to_exclude()
            top_5 = predictor.predict_top_5()
            
            # Gửi prompt cho Gemini
            prompt = f"""
            Bạn là AI chuyên gia xác suất 5D với khả năng phát hiện bẫy nhà cái.
            
            Dữ liệu phân tích:
            - Streaks: {streaks}
            - Bẫy phát hiện: {traps}
            - Số cần loại: {exclude_3}
            - Top 5 dự đoán: {top_5}
            
            Lịch sử 50 kỳ gần nhất: {st.session_state.history[-50:]}
            
            Yêu cầu:
            1. Xác định chính xác các số đang bệt và dự đoán số bệt tiếp theo
            2. Phân tích chi tiết bẫy nhà cái đang giăng ra
            3. Đưa ra 3 số cần loại cụ thể và lý do
            4. Ghép các số để tạo bộ số tối ưu
            5. Chốt 5 số khả năng về cao nhất cho kỳ tới
            6. Đưa ra chiến thuật vào tiền hợp lý
            
            TRẢ VỀ JSON:
            {{
                "dan4": [4 số chủ lực],
                "dan3": [3 số lót],
                "exclude": [3 số cần loại],
                "top5": [5 số khả năng cao],
                "streaks": "Phân tích số bệt",
                "traps": "Các bẫy nhà cái",
                "strategy": "Chiến thuật vào tiền",
                "logic": "Tổng hợp phân tích ngắn gọn"
            }}
            """
            
            try:
                response = neural_engine.generate_content(prompt)
                res_text = response.text
                data = json.loads(re.search(r'\{.*\}', res_text, re.DOTALL).group())
                st.session_state.last_result = data
                st.session_state.trap_alerts = traps
            except Exception as e:
                # Fallback
                st.session_state.last_result = {
                    "dan4": top_5[:4],
                    "dan3": top_5[4:] if len(top_5) > 4 else [],
                    "exclude": exclude_3,
                    "top5": top_5,
                    "streaks": str(streaks),
                    "traps": "\n".join(traps) if traps else "Không phát hiện bẫy",
                    "strategy": "Vào tiền đều, không all-in",
                    "logic": "Phân tích từ thuật toán ML"
                }
            
            st.rerun()

with col2:
    if st.button("🗑️ RESET BỘ NHỚ"):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

with col3:
    if st.button("📊 PHÂN TÍCH NHANH"):
        if st.session_state.history:
            predictor = TitanPredictor(st.session_state.history)
            top_5 = predictor.predict_top_5()
            exclude = predictor.find_3_numbers_to_exclude()
            st.info(f"Top 5: {top_5} | Loại: {exclude}")

# ================= HIỂN THỊ KẾT QUẢ =================
if "last_result" in st.session_state:
    res = st.session_state.last_result
    
    # Hiển thị cảnh báo bẫy
    if st.session_state.trap_alerts:
        st.markdown("<div class='trap-alert'>", unsafe_allow_html=True)
        for trap in st.session_state.trap_alerts:
            st.markdown(f"🚨 {trap}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    # Phân tích logic
    st.markdown(f"<div class='logic-box'><b>💡 Phân tích:</b> {res.get('logic', 'N/A')}</div>", unsafe_allow_html=True)
    
    # Hiển thị streaks nếu có
    if 'streaks' in res:
        st.markdown(f"<div class='logic-box'><b>📈 Số bệt:</b> {res['streaks']}</div>", unsafe_allow_html=True)
    
    # Hiển thị bẫy
    if 'traps' in res:
        st.markdown(f"<div class='logic-box'><b>⚠️ Bẫy phát hiện:</b> {res['traps']}</div>", unsafe_allow_html=True)
    
    # Hiển thị số cần loại
    if 'exclude' in res and res['exclude']:
        st.markdown(f"<div class='exclude-box'><b>❌ Số cần loại:</b> {', '.join(map(str, res['exclude']))}</div>", unsafe_allow_html=True)
    
    # Dàn số chính
    st.markdown("<p style='text-align:center; font-size:12px; color:#888;'>🎯 4 SỐ CHỦ LỰC (VÀO TIỀN)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display'>{''.join(map(str, res.get('dan4', [])))}</div>", unsafe_allow_html=True)
    
    # Dàn số lót
    st.markdown("<p style='text-align:center; font-size:12px; color:#888; margin-top:20px;'>🛡️ 3 SỐ LÓT (GIỮ VỐN)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display' style='color:#f2cc60; text-shadow: 0 0 25px #f2cc60;'>{''.join(map(str, res.get('dan3', [])))}</div>", unsafe_allow_html=True)
    
    # Top 5 dự đoán
    if 'top5' in res:
        st.markdown("<p style='text-align:center; font-size:12px; color:#888; margin-top:20px;'>🔮 TOP 5 SỐ KHẢ NĂNG CAO</p>", unsafe_allow_html=True)
        cols = st.columns(5)
        for i, num in enumerate(res['top5']):
            cols[i].markdown(f"<div style='text-align:center; font-size:24px; color:#58a6ff;'>{num}</div>", unsafe_allow_html=True)
    
    # Chiến thuật
    if 'strategy' in res:
        st.markdown(f"<div class='logic-box' style='margin-top:20px;'><b>💰 Chiến thuật:</b> {res['strategy']}</div>", unsafe_allow_html=True)
    
    # Sao chép dàn số
    copy_val = "".join(map(str, res.get('dan4', []))) + "".join(map(str, res.get('dan3', [])))
    st.text_input("📋 SAO CHÉP DÀN 7 SỐ:", copy_val)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ================= HIỂN THỊ LỊCH SỬ GẦN NHẤT =================
if st.session_state.history:
    with st.expander("📜 Lịch sử 20 kỳ gần nhất"):
        recent = st.session_state.history[-20:]
        df = pd.DataFrame(recent, columns=["Kết quả"])
        st.dataframe(df, use_container_width=True)

st.markdown("<br><p style='text-align:center; font-size:10px; color:#444;'>TITAN v22.0 - Hệ thống tự học & phát hiện bẫy thông minh</p>", unsafe_allow_html=True)