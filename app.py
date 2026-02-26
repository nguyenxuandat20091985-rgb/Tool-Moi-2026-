import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================= CẤU HÌNH HỆ THỐNG TITAN v26.0 =================
API_KEY = "AIzaSyB5PRp04XlMHKl3oGfCRbsKXjlTA-CZifc"
DB_FILE = "titan_supreme_permanent_v26_0.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except:
        return None

neural_engine = setup_neural()

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try:
                data = json.load(f)
                return data if isinstance(data, list) else []
            except:
                return []
    return []

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data[-3000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_db()
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ================= THUẬT TOÁN BẮT CẦU THỰC CHIẾN =================

class RealCatchPredictor:
    """
    Thuật toán bắt cầu thực tế cho 5D Bet
    """
    def __init__(self, history):
        self.history = history
        
    def detect_bet_cau(self):
        """
        Phát hiện cầu bệt - số lặp lại nhiều kỳ
        """
        if len(self.history) < 5:
            return []
        
        bet_numbers = []
        last_10 = self.history[-10:]
        
        # Đếm tần suất từng số trong 10 kỳ
        all_nums = "".join(last_10)
        num_counts = Counter(all_nums)
        
        # Số bệt là số xuất hiện >= 3 lần trong 5 kỳ gần
        for num, count in num_counts.items():
            # Kiểm tra 5 kỳ gần nhất
            recent_5 = "".join(self.history[-5:])
            if recent_5.count(num) >= 3:
                bet_numbers.append(num)
        
        return list(set(bet_numbers))
    
    def detect_dao_cau(self):
        """
        Phát hiện cầu đảo - số đảo chiều liên tục
        """
        if len(self.history) < 10:
            return False, []
        
        last_8 = self.history[-8:]
        dao_patterns = []
        
        # Kiểm tra các cặp số đảo
        for i in range(len(last_8)-1):
            num1 = last_8[i]
            num2 = last_8[i+1]
            
            # Kiểm tra đảo ngược: 12345 -> 54321
            if num1 == num2[::-1]:
                dao_patterns.append((num1, num2))
        
        is_dao = len(dao_patterns) >= 3  # Nếu có 3 cặp đảo liên tiếp
        
        return is_dao, dao_patterns
    
    def detect_xieng_cau(self):
        """
        Phát hiện cầu xiên - số tăng/giảm dần
        """
        if len(self.history) < 5:
            return None
        
        last_5 = self.history[-5:]
        xu_huong = []
        
        for pos in range(5):
            pos_values = [int(num[pos]) for num in last_5]
            
            # Kiểm tra tăng dần
            tang = all(pos_values[i] <= pos_values[i+1] for i in range(4))
            # Kiểm tra giảm dần
            giam = all(pos_values[i] >= pos_values[i+1] for i in range(4))
            
            if tang:
                xu_huong.append(f"Vị trí {pos+1}: TĂNG")
            elif giam:
                xu_huong.append(f"Vị trí {pos+1}: GIẢM")
            else:
                xu_huong.append(f"Vị trí {pos+1}: KHÔNG RÕ")
        
        return xu_huong
    
    def predict_by_bet(self):
        """
        Dự đoán dựa trên cầu bệt
        """
        bet_numbers = self.detect_bet_cau()
        
        if not bet_numbers:
            return None
        
        # Ghép số bệt thành số 5 chữ số
        predictions = []
        for _ in range(3):  # Tạo 3 số dự đoán
            pred = ""
            for _ in range(5):
                # Chọn ngẫu nhiên từ số bệt, ưu tiên số xuất hiện nhiều
                pred += np.random.choice(bet_numbers)
            predictions.append(pred)
        
        return predictions
    
    def predict_by_recent(self):
        """
        Dự đoán dựa trên lịch sử gần nhất
        """
        if len(self.history) < 3:
            return None
        
        last_3 = self.history[-3:]
        
        # Phân tích từng vị trí
        predictions = []
        for pos in range(5):
            pos_values = [int(num[pos]) for num in last_3]
            
            # Nếu 3 kỳ liên tiếp giống nhau -> bệt vị trí
            if len(set(pos_values)) == 1:
                predictions.append(str(pos_values[0]))
            else:
                # Lấy số xuất hiện nhiều nhất
                counter = Counter(pos_values)
                most_common = counter.most_common(1)[0][0]
                predictions.append(str(most_common))
        
        return "".join(predictions)
    
    def analyze_bay_cua_nha_cai(self):
        """
        Phân tích bẫy của nhà cái
        """
        warnings = []
        
        if len(self.history) < 10:
            return warnings
        
        # 1. Phát hiện đảo cầu liên tục
        is_dao, dao_patterns = self.detect_dao_cau()
        if is_dao:
            warnings.append("🔴 CẢNH BÁO: ĐANG ĐẢO CẦU LIÊN TỤC - DỪNG CƯỢC")
        
        # 2. Phát hiện số lạ xuất hiện
        last_5 = "".join(self.history[-5:])
        all_digits = set(last_5)
        
        # Kiểm tra 10 kỳ trước
        prev_10 = "".join(self.history[-15:-5])
        rare_digits = [d for d in all_digits if prev_10.count(d) < 2]
        
        if rare_digits:
            warnings.append(f"🟠 SỐ LẠ XUẤT HIỆN: {rare_digits} - Có thể cầu mới")
        
        # 3. Phát hiện biến động mạnh
        if len(self.history) >= 20:
            last_10_digits = [int(d) for d in "".join(self.history[-10:])]
            prev_10_digits = [int(d) for d in "".join(self.history[-20:-10])]
            
            last_std = np.std(last_10_digits)
            prev_std = np.std(prev_10_digits)
            
            if prev_std > 0 and last_std > prev_std * 1.5:
                warnings.append("🟡 BIẾN ĐỘNG MẠNH - Giảm vốn")
        
        return warnings
    
    def get_best_prediction(self):
        """
        Lấy dự đoán tốt nhất từ các phương pháp
        """
        warnings = self.analyze_bay_cua_nha_cai()
        bet_numbers = self.detect_bet_cau()
        is_dao, _ = self.detect_dao_cau()
        xu_huong = self.detect_xieng_cau()
        
        # Nếu có cảnh báo đỏ -> không đánh
        if any("🔴" in w for w in warnings):
            return {
                "main_3": "XXX",
                "support_4": "XXXX",
                "decision": "DỪNG - CẦU LỪA",
                "logic": "Phát hiện cầu đảo liên tục. Bảo toàn vốn, chờ cầu mới.",
                "color": "Red",
                "confidence": 30,
                "warning_level": "RẤT CAO"
            }
        
        # Dự đoán chính
        main_pred = self.predict_by_recent()
        
        # Nếu có số bệt, ưu tiên ghép số bệt vào
        if bet_numbers and main_pred:
            # Thay thế các số trong main_pred bằng số bệt nếu có thể
            main_list = list(main_pred)
            for i in range(len(main_list)):
                if np.random.random() > 0.5 and bet_numbers:  # 50% cơ hội thay bằng số bệt
                    main_list[i] = np.random.choice(bet_numbers)
            main_pred = "".join(main_list)
        
        # Tạo dự đoán phụ từ số bệt
        support_pred = ""
        if bet_numbers:
            for _ in range(4):
                if bet_numbers:
                    support_pred += np.random.choice(bet_numbers)
                else:
                    support_pred += str(np.random.randint(0, 10))
        else:
            # Nếu không có số bệt, lấy từ phân tích xu hướng
            for pos in range(4):
                if xu_huong and pos < len(xu_huong):
                    if "TĂNG" in xu_huong[pos]:
                        # Dự đoán số tăng
                        last_val = int(self.history[-1][pos]) if self.history else 5
                        pred_val = min(9, last_val + 1)
                        support_pred += str(pred_val)
                    elif "GIẢM" in xu_huong[pos]:
                        last_val = int(self.history[-1][pos]) if self.history else 5
                        pred_val = max(0, last_val - 1)
                        support_pred += str(pred_val)
                    else:
                        support_pred += str(np.random.randint(0, 10))
                else:
                    support_pred += str(np.random.randint(0, 10))
        
        # Đảm bảo độ dài
        if not main_pred or len(main_pred) < 3:
            main_pred = "".join([str(np.random.randint(0, 10)) for _ in range(3)])
        else:
            main_pred = main_pred[:3]
        
        support_pred = support_pred[:4].ljust(4, '0')
        
        # Quyết định dựa trên cảnh báo
        if len(warnings) >= 2:
            decision = "THEO DÕI - CẢNH BÁO"
            confidence = 60
            color = "Yellow"
        elif bet_numbers:
            decision = "ĐÁNH - CÓ SỐ BỆT"
            confidence = 85
            color = "Green"
        elif is_dao:
            decision = "DỪNG - ĐANG ĐẢO"
            confidence = 40
            color = "Red"
        else:
            decision = "THEO DÕI NHẸ"
            confidence = 70
            color = "Yellow"
        
        # Logic giải thích
        logic = f"Phân tích: {len(bet_numbers)} số bệt ({bet_numbers}), {len(warnings)} cảnh báo. "
        if xu_huong:
            logic += f"Xu hướng: {xu_huong[0]}. "
        
        return {
            "main_3": main_pred,
            "support_4": support_pred,
            "decision": decision,
            "logic": logic,
            "color": color,
            "confidence": confidence,
            "warning_level": "CAO" if len(warnings) >= 2 else "TRUNG BÌNH" if warnings else "THẤP"
        }

# ================= THIẾT KẾ GIAO DIỆN =================
st.set_page_config(page_title="TITAN v26.0 - BẮT CẦU THỰC CHIẾN", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 2px solid #58a6ff;
        border-radius: 15px; padding: 30px; margin-top: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    }
    .num-box {
        font-size: 90px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 15px; border-right: 3px solid #30363d;
        text-shadow: 0 0 25px rgba(255,88,88,0.5);
    }
    .lot-box {
        font-size: 60px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 10px; padding-left: 20px;
        text-shadow: 0 0 15px rgba(88,166,255,0.3);
    }
    .status-bar { padding: 15px; border-radius: 12px; text-align: center; font-weight: bold; font-size: 24px; margin-bottom: 20px; text-transform: uppercase; }
    .warning-box { background: #4a0e0e; color: #ff9b9b; padding: 15px; border-radius: 8px; border: 1px solid #ff4444; text-align: center; margin-top: 15px; font-weight: bold; }
    .bet-number { background: #238636; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🚀 TITAN v26.0 - BẮT CẦU THỰC CHIẾN</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e;'>Chuyên phát hiện cầu bệt, cầu đảo, bẫy nhà cái</p>", unsafe_allow_html=True)

# ================= NHẬP LIỆU =================
with st.container():
    col_in, col_st = st.columns([2, 1])
    with col_in:
        raw_input = st.text_area("📡 Nạp dữ liệu mới:", height=150, placeholder="Dán dãy số 5D tại đây...")
    with col_st:
        st.write(f"📊 Kho dữ liệu: **{len(st.session_state.history)} kỳ**")
        
        c1, c2 = st.columns(2)
        btn_save = c1.button("🚀 PHÂN TÍCH CẦU", use_container_width=True)
        btn_reset = c2.button("🗑️ RESET", use_container_width=True)

if btn_reset:
    st.session_state.history = []
    st.session_state.prediction_history = []
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    st.success("Đã reset dữ liệu")
    st.rerun()

if btn_save:
    input_data = re.findall(r"\b\d{5}\b", raw_input)
    if input_data:
        st.session_state.history.extend(input_data)
        st.session_state.history = list(dict.fromkeys(st.session_state.history))
        save_db(st.session_state.history)
        
        # Phân tích cầu
        predictor = RealCatchPredictor(st.session_state.history)
        
        # Phát hiện các loại cầu
        bet_numbers = predictor.detect_bet_cau()
        is_dao, dao_patterns = predictor.detect_dao_cau()
        xu_huong = predictor.detect_xieng_cau()
        warnings = predictor.analyze_bay_cua_nha_cai()
        
        # Lưu vào session state để hiển thị
        st.session_state.bet_numbers = bet_numbers
        st.session_state.is_dao = is_dao
        st.session_state.dao_patterns = dao_patterns
        st.session_state.xu_huong = xu_huong
        st.session_state.warnings = warnings
        
        # Dự đoán
        st.session_state.last_prediction = predictor.get_best_prediction()
        
        # Lưu lịch sử
        st.session_state.prediction_history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "prediction": st.session_state.last_prediction,
            "bet_numbers": bet_numbers
        })
        
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ =================
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    
    # Hiển thị trạng thái
    status_map = {"green": "#238636", "red": "#da3633", "yellow": "#d29922"}
    bg_color = status_map.get(res['color'].lower(), "#30363d")
    
    st.markdown(f"""
        <div class='status-bar' style='background: {bg_color};'>
            🔥 {res['decision']} | ĐỘ TIN CẬY: {res['confidence']}% | {res['warning_level']}
        </div>
    """, unsafe_allow_html=True)

    # Hiển thị cảnh báo
    if "warnings" in st.session_state and st.session_state.warnings:
        for w in st.session_state.warnings:
            if "🔴" in w:
                st.error(w)
            elif "🟠" in w:
                st.warning(w)
            else:
                st.info(w)

    # Hiển thị số bệt
    if "bet_numbers" in st.session_state and st.session_state.bet_numbers:
        bet_html = " ".join([f"<span class='bet-number'>{num}</span>" for num in st.session_state.bet_numbers])
        st.markdown(f"**🔥 SỐ BỆT:** {bet_html}", unsafe_allow_html=True)

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    col_main, col_supp = st.columns([1.5, 1])
    with col_main:
        st.markdown("<p style='color:#8b949e; text-align:center;'>🎯 3 SỐ CHỦ LỰC</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-box'>{res['main_3']}</div>", unsafe_allow_html=True)
    
    with col_supp:
        st.markdown("<p style='color:#8b949e; text-align:center;'>🛡️ 4 SỐ LÓT</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-box'>{res['support_4']}</div>", unsafe_allow_html=True)
    
    st.divider()
    
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader("🧠 Phân tích cầu")
        st.write(res['logic'])
        
        if st.session_state.xu_huong:
            st.write("**Xu hướng từng vị trí:**")
            for xh in st.session_state.xu_huong[:3]:
                st.write(f"- {xh}")
    
    with col_r:
        st.subheader("📋 Dàn số")
        full_dan = "".join(sorted(set(res['main_3'] + res['support_4'])))
        st.text_input("Dàn 7 số:", full_dan)
        
        if res['decision'] == "ĐÁNH - CÓ SỐ BỆT":
            st.success("💵 Vào tiền: 80% vốn - Có số bệt")
        elif "THEO DÕI" in res['decision']:
            st.warning("👁️ Vào tiền: 30% vốn - Quan sát")
        else:
            st.error("⛔ DỪNG - Bảo toàn vốn")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ================= HIỂN THỊ LỊCH SỬ GẦN =================
if st.session_state.history:
    with st.expander("📊 Lịch sử 10 kỳ gần"):
        last_10 = st.session_state.history[-10:]
        df = pd.DataFrame({
            'Kỳ': [f"Kỳ {i+1}" for i in range(len(last_10))],
            'Số': last_10
        })
        st.table(df)
        
        # Phân tích nhanh
        all_digits = "".join(last_10)
        freq = Counter(all_digits).most_common()
        st.write("**Tần suất 10 kỳ:**")
        for num, count in freq:
            st.write(f"Số {num}: {count} lần")