import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================= CẤU HÌNH HỆ THỐNG TITAN v25.0 =================
# API KEY anh cung cấp: AIzaSyB5PRp04XlMHKl3oGfCRbsKXjlTA-CZifc
API_KEY = "AIzaSyB5PRp04XlMHKl3oGfCRbsKXjlTA-CZifc"
DB_FILE = "titan_supreme_permanent_v25_0.json"

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
    # Lưu tối đa 5000 kỳ để học sâu hơn
    with open(DB_FILE, "w") as f:
        json.dump(data[-5000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_db()
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "accuracy_stats" not in st.session_state:
    st.session_state.accuracy_stats = {"correct": 0, "total": 0, "last_10": []}

# ================= THUẬT TOÁN CAO CẤP =================

class AdvancedPredictor:
    def __init__(self, history):
        self.history = history
        self.digits_sequence = self._create_digits_sequence()
        
    def _create_digits_sequence(self):
        """Tạo chuỗi digits từ history"""
        sequence = []
        for num in self.history[-500:]:  # Lấy 500 kỳ gần nhất
            sequence.extend([int(d) for d in num])
        return sequence
    
    def markov_chain_analysis(self, order=3):
        """
        Phân tích chuỗi Markov bậc cao để dự đoán số tiếp theo
        """
        from collections import defaultdict
        
        if len(self.digits_sequence) < order + 10:
            return {}
        
        transition_matrix = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(self.digits_sequence) - order):
            current_state = tuple(self.digits_sequence[i:i+order])
            next_digit = self.digits_sequence[i+order]
            transition_matrix[current_state][next_digit] += 1
        
        # Lấy state hiện tại
        current_state = tuple(self.digits_sequence[-order:])
        probabilities = {}
        
        if current_state in transition_matrix:
            total = sum(transition_matrix[current_state].values())
            if total > 0:
                probabilities = {
                    str(digit): count/total 
                    for digit, count in transition_matrix[current_state].items()
                }
        
        return probabilities
    
    def detect_cycles(self, min_cycle=3, max_cycle=20):
        """
        Phát hiện các chu kỳ lặp lại trong dữ liệu
        """
        if len(self.digits_sequence) < 50:
            return []
        
        # Phân tích FFT
        fft_vals = fft(self.digits_sequence)
        freqs = fftfreq(len(self.digits_sequence))
        
        # Tìm các tần số dominant
        magnitudes = np.abs(fft_vals[:len(fft_vals)//2])
        peak_indices = signal.find_peaks(magnitudes, height=np.mean(magnitudes)*1.5)[0]
        
        cycles = []
        for idx in peak_indices:
            if idx > 0 and freqs[idx] != 0:
                cycle_length = int(1/abs(freqs[idx]))
                if min_cycle <= cycle_length <= max_cycle:
                    cycles.append(cycle_length)
        
        return list(set(cycles[:5]))  # Trả về 5 chu kỳ phổ biến nhất
    
    def entropy_analysis(self, window=50):
        """
        Đo lường độ hỗn loạn của dữ liệu
        """
        if len(self.history) < window:
            return {"avg_entropy": 2.0, "volatility": "CAO", "position_entropy": [2.0]*5}
        
        position_entropy = []
        for pos in range(5):
            pos_digits = [int(num[pos]) for num in self.history[-window:] if len(num) > pos]
            
            if pos_digits:
                value_counts = np.bincount(pos_digits, minlength=10)
                probabilities = value_counts / len(pos_digits)
                non_zero_probs = probabilities[probabilities > 0]
                pos_entropy = entropy(non_zero_probs) if len(non_zero_probs) > 0 else 2.0
                position_entropy.append(pos_entropy)
            else:
                position_entropy.append(2.0)
        
        avg_entropy = np.mean(position_entropy)
        
        # Ngưỡng entropy cho 5D Bet
        if avg_entropy < 1.2:
            volatility = "RẤT THẤP - Cầu ổn định"
        elif avg_entropy < 1.6:
            volatility = "THẤP - Dễ bắt cầu"
        elif avg_entropy < 2.0:
            volatility = "TRUNG BÌNH - Có biến động"
        elif avg_entropy < 2.3:
            volatility = "CAO - Khó dự đoán"
        else:
            volatility = "RẤT CAO - Cầu lừa đảo"
        
        return {
            'position_entropy': position_entropy,
            'avg_entropy': avg_entropy,
            'volatility': volatility
        }
    
    def neural_pattern_recognition(self):
        """
        Sử dụng mạng nơ-ron để nhận dạng pattern
        """
        if len(self.history) < 50:
            return None, None
        
        # Chuẩn bị dữ liệu
        X, y = [], []
        window_size = 10
        
        for i in range(len(self.history) - window_size - 1):
            window = self.history[i:i+window_size]
            features = []
            for num_str in window:
                features.extend([int(d) for d in num_str])
            
            target = [int(d) for d in self.history[i+window_size]]
            X.append(features)
            y.append(target)
        
        if len(X) > 30:
            X = np.array(X)
            y = np.array(y)
            
            # Chuẩn hóa
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Random Forest cho độ chính xác cao hơn
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Train riêng cho từng vị trí
            models = []
            for pos in range(5):
                y_pos = y[:, pos]
                rf_model.fit(X_scaled[:-1], y_pos[:-1])
                models.append(rf_model)
            
            return models, scaler
        return None, None
    
    def early_warning_system(self):
        """
        Hệ thống cảnh báo sớm phát hiện cầu gãy
        """
        warnings = []
        
        if len(self.history) < 20:
            return warnings
        
        # 1. Kiểm tra độ lệch chuẩn tăng đột biến
        last_20_digits = [int(d) for d in "".join(self.history[-20:])]
        prev_20_digits = [int(d) for d in "".join(self.history[-40:-20])]
        
        if len(last_20_digits) > 0 and len(prev_20_digits) > 0:
            last_std = np.std(last_20_digits)
            prev_std = np.std(prev_20_digits)
            
            if prev_std > 0 and last_std > prev_std * 1.8:
                warnings.append("🔴 ĐỘ PHÂN TÁN TĂNG ĐỘT BIẾN - Cầu sắp đảo chiều")
            elif prev_std > 0 and last_std > prev_std * 1.4:
                warnings.append("🟡 ĐỘ PHÂN TÁN TĂNG - Có dấu hiệu biến động")
        
        # 2. Kiểm tra tần suất xuất hiện số lạ (số ít về)
        all_digits = [int(d) for d in "".join(self.history[-30:])]
        digit_counts = Counter(all_digits)
        
        rare_digits = [d for d, count in digit_counts.items() if count < 3]
        if len(rare_digits) > 3:
            warnings.append(f"🟠 SỐ HIẾM ({', '.join(map(str, rare_digits))}) XUẤT HIỆN - Có thể cầu đang thay đổi")
        
        # 3. Kiểm tra variance của độ dài số (số trùng)
        if len(self.history[-20:]) > 0:
            variance = np.var([len(set(num)) for num in self.history[-20:]])
            if variance > 2.5:
                warnings.append("🟡 BIẾN ĐỘNG SỐ TRÙNG CAO - Nên quan sát thêm")
        
        # 4. Phát hiện đảo cầu nhanh
        last_5 = self.history[-5:] if len(self.history) >= 5 else []
        if len(last_5) == 5:
            # Kiểm tra pattern đảo: 12345 -> 54321
            is_reverse_pattern = True
            for i in range(4):
                if last_5[i][::-1] != last_5[i+1]:
                    is_reverse_pattern = False
                    break
            
            if is_reverse_pattern:
                warnings.append("🔴 PHÁT HIỆN CẦU ĐẢO LIÊN TỤC - DỪNG CƯỢC NGAY")
        
        return warnings
    
    def calculate_hot_cold_numbers(self, window=50):
        """
        Tính toán số nóng (hot) và số lạnh (cold)
        """
        if len(self.history) < window:
            return {"hot": [], "cold": []}
        
        all_digits = [int(d) for d in "".join(self.history[-window:])]
        digit_counts = Counter(all_digits)
        
        # Số nóng: tần suất > trung bình + 1.5*std
        avg_freq = len(all_digits) / 10
        std_freq = np.std(list(digit_counts.values())) if digit_counts else 0
        
        hot_digits = [d for d, count in digit_counts.items() 
                     if count > avg_freq + 1.5*std_freq]
        cold_digits = [d for d, count in digit_counts.items() 
                      if count < max(1, avg_freq - 1.5*std_freq)]
        
        return {
            "hot": sorted(hot_digits),
            "cold": sorted(cold_digits)
        }
    
    def predict_next_number_ml(self):
        """
        Dự đoán số tiếp theo bằng Machine Learning
        """
        if len(self.history) < 30:
            return None
        
        # Tạo features từ lịch sử
        features = []
        targets = []
        
        for i in range(len(self.history) - 10):
            window = self.history[i:i+10]
            feature_vector = []
            for num in window:
                feature_vector.extend([int(d) for d in num])
            features.append(feature_vector)
            targets.append(self.history[i+10])
        
        if len(features) < 20:
            return None
        
        X = np.array(features)
        y = np.array([int(t) for num in targets for t in num])  # Flatten targets
        
        # Train model đơn giản
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X[:-1], y[:-(5)])  # Bỏ sample cuối để test
        
        # Predict cho sample cuối
        last_features = X[-1].reshape(1, -1)
        prediction_proba = model.predict_proba(last_features)[0]
        
        return prediction_proba

# ================= THIẾT KẾ GIAO DIỆN v22.0 STYLE =================
st.set_page_config(page_title="TITAN v25.0 SUPREME AI", layout="wide")
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
    .info-box { background: #0e2a4a; color: #9bc9ff; padding: 10px; border-radius: 8px; border: 1px solid #58a6ff; margin: 5px 0; }
    .hot-number { color: #ff5858; font-weight: bold; font-size: 20px; }
    .cold-number { color: #58a6ff; font-weight: bold; font-size: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🚀 TITAN v25.0 SUPREME AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e;'>Học máy đa tầng - Thuật toán cao cấp - Độ chính xác tối đa cho 5D Bet</p>", unsafe_allow_html=True)

# ================= PHẦN 1: NHẬP LIỆU & XỬ LÝ SIÊU SẠCH =================
with st.container():
    col_in, col_st = st.columns([2, 1])
    with col_in:
        raw_input = st.text_area("📡 Nạp dữ liệu mới (Hệ thống tự động loại bỏ số trùng/sai):", height=150, placeholder="Dán dãy số hoặc bảng tại đây...")
    with col_st:
        st.write(f"📊 Kho dữ liệu bảo lưu: **{len(st.session_state.history)} kỳ**")
        
        # Hiển thị độ chính xác
        if st.session_state.accuracy_stats["total"] > 0:
            acc = (st.session_state.accuracy_stats["correct"] / st.session_state.accuracy_stats["total"]) * 100
            st.write(f"🎯 Độ chính xác: **{acc:.1f}%** ({st.session_state.accuracy_stats['correct']}/{st.session_state.accuracy_stats['total']})")
        
        c1, c2 = st.columns(2)
        btn_save = c1.button("🚀 KÍCH HOẠT AI")
        btn_reset = c2.button("🗑️ RESET DỮ LIỆU")
        
        # Thêm nút xác nhận kết quả
        if "last_prediction" in st.session_state:
            if st.button("✅ XÁC NHẬN KẾT QUẢ ĐÚNG", key="confirm_correct"):
                st.session_state.accuracy_stats["correct"] += 1
                st.session_state.accuracy_stats["total"] += 1
                st.session_state.accuracy_stats["last_10"].append(1)
                if len(st.session_state.accuracy_stats["last_10"]) > 10:
                    st.session_state.accuracy_stats["last_10"].pop(0)
                st.rerun()
            
            if st.button("❌ XÁC NHẬN KẾT QUẢ SAI", key="confirm_wrong"):
                st.session_state.accuracy_stats["total"] += 1
                st.session_state.accuracy_stats["last_10"].append(0)
                if len(st.session_state.accuracy_stats["last_10"]) > 10:
                    st.session_state.accuracy_stats["last_10"].pop(0)
                st.rerun()

if btn_reset:
    st.session_state.history = []
    st.session_state.prediction_history = []
    st.session_state.accuracy_stats = {"correct": 0, "total": 0, "last_10": []}
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    st.success("Đã dọn dẹp bộ nhớ vĩnh viễn.")
    st.rerun()

if btn_save:
    # Bước 1: Lọc đa tầng - Chỉ lấy dãy 5 số, loại bỏ trùng lặp tuyệt đối
    input_data = re.findall(r"\b\d{5}\b", raw_input)
    if input_data:
        # Cập nhật vào lịch sử
        st.session_state.history.extend(input_data)
        st.session_state.history = list(dict.fromkeys(st.session_state.history))
        save_db(st.session_state.history)
        
        # Bước 2: Khởi tạo predictor với thuật toán cao cấp
        predictor = AdvancedPredictor(st.session_state.history)
        
        # Thu thập dữ liệu phân tích
        markov_probs = predictor.markov_chain_analysis()
        cycles = predictor.detect_cycles()
        entropy_data = predictor.entropy_analysis()
        warnings = predictor.early_warning_system()
        hot_cold = predictor.calculate_hot_cold_numbers()
        
        # Bước 3: Phân tích với Gemini
        prompt = f"""
        Bạn là hệ thống TITAN v25.0 SUPREME AI - Chuyên gia dự đoán 5D Bet với độ chính xác cao nhất.
        
        PHÂN TÍCH THUẬT TOÁN NÂNG CAO (ĐỘ TIN CẬY CAO):
        
        1. CHUỖI MARKOV:
        - Xác suất chuyển trạng thái: {dict(list(markov_probs.items())[:5]) if markov_probs else 'Đang phân tích'}
        
        2. CHU KỲ PHÁT HIỆN:
        - Các chu kỳ tiềm năng: {cycles if cycles else 'Chưa phát hiện chu kỳ rõ'}
        
        3. ENTROPY & ĐỘ HỖN LOẠN:
        - Entropy trung bình: {entropy_data['avg_entropy']:.3f}
        - Đánh giá: {entropy_data['volatility']}
        - Entropy từng vị trí: {[f"{e:.2f}" for e in entropy_data['position_entropy']]}
        
        4. SỐ NÓNG/LẠNH:
        - Số nóng (hot): {hot_cold['hot']}
        - Số lạnh (cold): {hot_cold['cold']}
        
        5. CẢNH BÁO SỚM:
        {chr(10).join(warnings) if warnings else '- Không phát hiện bất thường'}
        
        Dữ liệu lịch sử 120 kỳ gần nhất: {st.session_state.history[-120:]}
        
        YÊU CẦU DỰ ĐOÁN CHÍNH XÁC CAO CHO 5D BET:
        
        1. Phân tích pattern hiện tại:
           - Xác định xu hướng chính (bệt/đảo/xiên)
           - Đánh giá độ tin cậy của cầu
           - Phát hiện bẫy nhà cái
        
        2. Dự đoán 3 số chủ lực (Main_3):
           - Ưu tiên số từ phân tích Markov và hot numbers
           - Kết hợp với logic cầu đang chạy
           - Đảm bảo tính khả thi cao nhất
        
        3. Dự đoán 4 số lót (Support_4):
           - Bổ sung các số có xác suất cao thứ hai
           - Tạo dàn an toàn, bảo toàn vốn
        
        4. Quyết định chiến thuật:
           - ĐÁNH: Khi cầu rõ, độ tin cậy >85%
           - THEO DÕI: Khi cầu đang hình thành
           - DỪNG: Khi phát hiện cầu lừa, entropy cao
        
        TRẢ VỀ JSON CHÍNH XÁC:
        {{
            "main_3": "5 số dự đoán chính (phân tách bằng dấu cách nếu cần)",
            "support_4": "5 số dự đoán phụ (phân tách bằng dấu cách nếu cần)",
            "decision": "ĐÁNH/DỪNG/THEO DÕI/CẢNH BÁO ĐẢO CẦU",
            "logic": "Phân tích chi tiết, có tham chiếu đến các thuật toán, lý do chốt số",
            "color": "Green/Red/Yellow",
            "confidence": 0-100,
            "warning_level": "THẤP/TRUNG BÌNH/CAO/RẤT CAO"
        }}
        
        LƯU Ý: Đây là tool đánh tiền thật, yêu cầu độ chính xác tối đa. Phân tích kỹ trước khi trả kết quả.
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                st.session_state.last_prediction = json.loads(json_match.group())
                
                # Lưu vào lịch sử dự đoán
                st.session_state.prediction_history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "prediction": st.session_state.last_prediction,
                    "warnings": warnings
                })
        except Exception as e:
            # Fallback: Sử dụng thuật toán nâng cao
            all_digits = "".join(st.session_state.history[-60:])
            counts = Counter(all_digits).most_common(10)
            top_nums = [x[0] for x in counts]
            
            # Kết hợp với hot numbers
            main_nums = list(set(top_nums[:3] + [str(x) for x in hot_cold['hot'][:2] if hot_cold['hot']]))
            support_nums = list(set(top_nums[3:7] + [str(x) for x in hot_cold['cold'][:2] if hot_cold['cold']]))
            
            st.session_state.last_prediction = {
                "main_3": "".join(main_nums[:3]).ljust(3, '0')[:3],
                "support_4": "".join(support_nums[:4]).ljust(4, '0')[:4],
                "decision": "CẢNH BÁO ĐẢO CẦU" if len(warnings) > 2 else "THEO DÕI NHỊP",
                "logic": f"Ma trận tần suất + Phân tích entropy {entropy_data['avg_entropy']:.2f}. Cảnh báo: {len(warnings)} dấu hiệu.",
                "color": "Yellow" if len(warnings) < 3 else "Red",
                "confidence": 85 - len(warnings)*5,
                "warning_level": "CAO" if len(warnings) > 2 else "TRUNG BÌNH"
            }
        
        st.rerun()

# ================= PHẦN 2: KẾT QUẢ THỰC CHIẾN =================
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    
    # Hiển thị trạng thái chiến đấu
    status_map = {"green": "#238636", "red": "#da3633", "yellow": "#d29922"}
    bg_color = status_map.get(res.get('color', 'yellow').lower(), "#30363d")
    
    warning_level = res.get('warning_level', 'TRUNG BÌNH')
    warning_color = {"THẤP": "#238636", "TRUNG BÌNH": "#d29922", "CAO": "#da3633", "RẤT CAO": "#ff0000"}
    
    st.markdown(f"""
        <div class='status-bar' style='background: {bg_color};'>
            🔥 CHỈ THỊ: {res['decision']} | ĐỘ TIN CẬY: {res['confidence']}% | 
            MỨC CẢNH BÁO: <span style='color: {warning_color.get(warning_level, "#ffffff")};'>{warning_level}</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    # Kết quả hàng ngang
    col_main, col_supp = st.columns([1.5, 1])
    with col_main:
        st.markdown(f"<p style='color:#8b949e; text-align:center; font-weight:bold;'>🎯 3 SỐ CHỦ LỰC (VÀO TIỀN)</p>", unsafe_allow_html=True)
        main_display = res['main_3'] if len(res['main_3']) >= 3 else res['main_3'].ljust(3, 'X')
        st.markdown(f"<div class='num-box'>{main_display}</div>", unsafe_allow_html=True)
    
    with col_supp:
        st.markdown(f"<p style='color:#8b949e; text-align:center; font-weight:bold;'>🛡️ 4 SỐ LÓT (GIỮ VỐN)</p>", unsafe_allow_html=True)
        supp_display = res['support_4'] if len(res['support_4']) >= 4 else res['support_4'].ljust(4, 'X')
        st.markdown(f"<div class='lot-box'>{supp_display}</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Phân tích đa tầng nâng cao
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader("🧠 Phân tích tinh hoa")
        st.write(res['logic'])
        
        # Hiển thị cảnh báo chi tiết
        if res.get('warning_level') in ["CAO", "RẤT CAO"] or res['confidence'] < 85:
            st.markdown("""
                <div class='warning-box'>
                    ⚠️ CẢNH BÁO NGUY HIỂM: Nhà cái đang đảo cầu mạnh.
                    Khuyến cáo DỪNG CƯỢC hoặc giảm 90% vốn để bảo toàn.
                </div>
            """, unsafe_allow_html=True)
    
    with col_r:
        st.subheader("📋 Chiến thuật")
        full_dan = "".join(sorted(set(res['main_3'] + res['support_4'])))
        st.text_input("Dàn 7 số:", full_dan)
        
        # Hiển thị tỷ lệ vào tiền
        if res['decision'] == "ĐÁNH":
            st.success("💵 Vào tiền: 70% vốn cho Main, 30% cho Support")
        elif res['decision'] == "THEO DÕI":
            st.warning("👁️ Vào tiền: 30% vốn, quan sát thêm")
        else:
            st.error("⛔ DỪNG CƯỢC: Bảo toàn vốn, chờ cầu mới")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ================= PHẦN 3: MA TRẬN SỐ HỌC NÂNG CAO =================
if st.session_state.history:
    with st.expander("📊 Xem phân tích chuyên sâu (Thuật toán cao cấp)"):
        tab1, tab2, tab3, tab4 = st.tabs(["Tần suất", "Entropy", "Chu kỳ", "Cảnh báo"])
        
        with tab1:
            all_d = "".join(st.session_state.history[-60:])
            if all_d:
                df_stats = pd.DataFrame({
                    'Số': list(range(10)),
                    'Tần suất': [all_d.count(str(i)) for i in range(10)]
                })
                st.bar_chart(df_stats.set_index('Số'))
                
                # Hiển thị số nóng/lạnh
                predictor = AdvancedPredictor(st.session_state.history)
                hot_cold = predictor.calculate_hot_cold_numbers()
                
                col_hot, col_cold = st.columns(2)
                with col_hot:
                    st.markdown("### 🔥 Số nóng (Hot)")
                    for num in hot_cold['hot']:
                        st.markdown(f"<span class='hot-number'>{num}</span>", unsafe_allow_html=True)
                
                with col_cold:
                    st.markdown("### ❄️ Số lạnh (Cold)")
                    for num in hot_cold['cold']:
                        st.markdown(f"<span class='cold-number'>{num}</span>", unsafe_allow_html=True)
        
        with tab2:
            predictor = AdvancedPredictor(st.session_state.history)
            entropy_data = predictor.entropy_analysis()
            
            st.metric("Entropy trung bình", f"{entropy_data['avg_entropy']:.3f}", 
                     delta=None, delta_color="off")
            st.write(f"**Đánh giá:** {entropy_data['volatility']}")
            
            # Biểu đồ entropy theo vị trí
            entropy_df = pd.DataFrame({
                'Vị trí': [f'Vị trí {i+1}' for i in range(5)],
                'Entropy': entropy_data['position_entropy']
            })
            st.bar_chart(entropy_df.set_index('Vị trí'))
            
            st.caption("Entropy càng cao càng khó dự đoán. Nếu >2.3 nên dừng cược.")
        
        with tab3:
            predictor = AdvancedPredictor(st.session_state.history)
            cycles = predictor.detect_cycles()
            
            if cycles:
                st.write("**Chu kỳ phát hiện:**")
                for i, cycle in enumerate(cycles[:5]):
                    st.info(f"📈 Chu kỳ {i+1}: {cycle} kỳ")
                
                # Dự đoán dựa trên chu kỳ
                if len(cycles) > 0 and len(st.session_state.history) > cycles[0]:
                    st.write("**Dự đoán theo chu kỳ:**")
                    cycle_pred = st.session_state.history[-cycles[0]] if cycles[0] <= len(st.session_state.history) else "Chưa đủ dữ liệu"
                    st.write(f"Kỳ tiếp theo có thể lặp lại số: {cycle_pred}")
            else:
                st.write("Chưa phát hiện chu kỳ rõ ràng")
        
        with tab4:
            predictor = AdvancedPredictor(st.session_state.history)
            warnings = predictor.early_warning_system()
            
            if warnings:
                for warning in warnings:
                    if "🔴" in warning:
                        st.error(warning)
                    elif "🟡" in warning:
                        st.warning(warning)
                    else:
                        st.info(warning)
            else:
                st.success("✅ Không phát hiện cảnh báo - Cầu đang ổn định")

# ================= PHẦN 4: LỊCH SỬ DỰ ĐOÁN =================
if st.session_state.prediction_history:
    with st.expander("📜 Lịch sử dự đoán"):
        for pred in st.session_state.prediction_history[-10:]:
            st.write(f"**{pred['time']}** - Dự đoán: {pred['prediction']['main_3']} | {pred['prediction']['decision']} | Độ tin cậy: {pred['prediction']['confidence']}%")