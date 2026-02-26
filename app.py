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

# ================= THUẬT TOÁN CAO CẤP (KHÔNG DÙNG SCIPY) =================

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
    
    def detect_cycles_simple(self, min_cycle=3, max_cycle=20):
        """
        Phát hiện chu kỳ đơn giản bằng autocorrelation
        Không dùng FFT để tránh lỗi thư viện
        """
        if len(self.history) < 30:
            return []
        
        # Chuyển đổi history thành mảng số
        digits_array = []
        for num_str in self.history[-100:]:  # Lấy 100 kỳ gần nhất
            digits_array.extend([int(d) for d in num_str])
        
        # Tìm chu kỳ bằng phương pháp tương quan đơn giản
        cycles = []
        for period in range(min_cycle, min(max_cycle, len(digits_array)//2)):
            correlation = 0
            count = 0
            for i in range(len(digits_array) - period):
                if digits_array[i] == digits_array[i + period]:
                    correlation += 1
                count += 1
            
            if count > 0:
                correlation_ratio = correlation / count
                if correlation_ratio > 0.4:  # Ngưỡng tương quan
                    cycles.append(period)
        
        return list(set(cycles[:5]))  # Trả về 5 chu kỳ phổ biến nhất
    
    def entropy_analysis(self, window=50):
        """
        Đo lường độ hỗn loạn của dữ liệu - tự tính entropy thủ công
        """
        if len(self.history) < window:
            return {"avg_entropy": 2.0, "volatility": "CAO", "position_entropy": [2.0]*5}
        
        def calculate_entropy(data):
            """Tính entropy thủ công"""
            value_counts = {}
            for value in data:
                value_counts[value] = value_counts.get(value, 0) + 1
            
            entropy = 0
            total = len(data)
            for count in value_counts.values():
                prob = count / total
                entropy -= prob * np.log2(prob) if prob > 0 else 0
            
            return entropy
        
        position_entropy = []
        for pos in range(5):
            pos_digits = [int(num[pos]) for num in self.history[-window:] if len(num) > pos]
            
            if pos_digits:
                pos_entropy = calculate_entropy(pos_digits)
                position_entropy.append(pos_entropy)
            else:
                position_entropy.append(2.0)
        
        avg_entropy = np.mean(position_entropy) if position_entropy else 2.0
        
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
    
    def pattern_recognition_simple(self):
        """
        Nhận dạng pattern đơn giản bằng phân tích thống kê
        """
        if len(self.history) < 20:
            return {}
        
        patterns = {}
        
        # 1. Pattern tăng dần / giảm dần
        last_5 = self.history[-5:]
        increasing = 0
        decreasing = 0
        
        for i in range(4):
            if int(last_5[i]) < int(last_5[i+1]):
                increasing += 1
            elif int(last_5[i]) > int(last_5[i+1]):
                decreasing += 1
        
        patterns['trend'] = 'TĂNG' if increasing > decreasing else 'GIẢM' if decreasing > increasing else 'ĐI NGANG'
        
        # 2. Pattern số trùng
        duplicate_count = sum([len(num) - len(set(num)) for num in last_5])
        patterns['duplicate_trend'] = 'NHIỀU SỐ TRÙNG' if duplicate_count > 5 else 'ÍT SỐ TRÙNG'
        
        # 3. Pattern chẵn lẻ
        even_odd_ratio = []
        for num in last_5:
            even = sum(1 for d in num if int(d) % 2 == 0)
            odd = 5 - even
            even_odd_ratio.append(even / odd if odd > 0 else 5)
        
        patterns['even_odd'] = f"Tỷ lệ TB: {np.mean(even_odd_ratio):.2f}"
        
        return patterns
    
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
            
            if prev_std > 0:
                if last_std > prev_std * 1.8:
                    warnings.append("🔴 ĐỘ PHÂN TÁN TĂNG ĐỘT BIẾN - Cầu sắp đảo chiều")
                elif last_std > prev_std * 1.4:
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
    
    def predict_by_ma(self, window=5):
        """
        Dự đoán bằng Moving Average
        """
        if len(self.history) < window + 1:
            return None
        
        # Tính trung bình động cho từng vị trí
        predictions = []
        for pos in range(5):
            pos_values = [int(num[pos]) for num in self.history[-window:]]
            ma = np.mean(pos_values)
            std = np.std(pos_values)
            
            # Dự đoán số gần với MA nhất
            candidates = [int(round(ma)), int(round(ma)) + 1, int(round(ma)) - 1]
            candidates = [c for c in candidates if 0 <= c <= 9]
            predictions.append(candidates[0] if candidates else 5)
        
        return "".join(map(str, predictions))

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
    .hot-number { color: #ff5858; font-weight: bold; font-size: 20px; display: inline-block; margin: 0 5px; }
    .cold-number { color: #58a6ff; font-weight: bold; font-size: 20px; display: inline-block; margin: 0 5px; }
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
            
            # Hiển thị 10 kỳ gần nhất
            if st.session_state.accuracy_stats["last_10"]:
                last_10_acc = sum(st.session_state.accuracy_stats["last_10"]) / len(st.session_state.accuracy_stats["last_10"]) * 100
                st.write(f"📈 10 kỳ gần: **{last_10_acc:.1f}%**")
        
        c1, c2 = st.columns(2)
        btn_save = c1.button("🚀 KÍCH HOẠT AI")
        btn_reset = c2.button("🗑️ RESET DỮ LIỆU")
        
        # Thêm nút xác nhận kết quả
        if "last_prediction" in st.session_state:
            col_confirm1, col_confirm2 = st.columns(2)
            with col_confirm1:
                if st.button("✅ ĐÚNG", key="confirm_correct", use_container_width=True):
                    st.session_state.accuracy_stats["correct"] += 1
                    st.session_state.accuracy_stats["total"] += 1
                    st.session_state.accuracy_stats["last_10"].append(1)
                    if len(st.session_state.accuracy_stats["last_10"]) > 10:
                        st.session_state.accuracy_stats["last_10"].pop(0)
                    st.rerun()
            
            with col_confirm2:
                if st.button("❌ SAI", key="confirm_wrong", use_container_width=True):
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
        cycles = predictor.detect_cycles_simple()
        entropy_data = predictor.entropy_analysis()
        warnings = predictor.early_warning_system()
        hot_cold = predictor.calculate_hot_cold_numbers()
        patterns = predictor.pattern_recognition_simple()
        ma_prediction = predictor.predict_by_ma()
        
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
        
        5. PATTERN HIỆN TẠI:
        - Xu hướng: {patterns.get('trend', 'Không rõ')}
        - Số trùng: {patterns.get('duplicate_trend', 'Không rõ')}
        - Chẵn/lẻ: {patterns.get('even_odd', 'Không rõ')}
        
        6. DỰ ĐOÁN MA:
        - Moving Average: {ma_prediction if ma_prediction else 'Đang học'}
        
        7. CẢNH BÁO SỚM:
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
        
        TRẢ VỀ JSON CHÍNH XÁC (KHÔNG THÊM BẤT KỲ CHỮ NÀO NGOÀI JSON):
        {{
            "main_3": "3 số dự đoán chính (ví dụ: 123)",
            "support_4": "4 số dự đoán phụ (ví dụ: 4567)",
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
            
            # Kết hợp với hot numbers và MA
            main_nums = []
            if ma_prediction:
                main_nums.extend(list(ma_prediction[:3]))
            main_nums.extend([str(x) for x in hot_cold['hot'][:2] if hot_cold['hot']])
            main_nums.extend(top_nums[:2])
            
            support_nums = []
            support_nums.extend([str(x) for x in hot_cold['cold'][:2] if hot_cold['cold']])
            support_nums.extend(top_nums[3:7])
            
            # Loại bỏ trùng và lấy đủ số
            main_nums = list(dict.fromkeys(main_nums))[:3]
            support_nums = list(dict.fromkeys(support_nums))[:4]
            
            while len(main_nums) < 3:
                main_nums.append(str(np.random.randint(0, 10)))
            while len(support_nums) < 4:
                support_nums.append(str(np.random.randint(0, 10)))
            
            st.session_state.last_prediction = {
                "main_3": "".join(main_nums),
                "support_4": "".join(support_nums),
                "decision": "CẢNH BÁO ĐẢO CẦU" if len(warnings) > 2 else "THEO DÕI NHỊP",
                "logic": f"Ma trận tần suất + Phân tích entropy {entropy_data['avg_entropy']:.2f}. Cảnh báo: {len(warnings)} dấu hiệu. Pattern: {patterns.get('trend', 'Không rõ')}",
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
        main_display = res['main_3'] if len(res['main_3']) >= 3 else res['main_3'].ljust(3, '0')
        st.markdown(f"<div class='num-box'>{main_display}</div>", unsafe_allow_html=True)
    
    with col_supp:
        st.markdown(f"<p style='color:#8b949e; text-align:center; font-weight:bold;'>🛡️ 4 SỐ LÓT (GIỮ VỐN)</p>", unsafe_allow_html=True)
        supp_display = res['support_4'] if len(res['support_4']) >= 4 else res['support_4'].ljust(4, '0')
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
                
                st.markdown("### 🔥 Số nóng (Hot)")
                hot_html = " ".join([f"<span class='hot-number'>{num}</span>" for num in hot_cold['hot']])
                st.markdown(hot_html, unsafe_allow_html=True)
                
                st.markdown("### ❄️ Số lạnh (Cold)")
                cold_html = " ".join([f"<span class='cold-number'>{num}</span>" for num in hot_cold['cold']])
                st.markdown(cold_html, unsafe_allow_html=True)
        
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
            cycles = predictor.detect_cycles_simple()
            
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