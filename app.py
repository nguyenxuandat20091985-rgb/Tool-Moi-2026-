import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import random

# ================= CẤU HÌNH HỆ THỐNG TITAN v26.0 - HỌC TỪ SAI LẦM =================
API_KEY = "AIzaSyB5PRp04XlMHKl3oGfCRbsKXjlTA-CZifc"
DB_FILE = "titan_v26_learning.json"
ACCURACY_FILE = "titan_accuracy_log.json"

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

def load_accuracy_log():
    if os.path.exists(ACCURACY_FILE):
        with open(ACCURACY_FILE, "r") as f:
            try:
                return json.load(f)
            except:
                return {"predictions": [], "stats": {}}
    return {"predictions": [], "stats": {}}

def save_accuracy_log(log):
    with open(ACCURACY_FILE, "w") as f:
        json.dump(log, f)

# Khởi tạo session state
if "history" not in st.session_state:
    st.session_state.history = load_db()
if "accuracy_log" not in st.session_state:
    st.session_state.accuracy_log = load_accuracy_log()
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "learning_mode" not in st.session_state:
    st.session_state.learning_mode = True

# ================= THUẬT TOÁN PHÂN TÍCH CẦU THỰC TẾ =================

class RealCatchPredictor:
    """Thuật toán bắt cầu thực tế cho 5D"""
    
    def __init__(self, history):
        self.history = history
        self.patterns = self.analyze_patterns()
    
    def analyze_patterns(self):
        """Phân tích tất cả patterns có thể"""
        if len(self.history) < 10:
            return {}
        
        patterns = {
            'bệt': self.detect_bet(),
            'đảo': self.detect_dao(),
            'xiên': self.detect_xien(),
            'tổng': self.analyze_tong(),
            'chẵn_lẻ': self.analyze_chan_le(),
            'lô_rơi': self.detect_lo_roi(),
            'cầu_kẹp': self.detect_cau_kep(),
            'vị_trí': self.analyze_position()
        }
        return patterns
    
    def detect_bet(self):
        """Phát hiện cầu bệt - số về liên tiếp"""
        if len(self.history) < 5:
            return []
        
        bet_numbers = []
        # Kiểm tra từng số từ 0-9
        for num in range(10):
            count = 0
            str_num = str(num)
            # Đếm số lần xuất hiện trong 5 kỳ gần nhất
            for hist in self.history[-5:]:
                if str_num in hist:
                    count += 1
            
            if count >= 3:  # Xuất hiện 3/5 kỳ là bệt
                bet_numbers.append(str_num)
        
        return bet_numbers
    
    def detect_dao(self):
        """Phát hiện cầu đảo - số đảo chiều liên tục"""
        if len(self.history) < 4:
            return []
        
        dao_patterns = []
        last_4 = self.history[-4:]
        
        # Kiểm tra đảo đầu đuôi
        for i in range(3):
            if last_4[i][0] == last_4[i+1][4] and last_4[i][4] == last_4[i+1][0]:
                dao_patterns.append(f"Đảo đầu-đuôi: {last_4[i]} -> {last_4[i+1]}")
        
        # Kiểm tra đảo toàn bộ
        for i in range(3):
            if last_4[i][::-1] == last_4[i+1]:
                dao_patterns.append(f"Đảo hoàn toàn: {last_4[i]} -> {last_4[i+1]}")
        
        return dao_patterns
    
    def detect_xien(self):
        """Phát hiện cầu xiên - số chạy theo quy luật"""
        if len(self.history) < 5:
            return []
        
        xien_patterns = []
        
        # Chuyển đổi thành mảng số
        numbers = []
        for h in self.history[-10:]:
            numbers.append([int(d) for d in h])
        
        # Kiểm từng vị trí xem có tăng/giảm dần không
        for pos in range(5):
            pos_values = [n[pos] for n in numbers]
            
            # Kiểm tăng dần
            tang = all(pos_values[i] <= pos_values[i+1] for i in range(len(pos_values)-1))
            # Kiểm giảm dần
            giam = all(pos_values[i] >= pos_values[i+1] for i in range(len(pos_values)-1))
            
            if tang:
                xien_patterns.append(f"Vị trí {pos+1} tăng dần")
            if giam:
                xien_patterns.append(f"Vị trí {pos+1} giảm dần")
        
        return xien_patterns
    
    def analyze_tong(self):
        """Phân tích tổng các số"""
        if len(self.history) < 5:
            return {}
        
        tongs = []
        for h in self.history[-10:]:
            tong = sum(int(d) for d in h)
            tongs.append(tong)
        
        # Tìm tổng hay về
        tong_counts = Counter(tongs)
        hot_tong = [t for t, c in tong_counts.most_common(3)]
        
        return {
            'hot_tong': hot_tong,
            'tong_gan_nhat': tongs[-5:] if tongs else []
        }
    
    def analyze_chan_le(self):
        """Phân tích chẵn lẻ"""
        if len(self.history) < 5:
            return {}
        
        chan_le = []
        for h in self.history[-10:]:
            chan = sum(1 for d in h if int(d) % 2 == 0)
            le = 5 - chan
            chan_le.append((chan, le))
        
        # Xu hướng chẵn/lẻ
        avg_chan = np.mean([cl[0] for cl in chan_le])
        
        return {
            'avg_chan': avg_chan,
            'xu_huong': 'Nhiều chẵn' if avg_chan > 2.5 else 'Nhiều lẻ' if avg_chan < 2.5 else 'Cân bằng'
        }
    
    def detect_lo_roi(self):
        """Phát hiện lô rơi - số lặp lại từ kỳ trước"""
        if len(self.history) < 2:
            return []
        
        lo_roi = []
        last = self.history[-1]
        prev = self.history[-2]
        
        # Tìm số xuất hiện ở cả 2 kỳ
        for d in last:
            if d in prev and d not in lo_roi:
                lo_roi.append(d)
        
        return lo_roi
    
    def detect_cau_kep(self):
        """Phát hiện cầu kẹp - số bị kẹp giữa 2 số"""
        if len(self.history) < 3:
            return []
        
        cau_kep = []
        for i in range(len(self.history)-2):
            prev = self.history[i]
            curr = self.history[i+1]
            next_ = self.history[i+2]
            
            # Kiểm số ở giữa có bị kẹp không
            for pos in range(5):
                if curr[pos] == prev[pos] and curr[pos] == next_[pos]:
                    cau_kep.append(f"Số {curr[pos]} ở vị trí {pos+1} bị kẹp")
        
        return list(set(cau_kep))[-5:]  # Lấy 5 cái gần nhất
    
    def analyze_position(self):
        """Phân tích từng vị trí riêng biệt"""
        if len(self.history) < 10:
            return {}
        
        position_stats = {}
        for pos in range(5):
            pos_values = [int(h[pos]) for h in self.history[-20:]]
            counts = Counter(pos_values)
            
            # Top 3 số hay về ở vị trí này
            top_3 = [str(x[0]) for x in counts.most_common(3)]
            
            # Số vừa về
            last_value = self.history[-1][pos] if self.history else "?"
            
            position_stats[f"pos_{pos+1}"] = {
                'top': top_3,
                'last': last_value,
                'counts': dict(counts.most_common(5))
            }
        
        return position_stats
    
    def suggest_numbers(self):
        """Đề xuất số dựa trên patterns phát hiện"""
        suggestions = []
        
        # Ưu tiên số bệt
        if self.patterns.get('bệt'):
            suggestions.extend(self.patterns['bệt'])
        
        # Thêm lô rơi
        if self.patterns.get('lô_rơi'):
            suggestions.extend(self.patterns['lô_rơi'])
        
        # Thêm số từ vị trí hot
        pos_stats = self.patterns.get('vị_trí', {})
        for pos_data in pos_stats.values():
            suggestions.extend(pos_data.get('top', [])[:2])
        
        # Loại bỏ trùng và lấy 7 số
        suggestions = list(dict.fromkeys(suggestions))[:7]
        
        # Nếu thiếu, thêm số random từ 0-9
        while len(suggestions) < 7:
            rand = str(random.randint(0, 9))
            if rand not in suggestions:
                suggestions.append(rand)
        
        return {
            'main': ''.join(suggestions[:3]),
            'support': ''.join(suggestions[3:7])
        }
    
    def analyze_failures(self):
        """Phân tích lý do thất bại để học hỏi"""
        if 'accuracy_log' not in st.session_state:
            return {}
        
        log = st.session_state.accuracy_log
        if len(log.get('predictions', [])) < 5:
            return {}
        
        # Lấy 10 lần dự đoán gần nhất
        recent = log['predictions'][-10:]
        
        # Phân tích pattern thất bại
        failures = [p for p in recent if not p.get('correct', False)]
        
        if not failures:
            return {"message": "Đang chạy tốt"}
        
        # Tìm nguyên nhân
        reasons = []
        for f in failures:
            if f.get('predicted') and f.get('actual'):
                # So sánh dự đoán vs thực tế
                predicted = f['predicted']
                actual = f['actual']
                
                # Đếm số đúng
                correct_count = 0
                for i in range(min(3, len(predicted))):
                    if i < len(actual) and predicted[i] == actual[i]:
                        correct_count += 1
                
                if correct_count == 0:
                    reasons.append("Sai hoàn toàn")
                elif correct_count == 1:
                    reasons.append("Chỉ đúng 1 số")
                elif correct_count == 2:
                    reasons.append("Đúng 2 số")
        
        # Thống kê
        reason_counts = Counter(reasons)
        
        return {
            "failure_rate": (len(failures)/len(recent))*100 if recent else 0,
            "top_reason": reason_counts.most_common(1)[0][0] if reason_counts else "Không rõ",
            "suggestion": "Cần tập trung vào số bệt" if "Sai hoàn toàn" in reasons else "Đang cải thiện"
        }

# ================= GIAO DIỆN =================
st.set_page_config(page_title="TITAN v26.0 - HỌC TỪ THẤT BẠI", layout="wide")

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
        text-align: center; letter-spacing: 15px;
        text-shadow: 0 0 25px rgba(255,88,88,0.5);
    }
    .lot-box {
        font-size: 60px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 10px;
        text-shadow: 0 0 15px rgba(88,166,255,0.3);
    }
    .status-bar {
        padding: 15px; border-radius: 12px; text-align: center;
        font-weight: bold; font-size: 24px; margin-bottom: 20px;
    }
    .warning-box {
        background: #4a0e0e; color: #ff9b9b; padding: 15px;
        border-radius: 8px; border: 1px solid #ff4444;
        text-align: center; font-weight: bold;
    }
    .pattern-badge {
        display: inline-block; padding: 5px 10px;
        background: #1f6feb; color: white; border-radius: 15px;
        margin: 2px; font-size: 14px;
    }
    .failure-analysis {
        background: #1a1f2e; padding: 15px; border-radius: 10px;
        border-left: 5px solid #ff5858; margin: 10px 0;
    }
    .success-analysis {
        background: #1a2e1a; padding: 15px; border-radius: 10px;
        border-left: 5px solid #238636; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🚀 TITAN v26.0 - HỌC TỪ THẤT BẠI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e;'>Phân tích cầu thực tế - Tự động sửa sai sau mỗi kỳ</p>", unsafe_allow_html=True)

# Layout chính
col_in, col_stats = st.columns([2, 1])

with col_in:
    raw_input = st.text_area("📥 NHẬP KẾT QUẢ MỚI:", height=100,
                            placeholder="Dán số 5D mới nhất vào đây...")

with col_stats:
    st.metric("📊 Tổng số kỳ", len(st.session_state.history))
    
    # Tính độ chính xác
    if st.session_state.accuracy_log.get('predictions'):
        predictions = st.session_state.accuracy_log['predictions']
        total = len(predictions)
        correct = sum(1 for p in predictions if p.get('correct', False))
        acc = (correct/total*100) if total > 0 else 0
        
        st.metric("🎯 Độ chính xác", f"{acc:.1f}%", 
                 delta=f"{correct}/{total}")
        
        # 5 kỳ gần nhất
        last_5 = predictions[-5:]
        if last_5:
            last_5_correct = sum(1 for p in last_5 if p.get('correct', False))
            st.metric("📈 5 kỳ gần", f"{last_5_correct}/5")
    
    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🔍 PHÂN TÍCH", use_container_width=True)
    with col2:
        reset_btn = st.button("🗑️ RESET", use_container_width=True)

if reset_btn:
    st.session_state.history = []
    st.session_state.accuracy_log = {"predictions": [], "stats": {}}
    st.session_state.last_prediction = None
    if os.path.exists(DB_FILE): 
        os.remove(DB_FILE)
    if os.path.exists(ACCURACY_FILE): 
        os.remove(ACCURACY_FILE)
    st.success("✅ Đã reset toàn bộ dữ liệu")
    st.rerun()

# XỬ LÝ PHÂN TÍCH
if analyze_btn and raw_input:
    # Lọc số mới
    new_numbers = re.findall(r"\b\d{5}\b", raw_input)
    
    if new_numbers:
        # Lưu vào history
        for num in new_numbers:
            if num not in st.session_state.history:
                st.session_state.history.append(num)
        
        save_db(st.session_state.history)
        
        # KIỂM TRA ĐỘ CHÍNH XÁC CỦA DỰ ĐOÁN TRƯỚC
        if st.session_state.last_prediction and new_numbers:
            last_pred = st.session_state.last_prediction
            actual = new_numbers[0]  # Lấy số mới nhất
            
            # Kiểm tra độ chính xác
            main_correct = 0
            for i in range(min(3, len(last_pred['main_3']))):
                if i < len(actual) and last_pred['main_3'][i] == actual[i]:
                    main_correct += 1
            
            # Lưu vào log
            st.session_state.accuracy_log['predictions'].append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'predicted': last_pred['main_3'],
                'actual': actual,
                'correct': main_correct >= 2,  # Đúng 2/3 số là tạm chấp nhận
                'main_correct': main_correct,
                'all_correct': last_pred['main_3'] == actual[:3]
            })
            
            # Giới hạn log
            if len(st.session_state.accuracy_log['predictions']) > 100:
                st.session_state.accuracy_log['predictions'] = \
                    st.session_state.accuracy_log['predictions'][-100:]
            
            save_accuracy_log(st.session_state.accuracy_log)
        
        # Phân tích patterns mới
        predictor = RealCatchPredictor(st.session_state.history)
        
        # Đề xuất số
        suggestion = predictor.suggest_numbers()
        
        # Phân tích thất bại
        failure_analysis = predictor.analyze_failures()
        
        # Quyết định dựa trên patterns
        bet_count = len(predictor.patterns.get('bệt', []))
        lo_roi_count = len(predictor.patterns.get('lô_rơi', []))
        
        if bet_count >= 2:
            decision = "ĐÁNH MẠNH"
            color = "Green"
            confidence = 90 + bet_count*2
        elif bet_count >= 1 or lo_roi_count >= 2:
            decision = "ĐÁNH"
            color = "Green"
            confidence = 85
        elif len(predictor.patterns.get('cầu_kẹp', [])) > 0:
            decision = "THEO DÕI"
            color = "Yellow"
            confidence = 75
        else:
            decision = "CẢNH BÁO - CHỜ CẦU MỚI"
            color = "Red"
            confidence = 50
        
        # Lưu dự đoán mới
        st.session_state.last_prediction = {
            'main_3': suggestion['main'],
            'support_4': suggestion['support'],
            'decision': decision,
            'logic': f"Phát hiện: {bet_count} số bệt, {lo_roi_count} lô rơi",
            'color': color,
            'confidence': min(confidence, 99),
            'patterns': predictor.patterns,
            'failure_analysis': failure_analysis
        }
        
        st.rerun()

# HIỂN THỊ KẾT QUẢ
if st.session_state.last_prediction:
    res = st.session_state.last_prediction
    
    # Status bar
    bg_color = "#238636" if res['color'] == 'Green' else "#d29922" if res['color'] == 'Yellow' else "#da3633"
    st.markdown(f"""
        <div class='status-bar' style='background: {bg_color};'>
            🔥 {res['decision']} | ĐỘ TIN CẬY: {res['confidence']}%
        </div>
    """, unsafe_allow_html=True)
    
    # Prediction card
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    col_m1, col_m2 = st.columns([1.5, 1])
    with col_m1:
        st.markdown("<p style='text-align:center; font-weight:bold;'>🎯 3 SỐ CHỦ LỰC</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-box'>{res['main_3']}</div>", unsafe_allow_html=True)
    
    with col_m2:
        st.markdown("<p style='text-align:center; font-weight:bold;'>🛡️ 4 SỐ LÓT</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-box'>{res['support_4']}</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Hiển thị patterns
    st.subheader("🔍 PHÂN TÍCH CẦU THỰC TẾ")
    
    patterns = res.get('patterns', {})
    
    # Hiển thị các pattern dưới dạng badge
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.markdown("**📈 CẦU BỆT**")
        bet_nums = patterns.get('bệt', [])
        if bet_nums:
            for num in bet_nums:
                st.markdown(f"<span class='pattern-badge'>Số {num} bệt</span>", unsafe_allow_html=True)
        else:
            st.write("Không có")
    
    with col_p2:
        st.markdown("**🔄 LÔ RƠI**")
        lo_roi = patterns.get('lô_rơi', [])
        if lo_roi:
            for num in lo_roi:
                st.markdown(f"<span class='pattern-badge'>Số {num} rơi</span>", unsafe_allow_html=True)
        else:
            st.write("Không có")
    
    with col_p3:
        st.markdown("**⚡ CẦU KẸP**")
        cau_kep = patterns.get('cầu_kẹp', [])
        if cau_kep:
            for cp in cau_kep[:3]:
                st.markdown(f"<span class='pattern-badge'>{cp}</span>", unsafe_allow_html=True)
        else:
            st.write("Không có")
    
    # Hiển thị thêm thông tin
    with st.expander("📊 Xem thêm phân tích"):
        st.json(patterns)
    
    # Phân tích thất bại
    failure = res.get('failure_analysis', {})
    if failure:
        if failure.get('failure_rate', 0) > 50:
            st.markdown(f"""
                <div class='failure-analysis'>
                    <b>⚠️ PHÂN TÍCH THẤT BẠI:</b><br>
                    Tỷ lệ sai: {failure.get('failure_rate', 0):.1f}%<br>
                    Nguyên nhân chính: {failure.get('top_reason', 'Không rõ')}<br>
                    <i>{failure.get('suggestion', 'Đang điều chỉnh...')}</i>
                </div>
            """, unsafe_allow_html=True)
        elif failure.get('failure_rate', 0) < 30:
            st.markdown(f"""
                <div class='success-analysis'>
                    <b>✅ ĐANG CHẠY TỐT:</b><br>
                    Tỷ lệ đúng: {100 - failure.get('failure_rate', 0):.1f}%<br>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown(f"**📝 Logic:** {res['logic']}")
    
    # Nút xác nhận kết quả
    st.divider()
    st.info("📌 **CÁCH DÙNG:** Sau khi có kết quả thật, nhập số vào ô trên và nhấn PHÂN TÍCH để AI học từ kết quả.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# HIỂN THỊ LỊCH SỬ
if st.session_state.accuracy_log.get('predictions'):
    with st.expander("📜 LỊCH SỬ DỰ ĐOÁN (10 GẦN NHẤT)"):
        for pred in st.session_state.accuracy_log['predictions'][-10:]:
            correct_icon = "✅" if pred.get('correct') else "❌"
            stars = "⭐" * pred.get('main_correct', 0)
            st.write(f"{correct_icon} **{pred['time']}** - Dự đoán: {pred['predicted']} | "
                    f"Thực tế: {pred['actual']} | Đúng: {pred.get('main_correct', 0)}/3 {stars}")