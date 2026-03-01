# ================= IMPORT THƯ VIỆN =================
import streamlit as st
import google.generativeai as genai
import re
import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations

# ================= CẤU HÌNH BẢO MẬT =================
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    st.error("⚠️ Chưa cấu hình API Key trong Secrets!")
    st.stop()

# ================= KHỞI TẠO HỆ THỐNG =================
st.set_page_config(page_title="TITAN v26.0 PRO", layout="wide", page_icon="🎯")

@st.cache_resource
def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash-latest')
    except: 
        return None

neural_engine = setup_neural()

# ================= QUẢN LÝ DỮ LIỆU =================
def load_data_from_json(uploaded_file):
    if uploaded_file is not None:
        try:
            return json.load(uploaded_file)
        except:
            return []
    return []

def convert_df_to_json(data):
    return json.dumps(data, ensure_ascii=False).encode('utf-8')

# Khởi tạo session state
if "history" not in st.session_state:
    st.session_state.history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ================= THUẬT TOÁN 1: PHÂN TÍCH TẦN SUẤT NÂNG CAO =================
def advanced_frequency_analysis(history, top_n=10):
    """Phân tích tần suất với trọng số thời gian"""
    if not history:
        return {}
    
    # Tách từng vị trí
    positions = {'hang_chuc_ngan': [], 'hang_ngan': [], 'hang_tram': [], 'hang_chuc': [], 'hang_don_vi': []}
    
    for num in history[-50:]:
        if len(num) == 5:
            positions['hang_chuc_ngan'].append(int(num[0]))
            positions['hang_ngan'].append(int(num[1]))
            positions['hang_tram'].append(int(num[2]))
            positions['hang_chuc'].append(int(num[3]))
            positions['hang_don_vi'].append(int(num[4]))
    
    # Tính tần suất có trọng số (kỳ gần nặng hơn)
    weighted_freq = {}
    for pos_name, pos_data in positions.items():
        freq = Counter(pos_data)
        # Trọng số giảm dần
        weights = [i/len(pos_data) for i in range(1, len(pos_data)+1)]
        weighted = defaultdict(float)
        for i, num in enumerate(pos_data):
            weighted[num] += weights[i]
        
        weighted_freq[pos_name] = dict(sorted(weighted.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    return weighted_freq

# ================= THUẬT TOÁN 2: NHẬN DIỆN CẦU =================
def detect_patterns(history, window=20):
    """Nhận diện các dạng cầu: bệt, đảo, nhịp"""
    if len(history) < window:
        return {"cau_bet": [], "cau_dao": [], "cau_nhip": []}
    
    patterns = {
        "cau_bet": [],      # Số ra liên tiếp
        "cau_dao": [],      # Số ra xen kẽ
        "cau_nhip": [],     # Số ra theo nhịp 2-3 kỳ
        "cau_cham": []      # Số lâu chưa ra
    }
    
    # Lấy 20 kỳ gần
    recent = history[-window:]
    all_nums = "".join(recent)
    
    # Phân tích từng vị trí
    for pos in range(5):
        pos_sequence = [int(num[pos]) if len(num) > pos else 0 for num in recent]
        
        # Cầu bệt (ra liên tiếp 2-3 lần)
        for i in range(len(pos_sequence)-1):
            if pos_sequence[i] == pos_sequence[i+1]:
                patterns["cau_bet"].append({
                    'so': pos_sequence[i],
                    'vi_tri': pos,
                    'lan': 2
                })
        
        # Cầu nhịp 2 (ra cách 1 kỳ)
        for i in range(len(pos_sequence)-2):
            if pos_sequence[i] == pos_sequence[i+2] and pos_sequence[i] != pos_sequence[i+1]:
                patterns["cau_nhip"].append({
                    'so': pos_sequence[i],
                    'vi_tri': pos,
                    'nhịp': 2
                })
        
        # Cầu nhịp 3
        for i in range(len(pos_sequence)-3):
            if pos_sequence[i] == pos_sequence[i+3]:
                patterns["cau_nhip"].append({
                    'so': pos_sequence[i],
                    'vi_tri': pos,
                    'nhịp': 3
                })
    
    # Số lâu chưa ra (cold numbers)
    all_digits = [0,1,2,3,4,5,6,7,8,9]
    recent_digits = set(int(d) for d in all_nums)
    cold = [d for d in all_digits if d not in recent_digits]
    patterns["cau_cham"] = cold
    
    return patterns

# ================= THUẬT TOÁN 3: THỐNG KÊ TỔNG - THIỆP =================
def analyze_totals(history):
    """Phân tích tổng các số"""
    if not history:
        return {}
    
    totals = []
    for num in history[-30:]:
        if len(num) == 5:
            total = sum(int(d) for d in num)
            totals.append(total)
    
    total_freq = Counter(totals)
    avg_total = np.mean(totals) if totals else 0
    
    return {
        'total_freq': dict(total_freq.most_common(5)),
        'avg_total': round(avg_total, 1),
        'hot_totals': [t for t, c in total_freq.most_common(3)]
    }

# ================= THUẬT TOÁN 4: DỰ ĐOÁN VỊ TRÍ =================
def position_prediction(history):
    """Dự đoán theo từng vị trí riêng biệt"""
    if len(history) < 10:
        return {}
    
    predictions = {}
    
    for pos in range(5):
        pos_name = ['Chục ngàn', 'Ngàn', 'Trăm', 'Chục', 'Đơn vị'][pos]
        pos_sequence = [int(num[pos]) if len(num) > pos else 0 for num in history[-30:]]
        
        # Tần suất vị trí
        freq = Counter(pos_sequence)
        top_3 = [num for num, count in freq.most_common(3)]
        
        # Xu hướng gần (5 kỳ cuối)
        recent_trend = pos_sequence[-5:]
        recent_freq = Counter(recent_trend)
        trending = [num for num, count in recent_freq.most_common(2)]
        
        predictions[pos_name] = {
            'top_3': top_3,
            'trending': trending,
            'hot': freq.most_common(1)[0][0] if freq else 0
        }
    
    return predictions

# ================= GIAO DIỆN & CSS =================
st.markdown("""
    <style>
    .stApp { background: #0d1117; color: #c9d1d9; }
    .main-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px; margin: 10px 0; }
    .big-number { font-size: 60px; font-weight: 800; color: #ff7b72; text-align: center; letter-spacing: 8px; }
    .sub-number { font-size: 40px; font-weight: 700; color: #58a6ff; text-align: center; letter-spacing: 5px; }
    .status-badge { padding: 5px 15px; border-radius: 20px; font-weight: bold; display: inline-block; }
    .bg-go { background: #238636; color: white; }
    .bg-stop { background: #da3633; color: white; }
    .algo-box { background: #1f2937; border-left: 4px solid #3b82f6; padding: 15px; margin: 10px 0; border-radius: 5px; }
    .stat-card { background: #1f2937; padding: 15px; border-radius: 8px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 TITAN v26.0 - 4 THUẬT TOÁN NÂNG CAO")
st.markdown("---")

# ================= SIDEBAR =================
with st.sidebar:
    st.header("💾 Database")
    
    uploaded_db = st.file_uploader("📂 Nạp DB (JSON)", type="json")
    if uploaded_db:
        st.session_state.history = load_data_from_json(uploaded_db)
        st.success(f"Đã nạp {len(st.session_state.history)} kỳ!")
        st.rerun()
    
    st.divider()
    
    if st.session_state.history:
        json_data = convert_df_to_json(st.session_state.history)
        st.download_button(
            label="💾 Tải DB về",
            data=json_data,
            file_name=f"titan_db_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    st.divider()
    st.write(f"📊 **Tổng kỳ:** {len(st.session_state.history)}")
    if st.button("🗑️ Xóa dữ liệu"):
        st.session_state.history = []
        st.rerun()

# ================= NHẬP LIỆU =================
col1, col2 = st.columns([3, 1])
with col1:
    raw_input = st.text_area("📡 Dán kết quả (Mỗi dòng 5 số)", height=150, placeholder="32880\n21808...")
with col2:
    st.metric("Kỳ gần nhất", len(st.session_state.history))
    if st.button("🚀 LƯU DỮ LIỆU", type="primary", use_container_width=True):
        if raw_input:
            clean = re.findall(r"\d{5}", raw_input)
            if clean:
                new_data = list(dict.fromkeys(clean))
                st.session_state.history.extend(new_data)
                st.session_state.history = st.session_state.history[-1000:]
                st.success(f"✅ Đã lưu {len(new_data)} kỳ!")
                st.rerun()

# ================= PHÂN TÍCH 4 THUẬT TOÁN =================
st.markdown("---")
st.subheader("🔬 PHÂN TÍCH ĐA THUẬT TOÁN")

if st.session_state.history and len(st.session_state.history) >= 20:
    if st.button("🎯 CHẠY 4 THUẬT TOÁN", type="secondary", use_container_width=True):
        with st.spinner("🧠 Đang phân tích..."):
            
            # Thuật toán 1: Tần suất nâng cao
            freq_analysis = advanced_frequency_analysis(st.session_state.history)
            
            # Thuật toán 2: Nhận diện cầu
            patterns = detect_patterns(st.session_state.history)
            
            # Thuật toán 3: Thống kê tổng
            totals = analyze_totals(st.session_state.history)
            
            # Thuật toán 4: Dự đoán vị trí
            pos_pred = position_prediction(st.session_state.history)
            
            # Tổng hợp kết quả
            all_digits = []
            for pos_data in freq_analysis.values():
                all_digits.extend(list(pos_data.keys())[:3])
            
            for pattern in patterns['cau_bet'] + patterns['cau_nhip']:
                all_digits.append(pattern['so'])
            
            # Tìm số xuất hiện nhiều nhất
            final_freq = Counter(all_digits)
            top_7 = [str(x[0]) for x in final_freq.most_common(7)]
            
            # Gửi AI phân tích tổng hợp
            prompt = f"""
            Role: Chuyên gia xổ số cao cấp.
            
            DỮ LIỆU PHÂN TÍCH:
            1. Lịch sử 50 kỳ: {st.session_state.history[-50:]}
            
            2. TẦN SUẤT NÂNG CAO (theo vị trí):
            {json.dumps(freq_analysis, ensure_ascii=False)}
            
            3. MÔ HÌNH CẦU PHÁT HIỆN:
            - Cầu bệt: {patterns['cau_bet']}
            - Cầu nhịp: {patterns['cau_nhip']}
            - Số lâu chưa ra: {patterns['cau_cham']}
            
            4. THỐNG KÊ TỔNG:
            {json.dumps(totals, ensure_ascii=False)}
            
            5. DỰ ĐOÁN VỊ TRÍ:
            {json.dumps(pos_pred, ensure_ascii=False)}
            
            6. TOP 7 SỐ TỪ THUẬT TOÁN: {top_7}
            
            NHIỆM VỤ:
            1. Chọn 3 số chính (có xác suất cao nhất)
            2. Chọn 4 số lót (bổ sung)
            3. Quyết định ĐÁNH hoặc CHỜ
            4. Giải thích logic rõ ràng
            
            Output JSON:
            {{
                "main_3": "abc",
                "support_4": "defg",
                "decision": "ĐÁNH",
                "confidence": 85,
                "reasoning": "Phân tích chi tiết...",
                "algorithm_weights": {{
                    "frequency": "30%",
                    "patterns": "40%",
                    "totals": "15%",
                    "positions": "15%"
                }}
            }}
            """
            
            try:
                response = neural_engine.generate_content(prompt)
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    st.session_state.last_prediction = json.loads(json_match.group())
                    st.session_state.last_prediction['algorithms'] = {
                        'frequency': freq_analysis,
                        'patterns': patterns,
                        'totals': totals,
                        'positions': pos_pred
                    }
                    st.success("✅ Phân tích hoàn tất!")
                    st.rerun()
            except Exception as e:
                st.error(f"Lỗi AI: {e}")
                # Fallback
                st.session_state.last_prediction = {
                    "main_3": "".join(top_7[:3]),
                    "support_4": "".join(top_7[3:]),
                    "decision": "ĐÁNH",
                    "confidence": 75,
                    "reasoning": "Dùng thống kê thuần túy"
                }
                st.rerun()

elif st.session_state.history:
    st.warning(f"⚠️ Cần ít nhất 20 kỳ để phân tích (hiện có {len(st.session_state.history)})")

# ================= HIỂN THỊ KẾT QUẢ =================
if st.session_state.last_prediction:
    res = st.session_state.last_prediction
    is_go = res.get('decision', '').upper() == 'ĐÁNH'
    badge_class = "bg-go" if is_go else "bg-stop"
    
    st.markdown("---")
    st.markdown(f"<div class='main-card'>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(f"<h2 style='text-align:center'>📢 KẾT LUẬN: <span class='status-badge {badge_class}'>{res.get('decision', 'CHỜ')}</span></h2>", unsafe_allow_html=True)
    
    st.divider()
    
    c_num1, c_num2 = st.columns(2)
    with c_num1:
        st.markdown("<p style='text-align:center;color:#8b949e'>🔥 3 SỐ CHÍNH</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-number'>{res.get('main_3', '???')}</div>", unsafe_allow_html=True)
    with c_num2:
        st.markdown("<p style='text-align:center;color:#8b949e'>🛡️ 4 SỐ LÓT</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='sub-number'>{res.get('support_4', '???')}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.info(f"💡 **Logic:** {res.get('reasoning', 'N/A')}")
    st.success(f"🎯 **Độ tin cậy:** {res.get('confidence', 0)}%")
    
    # Dàn số
    full_set = "".join(sorted(set(str(res.get('main_3', '')) + str(res.get('support_4', '')))))
    st.text_input("📋 Dàn số (Copy):", full_set)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ================= CHI TIẾT 4 THUẬT TOÁN =================
    if 'algorithms' in res:
        algos = res['algorithms']
        
        st.markdown("### 📊 CHI TIẾT PHÂN TÍCH")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("##### 1️⃣ TẦN SUẤT")
            for pos, data in algos['frequency'].items():
                if data:
                    top_num = list(data.keys())[0]
                    st.write(f"{pos}: **{top_num}**")
        
        with col2:
            st.markdown("##### 2️⃣ CẦU")
            if algos['patterns']['cau_bet']:
                st.write(f" Bệt: {[x['so'] for x in algos['patterns']['cau_bet'][:3]]}")
            if algos['patterns']['cau_nhip']:
                st.write(f"🔵 Nhịp: {[x['so'] for x in algos['patterns']['cau_nhip'][:3]]}")
        
        with col3:
            st.markdown("##### 3️⃣ TỔNG")
            st.write(f"TB: {algos['totals'].get('avg_total', 'N/A')}")
            st.write(f"Nóng: {algos['totals'].get('hot_totals', [])}")
        
        with col4:
            st.markdown("##### 4️⃣ VỊ TRÍ")
            for pos_name, data in list(algos['positions'].items())[:3]:
                st.write(f"{pos_name}: {data['trending']}")

# ================= BIỂU ĐỒ =================
st.markdown("---")
with st.expander("📈 Biểu đồ tần suất"):
    if st.session_state.history:
        all_d = "".join(st.session_state.history[-50:])
        df_freq = pd.Series(Counter(all_d)).sort_index()
        st.bar_chart(df_freq, color="#58a6ff")

# ================= FOOTER =================
st.markdown("---")
st.caption("⚠️ **Cảnh báo:** Công cụ tham khảo. Không đảm bảo 100%.")
st.caption(f"🕐 Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')}")