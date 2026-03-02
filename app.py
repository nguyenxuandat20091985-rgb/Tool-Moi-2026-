# ================= IMPORT THƯ VIỆN =================
import streamlit as st
import google.generativeai as genai
import re
import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import time

# ================= CẤU HÌNH =================
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    st.error("⚠️ Chưa cấu hình API Key trong Secrets!")
    st.stop()

st.set_page_config(page_title="TITAN v30.0 PRO MAX", layout="wide", page_icon="🎯")

# ================= KHỞI TẠO AI THÔNG MINH =================
@st.cache_resource
def get_available_models():
    try:
        genai.configure(api_key=API_KEY)
        models = genai.list_models()
        available = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available.append(model.name)
        return available
    except:
        return []

@st.cache_resource
def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        
        available_models = get_available_models()
        
        # ✅ Ưu tiên model ổn định nhất
        preferred_models = [
            'models/gemini-2.0-flash-exp',
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro',
            'models/gemini-pro',
            'models/gemini-1.0-pro'
        ]
        
        selected_model = None
        for model in preferred_models:
            if model in available_models:
                selected_model = model
                break
        
        if not selected_model and available_models:
            selected_model = available_models[0]
        
        if not selected_model:
            st.error("❌ Không tìm thấy model!")
            return None, None
        
        model_instance = genai.GenerativeModel(selected_model)
        
        return model_instance, selected_model
        
    except Exception as e:
        st.error(f"❌ Lỗi AI: {str(e)}")
        return None, None

neural_engine, model_name = setup_neural()

# ================= QUẢN LÝ DỮ LIỆU =================
def load_data_from_json(uploaded_file):
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            return data if isinstance(data, list) else []
        except:
            return []
    return []

def convert_df_to_json(data):
    return json.dumps(data[-3000:], ensure_ascii=False).encode('utf-8')

def clean_and_validate_data(raw_text, existing_history):
    """
    ✅ LÀM SẠCH DỮ LIỆU ĐẦU VÀO
    - Chỉ lấy số đúng 5 chữ số
    - Loại bỏ trùng lặp
    - Hiển thị số đã lọc
    """
    # Tìm tất cả dãy 5 số
    all_matches = re.findall(r"\b\d{5}\b", raw_text)
    
    # Loại bỏ trùng lặp, giữ thứ tự
    unique_numbers = list(dict.fromkeys(all_matches))
    
    # Lọc số đã có trong history
    new_numbers = [num for num in unique_numbers if num not in existing_history]
    
    return {
        'total_found': len(all_matches),
        'unique': len(unique_numbers),
        'new': len(new_numbers),
        'duplicates_removed': len(all_matches) - len(unique_numbers),
        'numbers': new_numbers,
        'all_unique': unique_numbers
    }

if "history" not in st.session_state:
    st.session_state.history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False
if "last_cleaned_data" not in st.session_state:
    st.session_state.last_cleaned_data = None

# ================= THUẬT TOÁN 1: PHÁT HIỆN CẦU LỪA =================
def detect_scam_patterns(history, window=20):
    if len(history) < window:
        return {"risk_score": 0, "warnings": [], "level": "UNKNOWN"}
    
    warnings = []
    risk_score = 0
    recent = history[-window:]
    
    all_nums = "".join(recent)
    digit_freq = Counter(all_nums)
    most_common = digit_freq.most_common(1)
    if most_common and most_common[0][1] > 15:
        warnings.append(f"Số {most_common[0][0]} ra {most_common[0][1]}/{window}")
        risk_score += 25
    
    for pos in range(5):
        pos_seq = [int(num[pos]) if len(num) > pos else 0 for num in recent]
        max_streak = current = 1
        for i in range(1, len(pos_seq)):
            if pos_seq[i] == pos_seq[i-1]:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 1
        if max_streak >= 4:
            warnings.append(f"Vị {pos} bệt {max_streak}")
            risk_score += 20
    
    freq = Counter(all_nums)
    total = len(all_nums)
    if total > 0:
        entropy = -sum((c/total) * np.log2(c/total) for c in freq.values() if c > 0)
        if entropy < 2.8:
            warnings.append(f"Entropy thấp {entropy:.2f}")
            risk_score += 25
    
    if risk_score >= 60:
        level = "🔴 HIGH"
    elif risk_score >= 40:
        level = "🟡 MEDIUM"
    elif risk_score >= 20:
        level = "🟢 LOW"
    else:
        level = "✅ NORMAL"
    
    return {"risk_score": risk_score, "warnings": warnings, "level": level}

# ================= THUẬT TOÁN 2: PHÂN TÍCH NHỊP =================
def analyze_bridge_rhythm(history):
    if len(history) < 15:
        return {"patterns": [], "trend": "UNKNOWN", "hot_numbers": []}
    
    patterns = []
    recent = history[-20:]
    
    for pos in range(5):
        pos_seq = [int(num[pos]) if len(num) > pos else 0 for num in recent]
        
        for i in range(len(pos_seq)-1):
            if pos_seq[i] == pos_seq[i+1]:
                patterns.append(f"Vị{pos}:Bệt-{pos_seq[i]}")
                break
        
        for i in range(len(pos_seq)-2):
            if pos_seq[i] == pos_seq[i+2] and pos_seq[i] != pos_seq[i+1]:
                patterns.append(f"Vị{pos}:Nhịp2-{pos_seq[i]}")
                break
    
    all_digits = "".join(recent)
    hot = [str(x[0]) for x in Counter(all_digits).most_common(3)]
    
    return {
        "patterns": list(set(patterns))[:5],
        "trend": f"Nóng: {', '.join(hot)}",
        "hot_numbers": hot
    }

# ================= THUẬT TOÁN 3: THỐNG KÊ NÂNG CAO =================
def advanced_stats(history):
    if not history:
        return {}
    
    stats = {}
    recent = history[-50:]
    
    for pos, name in enumerate(['Chục ngàn', 'Ngàn', 'Trăm', 'Chục', 'Đơn vị']):
        pos_data = [int(num[pos]) for num in recent if len(num) > pos]
        weights = np.linspace(0.5, 1.0, len(pos_data))
        weighted_freq = defaultdict(float)
        for i, val in enumerate(pos_data):
            weighted_freq[val] += weights[i]
        
        top_3 = sorted(weighted_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        stats[name] = [str(x[0]) for x in top_3]
    
    totals = [sum(int(d) for d in num) for num in recent if len(num) == 5]
    stats['total'] = {
        'avg': round(np.mean(totals), 1) if totals else 0,
        'hot': [t for t, _ in Counter(totals).most_common(3)] if totals else []
    }
    
    return stats

# ================= THUẬT TOÁN 4: DỰ ĐOÁN THEO VỊ TRÍ =================
def position_based_prediction(history):
    """
    ✅ DỰ ĐOÁN RIÊNG TỪNG VỊ TRÍ
    Tăng độ chính xác bằng cách phân tích từng cột
    """
    if len(history) < 20:
        return {"numbers": [], "confidence": 50}
    
    recent = history[-30:]
    predictions = []
    
    for pos in range(5):
        pos_seq = [int(num[pos]) if len(num) > pos else 0 for num in recent]
        
        # Tần suất có trọng số
        weights = np.linspace(0.3, 1.0, len(pos_seq))
        weighted_freq = defaultdict(float)
        for i, val in enumerate(pos_seq):
            weighted_freq[val] += weights[i]
        
        # Top 2 số cho vị trí này
        top_2 = sorted(weighted_freq.items(), key=lambda x: x[1], reverse=True)[:2]
        predictions.append({
            'position': pos,
            'top_numbers': [x[0] for x in top_2],
            'confidence': top_2[0][1] if top_2 else 0
        })
    
    # Chọn số có confidence cao nhất từ mỗi vị trí
    main_numbers = [p['top_numbers'][0] for p in predictions if p['top_numbers']]
    
    return {
        "numbers": main_numbers[:5],
        "by_position": predictions,
        "confidence": np.mean([p['confidence'] for p in predictions]) if predictions else 50
    }

# ================= GIAO DIỆN =================
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 12px; padding: 25px; margin: 15px 0;
    }
    .num-box {
        font-size: 80px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 12px;
    }
    .lot-box {
        font-size: 55px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 8px;
    }
    .status-bar { 
        padding: 15px; border-radius: 10px; text-align: center; 
        font-weight: bold; font-size: 20px; margin: 15px 0; 
    }
    .risk-high { background: #7c2d12; border-left: 5px solid #fbbf24; }
    .risk-med { background: #451a03; border-left: 5px solid #f59e0b; }
    .risk-low { background: #064e3b; border-left: 5px solid #10b981; }
    .data-info { 
        background: #1f2937; border-radius: 8px; padding: 15px; 
        margin: 10px 0; border: 1px solid #374151;
    }
    .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
    .stat-item { background: #374151; padding: 10px; border-radius: 8px; text-align: center; }
    .stat-value { font-size: 24px; font-weight: bold; color: #58a6ff; }
    .stat-label { font-size: 12px; color: #8b949e; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🎯 TITAN v30.0 PRO MAX</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: #8b949e;'>🤖 Model: <code>{model_name.split('/')[-1] if model_name else 'None'}</code> | 📊 Data: <code>{len(st.session_state.history)}</code> kỳ</p>", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Cài đặt")
    
    st.session_state.show_debug = st.checkbox("🐛 Debug Mode", value=False)
    
    st.divider()
    
    uploaded_db = st.file_uploader("📂 Nạp DB", type="json")
    if uploaded_db:
        st.session_state.history = load_data_from_json(uploaded_db)
        st.success(f"✅ {len(st.session_state.history)} kỳ")
        st.rerun()
    
    st.divider()
    
    if st.session_state.history:
        json_data = convert_df_to_json(st.session_state.history)
        st.download_button(
            label="💾 Tải DB",
            data=json_data,
            file_name=f"titan_db_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )
    
    st.divider()
    st.metric("📊 Tổng kỳ", len(st.session_state.history))
    
    if st.button("🗑️ Xóa toàn bộ"):
        st.session_state.history = []
        st.session_state.last_prediction = None
        st.session_state.last_cleaned_data = None
        st.rerun()
    
    st.divider()
    if model_name:
        st.success(f"✅ AI: {model_name.split('/')[-1]}")
    else:
        st.error("❌ AI không khả dụng")

# ================= NHẬP LIỆU & LÀM SẠCH =================
st.markdown("### 📡 NHẬP DỮ LIỆU")
col1, col2 = st.columns([3, 1])
with col1:
    raw_input = st.text_area("Dán kết quả (5 số/dòng)", height=150, 
                            placeholder="25878\n58261\n83104...",
                            help="Tool sẽ tự động lọc số trùng và chỉ lưu số mới")
with col2:
    st.metric("Kỳ trong DB", len(st.session_state.history))
    
    if st.button("🔍 XEM TRƯỚC DỮ LIỆU", use_container_width=True):
        if raw_input:
            clean_result = clean_and_validate_data(raw_input, st.session_state.history)
            st.session_state.last_cleaned_data = clean_result
    
    if st.button("🚀 LƯU & XỬ LÝ", type="primary", use_container_width=True):
        if raw_input:
            clean_result = clean_and_validate_data(raw_input, st.session_state.history)
            st.session_state.last_cleaned_data = clean_result
            
            if clean_result['new'] > 0:
                st.session_state.history.extend(clean_result['numbers'])
                st.session_state.history = st.session_state.history[-3000:]
                st.success(f"✅ Lưu {clean_result['new']} kỳ mới!")
                st.info(f"📊 Tìm: {clean_result['total_found']} | Riêng: {clean_result['unique']} | Trùng: {clean_result['duplicates_removed']}")
                st.rerun()
            else:
                st.warning("⚠️ Tất cả số đã có trong DB!")
        else:
            st.warning("Vui lòng nhập dữ liệu!")

# Hiển thị kết quả làm sạch
if st.session_state.last_cleaned_data:
    st.markdown("### 📊 KẾT QUẢ LÀM SẠCH DỮ LIỆU")
    c1, c2, c3, c4 = st.columns(4)
    d = st.session_state.last_cleaned_data
    c1.metric("🔍 Tìm thấy", d['total_found'])
    c2.metric("✅ Riêng", d['unique'])
    c3.metric("➕ Mới", d['new'])
    c4.metric("🗑️ Trùng", d['duplicates_removed'])
    
    if d['duplicates_removed'] > 0:
        st.warning(f"⚠️ Đã loại {d['duplicates_removed']} số trùng lặp tự động!")

# ================= PHÂN TÍCH =================
st.markdown("---")
st.subheader("🔬 PHÂN TÍCH THÔNG MINH")

if st.session_state.history and len(st.session_state.history) >= 15:
    if st.button("🎯 CHẠY AI PHÂN TÍCH", type="secondary", use_container_width=True):
        if neural_engine is None:
            st.error("❌ AI không khả dụng")
        else:
            with st.spinner("🤖 Đang phân tích đa lớp..."):
                
                # Chạy 4 thuật toán
                scam_detect = detect_scam_patterns(st.session_state.history)
                bridge_rhythm = analyze_bridge_rhythm(st.session_state.history)
                adv_stats = advanced_stats(st.session_state.history)
                pos_pred = position_based_prediction(st.session_state.history)
                
                # ✅ PROMPT CẢI TIẾN - RÕ RÀNG HƠN
                prompt = f"""Bạn là TITAN v30.0 - Chuyên gia xổ số cao cấp.

=== DỮ LIỆU ===
Lịch sử 50 kỳ gần: {st.session_state.history[-50:]}
Risk Score: {scam_detect['risk_score']}/100 ({scam_detect['level']})
Cầu phát hiện: {bridge_rhythm['patterns']}
Số nóng: {bridge_rhythm['hot_numbers']}
Dự đoán theo vị trí: {pos_pred['numbers']}

=== YÊU CẦU ===
1. Nếu Risk >= 60: decision = "DỪNG" (không cần dự đoán số)
2. Nếu Risk < 60: 
   - main_3: 3 số có xác suất cao nhất (3 chữ số, ví dụ: "123")
   - support_4: 4 số lót (4 chữ số, ví dụ: "4567")
   - decision: "ĐÁNH" hoặc "THEO DÕI"
   - confidence: 0-100

=== QUY TẮC ===
- main_3 và support_4 PHẢI là chuỗi số, không phải mảng
- confidence PHẢI là số nguyên
- decision PHẢI là "ĐÁNH", "THEO DÕI", hoặc "DỪNG"
- KHÔNG thêm text ngoài JSON

=== JSON OUTPUT ===
{{"main_3":"123","support_4":"4567","decision":"ĐÁNH","confidence":85,"logic":"Phân tích ngắn"}}"""
                
                try:
                    # Retry logic
                    max_retries = 3
                    result = None
                    
                    for attempt in range(max_retries):
                        response = neural_engine.generate_content(
                            prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=0.1,  # Rất thấp để ổn định
                                top_p=0.9,
                                max_output_tokens=512
                            )
                        )
                        
                        raw_text = response.text.strip()
                        
                        if st.session_state.show_debug:
                            st.markdown(f"##### 📄 Raw Response (lần {attempt+1}):")
                            st.code(raw_text[:1000], language="text")
                        
                        # Parse JSON - nhiều cách
                        json_str = None
                        
                        # Cách 1: Markdown code block
                        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
                        if match:
                            json_str = match.group(1)
                        
                        # Cách 2: Tìm JSON object
                        if not json_str:
                            match = re.search(r'\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}', raw_text, re.DOTALL)
                            if match:
                                json_str = match.group(0)
                        
                        # Cách 3: Parse trực tiếp
                        if not json_str:
                            json_str = raw_text
                        
                        try:
                            result = json.loads(json_str)
                            
                            # Validate fields
                            required = ['main_3', 'support_4', 'decision', 'confidence', 'logic']
                            if all(k in result for k in required):
                                # Validate types
                                if not isinstance(result['main_3'], str) or len(result['main_3']) != 3:
                                    result['main_3'] = str(result['main_3'])[:3].zfill(3)
                                if not isinstance(result['support_4'], str) or len(result['support_4']) != 4:
                                    result['support_4'] = str(result['support_4'])[:4].zfill(4)
                                if not isinstance(result['confidence'], (int, float)):
                                    result['confidence'] = 70
                                
                                result['risk_assessment'] = scam_detect
                                result['position_prediction'] = pos_pred
                                st.session_state.last_prediction = result
                                
                                if attempt > 0:
                                    st.success(f"✅ OK sau {attempt+1} lần thử!")
                                else:
                                    st.success("✅ Phân tích hoàn tất!")
                                break
                            else:
                                missing = [k for k in required if k not in result]
                                if attempt == max_retries - 1:
                                    st.error(f"❌ Thiếu: {missing}")
                        except json.JSONDecodeError:
                            if attempt == max_retries - 1:
                                st.error("❌ Không parse được JSON sau 3 lần")
                    
                    # Fallback nếu tất cả retry thất bại
                    if result is None:
                        st.warning("⚠️ Dùng fallback thống kê + vị trí")
                        all_n = "".join(st.session_state.history[-50:])
                        top = [str(x[0]) for x in Counter(all_n).most_common(7)]
                        
                        # Kết hợp với dự đoán vị trí
                        pos_nums = [str(n) for n in pos_pred['numbers'][:3]]
                        main_3 = "".join(pos_nums) if pos_nums else "".join(top[:3])
                        
                        st.session_state.last_prediction = {
                            "main_3": main_3,
                            "support_4": "".join(top[3:7]) if len(top) >= 7 else "".join(top[3:]),
                            "decision": "THEO DÕI",
                            "confidence": 75,
                            "logic": f"Fallback: Thống kê + Vị trí (Hot: {bridge_rhythm['hot_numbers']})",
                            "risk_assessment": scam_detect,
                            "position_prediction": pos_pred
                        }
                
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
                    if st.session_state.show_debug:
                        import traceback
                        st.code(traceback.format_exc(), language="text")

elif st.session_state.history:
    st.info(f"💡 Cần ≥15 kỳ (có {len(st.session_state.history)})")

# ================= HIỂN THỊ KẾT QUẢ =================
if st.session_state.last_prediction:
    res = st.session_state.last_prediction
    risk = res.get('risk_assessment', {})
    risk_score = risk.get('risk_score', 0)
    
    st.markdown("---")
    st.markdown("### 🎯 KẾT QUẢ DỰ ĐOÁN")
    
    if risk_score >= 60:
        st.markdown(f"<div class='status-bar risk-high'>🚨 RỦI RO: {risk_score}/100 - {risk.get('level')}<br>{' | '.join(risk.get('warnings', []))}</div>", unsafe_allow_html=True)
        st.error("**KHUYẾN NGHỊ: DỪNG CHƠI!** Pattern bất thường, không nên vào tiền.")
    elif risk_score >= 40:
        st.markdown(f"<div class='status-bar risk-med'>⚠️ RỦI RO: {risk_score}/100 - {risk.get('level')}</div>", unsafe_allow_html=True)
        st.warning("**CẨN THẬN** - Đánh nhỏ, quản lý vốn chặt.")
    else:
        st.markdown(f"<div class='status-bar risk-low'>✅ RỦI RO: {risk_score}/100 - {risk.get('level')}</div>", unsafe_allow_html=True)
    
    if risk_score < 60:
        decision = res.get('decision', 'N/A')
        conf = res.get('confidence', 0)
        
        colors = {"ĐÁNH": ("#238636", "✅"), "THEO DÕI": ("#d29922", "⏳"), "DỪNG": ("#da3633", "🛑")}
        bg, icon = colors.get(decision, ("#30363d", "❓"))
        
        st.markdown(f"""
            <div class='status-bar' style='background:{bg};color:white;'>
                {icon} KẾT LUẬN: {decision} | Độ tin cậy: {conf}%
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.markdown("<p style='text-align:center;color:#8b949e;'>🔥 3 SỐ CHÍNH</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='num-box'>{res.get('main_3', '???')}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<p style='text-align:center;color:#8b949e;'>🛡️ 4 SỐ LÓT</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='lot-box'>{res.get('support_4', '???')}</div>", unsafe_allow_html=True)
        
        st.divider()
        
        col_logic, col_copy = st.columns([2, 1])
        with col_logic:
            st.write(f"💡 **Logic:** {res.get('logic', 'N/A')}")
            
            # Hiển thị dự đoán theo vị trí nếu có
            if 'position_prediction' in res:
                pos_pred = res['position_prediction']
                if 'by_position' in pos_pred:
                    st.write("📍 **Dự đoán theo vị trí:**")
                    pos_names = ['Chục ngàn', 'Ngàn', 'Trăm', 'Chục', 'Đơn vị']
                    for i, pos_data in enumerate(pos_pred['by_position'][:5]):
                        st.write(f"- {pos_names[i]}: `{pos_data['top_numbers']}`")
            
            if conf < 75:
                st.warning("⚠️ Độ tin thấp - Nên đánh nhỏ hoặc chờ")
        
        with col_copy:
            full_dan = "".join(sorted(set(str(res.get('main_3', '')) + str(res.get('support_4', '')))))
            st.text_input("📋 Dàn 7 số:", full_dan)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ================= BIỂU ĐỒ =================
st.markdown("---")
with st.expander("📊 Thống kê chi tiết"):
    if st.session_state.history:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("##### Tần suất số (50 kỳ)")
            all_d = "".join(st.session_state.history[-50:])
            st.bar_chart(pd.Series(Counter(all_d)).sort_index(), color="#58a6ff")
        
        with col2:
            st.write("##### Tổng các kỳ")
            totals = [sum(int(d) for d in num) for num in st.session_state.history[-50:] if len(num)==5]
            st.line_chart(pd.Series(totals))

# ================= FOOTER =================
st.markdown("---")
st.caption("""
⚠️ **LƯU Ý:** Tool tham khảo dựa trên thống kê + AI. Nhà cái có thể điều khiển kết quả.
📊 **Độ chính xác:** Không đảm bảo 100%. Quản lý vốn chặt, dừng đúng lúc!
🔐 **Bảo mật:** API Key trong Secrets, không lộ trên GitHub.
""")
st.caption(f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')} | TITAN v30.0 PRO MAX")