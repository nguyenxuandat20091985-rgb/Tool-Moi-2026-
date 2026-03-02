# ================= IMPORT THƯ VIỆN =================
import streamlit as st
import google.generativeai as genai
import re
import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import time

# ================= CẤU HÌNH =================
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    st.error("⚠️ Chưa cấu hình API Key trong Secrets!")
    st.stop()

st.set_page_config(page_title="TITAN v30.1 QUOTA-SAFE", layout="wide", page_icon="🎯")

# ================= QUẢN LÝ QUOTA =================
@st.cache_resource
def init_quota_tracker():
    """Theo dõi số lần gọi API trong ngày"""
    return {
        'count': 0,
        'last_reset': datetime.now().date(),
        'limit': 18,  # Để dự phòng 2 request
        'last_error': None
    }

quota = init_quota_tracker()

def check_quota():
    """Kiểm tra còn quota không"""
    today = datetime.now().date()
    if quota['last_reset'] != today:
        quota['count'] = 0
        quota['last_reset'] = today
        quota['last_error'] = None
    
    remaining = quota['limit'] - quota['count']
    return remaining > 0, remaining

def increment_quota():
    """Tăng counter sau khi gọi API thành công"""
    quota['count'] += 1

def set_quota_error(error_msg):
    """Lưu lỗi quota để hiển thị"""
    quota['last_error'] = error_msg

# ================= KHỞI TẠO AI =================
@st.cache_resource
def get_available_models():
    try:
        genai.configure(api_key=API_KEY)
        models = genai.list_models()
        return [m.name for m in models if 'generateContent' in m.supported_generation_methods]
    except:
        return []

@st.cache_resource
def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        
        # ✅ Ưu tiên model có quota cao hơn
        preferred = [
            'models/gemini-1.5-flash',  # Free tier: 15 RPM, 1500 RPD
            'models/gemini-pro',
            'models/gemini-1.0-pro'
        ]
        
        available = get_available_models()
        selected = next((m for m in preferred if m in available), None)
        
        if not selected and available:
            selected = available[0]
        
        if not selected:
            return None, None
        
        return genai.GenerativeModel(selected), selected
        
    except Exception as e:
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
    ✅ LÀM SẠCH DỮ LIỆU NÂNG CAO
    - Loại bỏ khoảng trắng, ký tự đặc biệt
    - Chỉ giữ số đúng 5 chữ số
    - Báo cáo chi tiết số bị loại
    """
    # 1. Chuẩn hóa: xóa space, tab, newline thừa
    cleaned = re.sub(r'[\s\t]+', ' ', raw_text.strip())
    
    # 2. Tìm tất cả dãy 5 số (có thể có space giữa)
    # Match: 12345 hoặc 12 345 hoặc 1 2 3 4 5
    patterns = [
        r'\b\d{5}\b',  # 12345
        r'\b\d\s*\d\s*\d\s*\d\s*\d\b',  # 1 2 3 4 5 hoặc 12 345
    ]
    
    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, cleaned)
        for m in matches:
            # Chuẩn hóa về 5 số liên tiếp
            normalized = re.sub(r'\D', '', m)
            if len(normalized) == 5:
                all_matches.append(normalized)
    
    # Loại trùng, giữ thứ tự
    unique = list(dict.fromkeys(all_matches))
    new_numbers = [n for n in unique if n not in existing_history]
    
    # Phân tích số bị loại
    rejected = []
    for m in re.findall(r'\b\d{3,7}\b', cleaned):
        normalized = re.sub(r'\D', '', m)
        if len(normalized) != 5 and normalized not in unique:
            rejected.append(f"{m}→{len(normalized)}digit")
    
    return {
        'total_found': len(all_matches),
        'unique': len(unique),
        'new': len(new_numbers),
        'duplicates': len(all_matches) - len(unique),
        'rejected_sample': list(set(rejected))[:10],  # Sample 10 số bị loại
        'numbers': new_numbers,
        'all_unique': unique
    }

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False
if "last_cleaned_data" not in st.session_state:
    st.session_state.last_cleaned_data = None

# ================= THUẬT TOÁN =================
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
    
    return {"patterns": list(set(patterns))[:5], "trend": f"Nóng: {', '.join(hot)}", "hot_numbers": hot}

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

def position_based_prediction(history):
    if len(history) < 20:
        return {"numbers": [], "confidence": 50}
    
    recent = history[-30:]
    predictions = []
    
    for pos in range(5):
        pos_seq = [int(num[pos]) if len(num) > pos else 0 for num in recent]
        weights = np.linspace(0.3, 1.0, len(pos_seq))
        weighted_freq = defaultdict(float)
        for i, val in enumerate(pos_seq):
            weighted_freq[val] += weights[i]
        top_2 = sorted(weighted_freq.items(), key=lambda x: x[1], reverse=True)[:2]
        predictions.append({'position': pos, 'top_numbers': [x[0] for x in top_2], 'confidence': top_2[0][1] if top_2 else 0})
    
    main_numbers = [p['top_numbers'][0] for p in predictions if p['top_numbers']]
    return {"numbers": main_numbers[:5], "by_position": predictions, "confidence": np.mean([p['confidence'] for p in predictions]) if predictions else 50}

def fallback_prediction(history, bridge_rhythm):
    """✅ DỰ PHÒNG KHI AI LỖI/QUOTA"""
    all_n = "".join(history[-50:])
    top = [str(x[0]) for x in Counter(all_n).most_common(7)]
    pos_pred = position_based_prediction(history)
    
    # Kết hợp: ưu tiên số từ position prediction
    pos_nums = [str(n) for n in pos_pred['numbers'][:3]]
    main_3 = "".join(pos_nums) if pos_nums else "".join(top[:3])
    
    return {
        "main_3": main_3,
        "support_4": "".join(top[3:7]) if len(top) >= 7 else "".join(top[3:]),
        "decision": "THEO DÕI",
        "confidence": 75,
        "logic": f"Fallback: Thống kê + Vị trí (Hot: {bridge_rhythm['hot_numbers']})",
        "is_fallback": True
    }

# ================= GIAO DIỆN =================
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card { background: #0d1117; border: 1px solid #30363d; border-radius: 12px; padding: 25px; margin: 15px 0; }
    .num-box { font-size: 80px; font-weight: 900; color: #ff5858; text-align: center; letter-spacing: 12px; }
    .lot-box { font-size: 55px; font-weight: 700; color: #58a6ff; text-align: center; letter-spacing: 8px; }
    .status-bar { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 20px; margin: 15px 0; }
    .risk-high { background: #7c2d12; border-left: 5px solid #fbbf24; }
    .risk-med { background: #451a03; border-left: 5px solid #f59e0b; }
    .risk-low { background: #064e3b; border-left: 5px solid #10b981; }
    .quota-warning { background: #422006; border: 1px solid #9a6700; padding: 15px; border-radius: 8px; margin: 10px 0; }
    .quota-ok { background: #064e3b; border: 1px solid #10b981; padding: 15px; border-radius: 8px; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🎯 TITAN v30.1 QUOTA-SAFE</h1>", unsafe_allow_html=True)

# Hiển thị quota status
has_quota, remaining = check_quota()
if has_quota:
    st.markdown(f"""
        <div class='quota-ok'>
            ✅ Quota API: <strong>{remaining}/{quota['limit']}</strong> request còn lại hôm nay
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
        <div class='quota-warning'>
            ⚠️ <strong>ĐÃ HẾT QUOTA API!</strong> Còn 0/{quota['limit']} request.
            <br>🔄 Tool sẽ tự động chuyển sang chế độ thống kê thuần túy.
            <br>⏰ Quota sẽ reset vào 00:00 UTC ngày mai.
        </div>
    """, unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Cài đặt")
    st.session_state.show_debug = st.checkbox("🐛 Debug Mode", value=False)
    
    st.divider()
    uploaded_db = st.file_uploader("📂 Nạp DB", type="json")
    if uploaded_db:
        st.session_state.history = load_data_from_json(uploaded_db)
        st.success(f"✅ {len(st.session_state.history)} kỳ")
    
    st.divider()
    if st.session_state.history:
        json_data = convert_df_to_json(st.session_state.history)
        st.download_button(label="💾 Tải DB", data=json_data, file_name=f"titan_db_{datetime.now().strftime('%Y%m%d')}.json", mime="application/json")
    
    st.divider()
    st.metric("📊 Tổng kỳ", len(st.session_state.history))
    st.metric("🔌 API calls today", quota['count'])
    
    if st.button("🗑️ Xóa toàn bộ"):
        st.session_state.history = []
        st.session_state.last_prediction = None
        st.rerun()
    
    if model_name:
        st.success(f"✅ Model: {model_name.split('/')[-1]}")
    else:
        st.error("❌ AI không khả dụng")

# ================= NHẬP LIỆU =================
st.markdown("### 📡 NHẬP DỮ LIỆU")
col1, col2 = st.columns([3, 1])
with col1:
    raw_input = st.text_area("Dán kết quả (5 số/dòng)", height=150, placeholder="25878\n58261...", help="Tool tự động loại space, trùng lặp")
with col2:
    st.metric("Kỳ trong DB", len(st.session_state.history))
    
    if st.button("🔍 XEM TRƯỚC", use_container_width=True):
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
                st.info(f"📊 Tìm: {clean_result['total_found']} | Riêng: {clean_result['unique']} | Trùng: {clean_result['duplicates']}")
                st.rerun()
            else:
                st.warning("⚠️ Tất cả số đã có trong DB!")

# Hiển thị kết quả làm sạch
if st.session_state.last_cleaned_
    st.markdown("### 📊 KẾT QUẢ LÀM SẠCH")
    c1, c2, c3, c4 = st.columns(4)
    d = st.session_state.last_cleaned_data
    c1.metric("🔍 Tìm thấy", d['total_found'])
    c2.metric("✅ Riêng", d['unique'])
    c3.metric("➕ Mới", d['new'])
    c4.metric("🗑️ Trùng", d['duplicates'])
    
    if d['rejected_sample']:
        with st.expander("🚫 Số bị loại (định dạng sai)"):
            for r in d['rejected_sample']:
                st.write(f"- `{r}`")

# ================= PHÂN TÍCH =================
st.markdown("---")
st.subheader("🔬 PHÂN TÍCH")

if st.session_state.history and len(st.session_state.history) >= 15:
    if st.button("🎯 CHẠY PHÂN TÍCH", type="secondary", use_container_width=True):
        
        # Chạy thuật toán thống kê trước (không tốn quota)
        scam_detect = detect_scam_patterns(st.session_state.history)
        bridge_rhythm = analyze_bridge_rhythm(st.session_state.history)
        adv_stats = advanced_stats(st.session_state.history)
        pos_pred = position_based_prediction(st.session_state.history)
        
        # Nếu risk cao, không cần gọi AI
        if scam_detect['risk_score'] >= 60:
            st.session_state.last_prediction = {
                "main_3": "000", "support_4": "0000", "decision": "DỪNG",
                "confidence": 99, "logic": "Risk cao - Không nên chơi",
                "risk_assessment": scam_detect, "is_fallback": True
            }
            st.warning("⚠️ Risk cao → Không gọi AI, dùng cảnh báo thống kê")
            st.rerun()
        
        # Kiểm tra quota trước khi gọi AI
        has_quota, remaining = check_quota()
        
        if not has_quota or neural_engine is None:
            st.warning("⚠️ Hết quota hoặc AI lỗi → Dùng fallback")
            result = fallback_prediction(st.session_state.history, bridge_rhythm)
            result['risk_assessment'] = scam_detect
            st.session_state.last_prediction = result
            st.rerun()
        
        # Gọi AI với retry logic
        with st.spinner("🤖 Đang phân tích..."):
            prompt = f"""TITAN v30.1 - Chuyên gia xổ số.
DATA: Lịch sử: {st.session_state.history[-50:]} | Risk: {scam_detect['risk_score']}/100 | Patterns: {bridge_rhythm['patterns']} | Hot: {bridge_rhythm['hot_numbers']}
YÊU CẦU: 1) Nếu Risk>=60: decision="DỪNG" 2) Nếu Risk<60: chọn main_3(3 số), support_4(4 số), decision="ĐÁNH"/"THEO DÕI"
JSON: {{"main_3":"123","support_4":"4567","decision":"ĐÁNH","confidence":85,"logic":"Ngắn gọn"}}"""
            
            try:
                response = neural_engine.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.2, top_p=0.9, max_output_tokens=512)
                )
                
                raw_text = response.text.strip()
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
                if not json_match:
                    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                
                if json_match:
                    result = json.loads(json_match.group(1))
                    required = ['main_3', 'support_4', 'decision', 'confidence', 'logic']
                    if all(k in result for k in required):
                        result['main_3'] = str(result['main_3'])[:3].zfill(3)
                        result['support_4'] = str(result['support_4'])[:4].zfill(4)
                        result['risk_assessment'] = scam_detect
                        result['is_fallback'] = False
                        st.session_state.last_prediction = result
                        increment_quota()
                        st.success(f"✅ OK! Quota còn: {check_quota()[1]}")
                        st.rerun()
                
                # Nếu parse lỗi
                raise ValueError("Invalid JSON format")
                
            except Exception as e:
                error_str = str(e)
                
                # ✅ Xử lý quota exceeded
                if '429' in error_str or 'quota' in error_str.lower() or 'rate limit' in error_str.lower():
                    set_quota_error(error_str)
                    st.warning("⚠️ Hết quota API → Chuyển sang fallback")
                    result = fallback_prediction(st.session_state.history, bridge_rhythm)
                    result['risk_assessment'] = scam_detect
                    result['quota_error'] = True
                    st.session_state.last_prediction = result
                    st.rerun()
                
                # Lỗi khác
                st.error(f"❌ Lỗi AI: {error_str[:200]}")
                result = fallback_prediction(st.session_state.history, bridge_rhythm)
                result['risk_assessment'] = scam_detect
                st.session_state.last_prediction = result
                st.info("⚠️ Dùng fallback thống kê")

elif st.session_state.history:
    st.info(f"💡 Cần ≥15 kỳ (có {len(st.session_state.history)})")

# ================= HIỂN THỊ KẾT QUẢ =================
if st.session_state.last_prediction:
    res = st.session_state.last_prediction
    risk = res.get('risk_assessment', {})
    risk_score = risk.get('risk_score', 0)
    
    st.markdown("---")
    st.markdown("### 🎯 KẾT QUẢ")
    
    if risk_score >= 60:
        st.markdown(f"<div class='status-bar risk-high'>🚨 RỦI RO: {risk_score}/100 - {risk.get('level')}<br>{' | '.join(risk.get('warnings', []))}</div>", unsafe_allow_html=True)
        st.error("**KHUYẾN NGHỊ: DỪNG CHƠI!** Pattern bất thường.")
    elif risk_score >= 40:
        st.markdown(f"<div class='status-bar risk-med'>⚠️ RỦI RO: {risk_score}/100 - {risk.get('level')}</div>", unsafe_allow_html=True)
        st.warning("**CẨN THẬN** - Đánh nhỏ.")
    else:
        st.markdown(f"<div class='status-bar risk-low'>✅ RỦI RO: {risk_score}/100 - {risk.get('level')}</div>", unsafe_allow_html=True)
    
    if risk_score < 60:
        decision = res.get('decision', 'N/A')
        conf = res.get('confidence', 0)
        is_fallback = res.get('is_fallback', False)
        
        colors = {"ĐÁNH": ("#238636", "✅"), "THEO DÕI": ("#d29922", "⏳"), "DỪNG": ("#da3633", "🛑")}
        bg, icon = colors.get(decision, ("#30363d", "❓"))
        
        badge = "🔄 FALLBACK" if is_fallback else "🤖 AI"
        st.markdown(f"""
            <div class='status-bar' style='background:{bg};color:white;'>
                {icon} {decision} | Độ tin: {conf}% | {badge}
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
        st.write(f"💡 **Logic:** {res.get('logic', 'N/A')}")
        
        full_dan = "".join(sorted(set(str(res.get('main_3', '')) + str(res.get('support_4', '')))))
        st.text_input("📋 Dàn 7 số:", full_dan)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("---")
st.caption("""
⚠️ **LƯU Ý:** Tool tham khảo. Nhà cái có thể điều khiển kết quả. Quản lý vốn chặt!
🔌 **Quota API:** Free tier ~20 request/ngày. Tool tự chuyển fallback khi hết quota.
🔐 **Bảo mật:** API Key trong Secrets, không lộ trên GitHub.
""")
st.caption(f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')} | TITAN v30.1 QUOTA-SAFE")