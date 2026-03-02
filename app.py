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

# ================= CẤU HÌNH BẢO MẬT =================
# ✅ Lấy API Key từ Secrets - KHÔNG hardcode
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    st.error("⚠️ Chưa cấu hình API Key! Vào Settings → Secrets để thêm: `GEMINI_API_KEY = 'your_key'`")
    st.stop()

# ================= KHỞI TẠO HỆ THỐNG =================
st.set_page_config(page_title="TITAN v28.0 PRO", layout="wide", page_icon="🎯")

@st.cache_resource
def setup_neural():
    """Khởi tạo AI với retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            genai.configure(api_key=API_KEY)
            # ✅ Dùng model mới nhất, ổn định nhất
            return genai.GenerativeModel('gemini-1.5-flash-latest')
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Lỗi khởi tạo AI sau {max_retries} lần thử: {e}")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff
    return None

neural_engine = setup_neural()

# ================= QUẢN LÝ DỮ LIỆU CLOUD-SAFE =================
def load_data_from_json(uploaded_file):
    """Load dữ liệu từ file upload"""
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            return data if isinstance(data, list) else []
        except:
            return []
    return []

def convert_df_to_json(data):
    """Convert list sang JSON bytes để download"""
    return json.dumps(data[-3000:], ensure_ascii=False).encode('utf-8')

# Khởi tạo session state
if "history" not in st.session_state:
    st.session_state.history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}

# ================= THUẬT TOÁN 1: PHÁT HIỆN CẦU LỪA =================
def detect_scam_patterns(history, window=20):
    """Phát hiện dấu hiệu nhà cái điều khiển kết quả"""
    if len(history) < window:
        return {"risk_score": 0, "warnings": [], "level": "UNKNOWN"}
    
    warnings = []
    risk_score = 0
    recent = history[-window:]
    
    # 1. Kiểm tra số ra QUÁ NHIỀU
    all_nums = "".join(recent)
    digit_freq = Counter(all_nums)
    most_common = digit_freq.most_common(1)
    if most_common and most_common[0][1] > 15:
        warnings.append(f"⚠️ Số {most_common[0][0]} ra {most_common[0][1]} lần/{window} kỳ")
        risk_score += 25
    
    # 2. Kiểm tra cầu bệt bất thường theo vị trí
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
            warnings.append(f"🎭 Vị trí {pos} bệt {max_streak} kỳ → DẤU HIỆU LỪA")
            risk_score += 20
    
    # 3. Kiểm tra độ ổn định QUÁ MỨC (entropy thấp)
    freq = Counter(all_nums)
    total = len(all_nums)
    if total > 0:
        entropy = -sum((c/total) * np.log2(c/total) for c in freq.values() if c > 0)
        if entropy < 2.8:  # Quá ổn định = giả
            warnings.append(f"⚡ Entropy thấp ({entropy:.2f}) → Pattern QUÁ ĐỀU")
            risk_score += 25
    
    # 4. Kiểm tra thay đổi đột ngột
    if len(history) >= 30:
        old_avg = np.mean([sum(int(d) for d in n) for n in history[-30:-10]])
        new_avg = np.mean([sum(int(d) for d in n) for n in history[-10:]])
        if abs(new_avg - old_avg) > 3:
            warnings.append(f"📉 Thay đổi đột ngột: {old_avg:.1f} → {new_avg:.1f}")
            risk_score += 20
    
    # Đánh giá mức độ
    if risk_score >= 60:
        level = "🔴 HIGH - DỪNG NGAY"
    elif risk_score >= 40:
        level = "🟡 MEDIUM - CẨN THẬN"
    elif risk_score >= 20:
        level = "🟢 LOW - THEO DÕI"
    else:
        level = "✅ NORMAL - ỔN"
    
    return {"risk_score": risk_score, "warnings": warnings, "level": level}

# ================= THUẬT TOÁN 2: PHÂN TÍCH NHỊP CẦU =================
def analyze_bridge_rhythm(history):
    """Phân tích nhịp cầu: bệt, đảo, nhịp 2-3"""
    if len(history) < 15:
        return {"patterns": [], "trend": "UNKNOWN"}
    
    patterns = []
    recent = history[-20:]
    
    for pos in range(5):
        pos_seq = [int(num[pos]) if len(num) > pos else 0 for num in recent]
        
        # Cầu bệt (2+ liên tiếp)
        for i in range(len(pos_seq)-1):
            if pos_seq[i] == pos_seq[i+1]:
                patterns.append(f"Vị {pos}: Bệt {pos_seq[i]}")
                break
        
        # Cầu nhịp 2 (cách 1 kỳ)
        for i in range(len(pos_seq)-2):
            if pos_seq[i] == pos_seq[i+2] and pos_seq[i] != pos_seq[i+1]:
                patterns.append(f"Vị {pos}: Nhịp-2 {pos_seq[i]}")
                break
    
    # Xu hướng chung
    all_digits = "".join(recent)
    hot = [str(x[0]) for x in Counter(all_digits).most_common(3)]
    
    return {
        "patterns": list(set(patterns))[:5],  # Max 5 patterns
        "trend": f"Nóng: {', '.join(hot)}",
        "hot_numbers": hot
    }

# ================= THUẬT TOÁN 3: THỐNG KÊ NÂNG CAO =================
def advanced_stats(history):
    """Thống kê đa chiều: vị trí, tổng, chẵn lẻ"""
    if not history:
        return {}
    
    stats = {}
    recent = history[-50:]
    
    # 1. Tần suất theo vị trí có trọng số
    for pos, name in enumerate(['Chục ngàn', 'Ngàn', 'Trăm', 'Chục', 'Đơn vị']):
        pos_data = [int(num[pos]) for num in recent if len(num) > pos]
        weights = np.linspace(0.5, 1.0, len(pos_data))  # Kỳ gần nặng hơn
        weighted_freq = defaultdict(float)
        for i, val in enumerate(pos_data):
            weighted_freq[val] += weights[i]
        
        top_3 = sorted(weighted_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        stats[name] = [str(x[0]) for x in top_3]
    
    # 2. Thống kê tổng
    totals = [sum(int(d) for d in num) for num in recent if len(num) == 5]
    stats['total'] = {
        'avg': round(np.mean(totals), 1) if totals else 0,
        'hot': [t for t, _ in Counter(totals).most_common(3)] if totals else []
    }
    
    # 3. Chẵn/Lẻ theo vị trí
    for pos in range(5):
        even_odd = [int(num[pos]) % 2 for num in recent if len(num) > pos]
        if even_odd:
            trend = "Chẵn" if Counter(even_odd)[0] > len(even_odd)/2 else "Lẻ"
            stats[f'parity_{pos}'] = trend
    
    return stats

# ================= GIAO DIỆN & CSS =================
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 12px; padding: 25px; margin: 15px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .num-box {
        font-size: 80px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 12px;
        text-shadow: 0 0 15px rgba(255,88,88,0.4);
    }
    .lot-box {
        font-size: 55px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 8px;
    }
    .status-bar { 
        padding: 15px; border-radius: 10px; text-align: center; 
        font-weight: bold; font-size: 20px; margin: 15px 0; 
    }
    .warning-box { 
        background: #331010; color: #ff7b72; padding: 12px; 
        border-radius: 8px; border: 1px solid #6e2121; 
        text-align: center; margin: 10px 0;
    }
    .risk-high { background: #7c2d12; border-left: 5px solid #fbbf24; }
    .risk-med { background: #451a03; border-left: 5px solid #f59e0b; }
    .risk-low { background: #064e3b; border-left: 5px solid #10b981; }
    .algo-tag { 
        display: inline-block; background: #30363d; padding: 3px 10px; 
        border-radius: 15px; font-size: 12px; margin: 2px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🎯 TITAN v28.0 PRO</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e;'>🛡️ Phát hiện cầu lừa + 3 thuật toán nâng cao + AI tổng hợp</p>", unsafe_allow_html=True)

# ================= SIDEBAR: QUẢN LÝ DỮ LIỆU =================
with st.sidebar:
    st.header("💾 Database Control")
    st.info("Cloud Mode: Upload/Download DB để lưu dữ liệu")
    
    uploaded_db = st.file_uploader("📂 Nạp DB cũ (JSON)", type="json")
    if uploaded_db:
        st.session_state.history = load_data_from_json(uploaded_db)
        st.success(f"✅ Đã nạp {len(st.session_state.history)} kỳ!")
    
    st.divider()
    
    if st.session_state.history:
        json_data = convert_df_to_json(st.session_state.history)
        st.download_button(
            label="💾 Tải DB về máy",
            data=json_data,
            file_name=f"titan_db_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )
    
    st.divider()
    st.metric("📊 Tổng kỳ", len(st.session_state.history))
    
    if st.button("🗑️ Xóa toàn bộ"):
        st.session_state.history = []
        st.session_state.last_prediction = None
        st.rerun()
    
    st.divider()
    st.caption("🔐 API Key: Bảo mật trong Secrets")

# ================= PHẦN 1: NHẬP LIỆU =================
col1, col2 = st.columns([3, 1])
with col1:
    raw_input = st.text_area(
        "📡 Dán kết quả (Mỗi dòng 5 số)", 
        height=120, 
        placeholder="32880\n21808\n99215..."
    )
with col2:
    st.metric("Kỳ mới", len(st.session_state.history))
    if st.button("🚀 LƯU & XỬ LÝ", type="primary", use_container_width=True):
        if raw_input:
            # ✅ Lọc sạch: chỉ lấy số đúng 5 chữ số
            clean = re.findall(r"\b\d{5}\b", raw_input)
            if clean:
                new_data = list(dict.fromkeys(clean))  # Loại trùng, giữ thứ tự
                st.session_state.history.extend(new_data)
                st.session_state.history = st.session_state.history[-3000:]  # Giới hạn
                st.success(f"✅ Đã lưu {len(new_data)} kỳ mới!")
                st.rerun()
        else:
            st.warning("Vui lòng nhập dữ liệu!")

# ================= PHẦN 2: PHÂN TÍCH ĐA THUẬT TOÁN =================
st.markdown("---")
st.subheader("🔬 PHÂN TÍCH THÔNG MINH")

if st.session_state.history and len(st.session_state.history) >= 15:
    if st.button("🎯 CHẠY PHÂN TÍCH FULL", type="secondary", use_container_width=True):
        with st.spinner("🧠 Titan đang phân tích đa lớp..."):
            
            # 🔄 Thuật toán 1: Phát hiện cầu lừa
            scam_detect = detect_scam_patterns(st.session_state.history)
            
            # 🔄 Thuật toán 2: Phân tích nhịp cầu
            bridge_rhythm = analyze_bridge_rhythm(st.session_state.history)
            
            # 🔄 Thuật toán 3: Thống kê nâng cao
            adv_stats = advanced_stats(st.session_state.history)
            
            # 🔄 Chuẩn bị prompt cho AI với dữ liệu đã xử lý
            prompt = f"""
            Role: Chuyên gia xổ số cao cấp TITAN v28.0.
            
            === DỮ LIỆU ĐẦU VÀO ===
            Lịch sử 100 kỳ gần: {st.session_state.history[-100:]}
            
            === PHÂN TÍCH CẦU LỪA ===
            Risk Score: {scam_detect['risk_score']}/100
            Level: {scam_detect['level']}
            Warnings: {scam_detect['warnings']}
            
            === NHỊP CẦU PHÁT HIỆN ===
            Patterns: {bridge_rhythm['patterns']}
            Trend: {bridge_rhythm['trend']}
            Hot numbers: {bridge_rhythm['hot_numbers']}
            
            === THỐNG KÊ NÂNG CAO ===
            Theo vị trí: {json.dumps(adv_stats, ensure_ascii=False)}
            
            === NHIỆM VỤ ===
            1. Nếu Risk Score >= 60: Khuyến nghị DỪNG, không cần dự đoán số
            2. Nếu Risk Score < 60: 
               - Chọn 3 số chính (Main_3) có xác suất cao nhất
               - Chọn 4 số lót (Support_4) để tạo dàn 7 số
               - Quyết định: "ĐÁNH", "THEO DÕI", hoặc "DỪNG"
            3. Giải thích logic rõ ràng, ngắn gọn
            
            === ĐỊNH DẠNG JSON ===
            {{
                "main_3": "123",
                "support_4": "4567", 
                "decision": "ĐÁNH",
                "confidence": 85,
                "logic": "Phân tích ngắn gọn...",
                "risk_assessment": {json.dumps(scam_detect)},
                "algorithm_contributions": {{
                    "frequency": "30%",
                    "patterns": "40%", 
                    "stats": "20%",
                    "ai_synthesis": "10%"
                }}
            }}
            
            Lưu ý: Trả về JSON THUẦN, không markdown, không giải thích thêm.
            """
            
            try:
                # ✅ Gọi AI với timeout và config rõ ràng
                response = neural_engine.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,  # Thấp để kết quả ổn định
                        top_p=0.9,
                        max_output_tokens=2048
                    )
                )
                
                # ✅ Extract JSON an toàn hơn
                text = response.text.strip()
                
                # Tìm JSON trong markdown code block trước
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
                if not json_match:
                    json_match = re.search(r'\{.*\}', text, re.DOTALL)
                
                if json_match:
                    result = json.loads(json_match.group(1))
                    st.session_state.last_prediction = result
                    st.success("✅ Phân tích hoàn tất!")
                else:
                    st.error("❌ AI trả về không đúng định dạng JSON")
                    st.code(text[:500], language="text")
                    
            except Exception as e:
                st.error(f"❌ Lỗi AI: {str(e)}")
                st.warning("⚠️ Chuyển sang chế độ thống kê thuần túy...")
                
                # ✅ Fallback: Dùng thống kê nếu AI lỗi
                all_n = "".join(st.session_state.history[-50:])
                top = [str(x[0]) for x in Counter(all_n).most_common(7)]
                st.session_state.last_prediction = {
                    "main_3": "".join(top[:3]),
                    "support_4": "".join(top[3:]),
                    "decision": "THEO DÕI",
                    "confidence": 70,
                    "logic": "Dùng thống kê tần suất (AI tạm thời không khả dụng)",
                    "risk_assessment": scam_detect
                }

elif st.session_state.history:
    st.info(f"💡 Cần ít nhất 15 kỳ để phân tích (hiện có {len(st.session_state.history)})")

# ================= PHẦN 3: HIỂN THỊ KẾT QUẢ =================
if st.session_state.last_prediction:
    res = st.session_state.last_prediction
    
    st.markdown("---")
    
    # 🚨 Hiển thị cảnh báo rủi ro TRƯỚC tiên
    risk = res.get('risk_assessment', {})
    risk_score = risk.get('risk_score', 0)
    
    if risk_score >= 60:
        st.markdown(f"""
            <div class='status-bar risk-high'>
                🚨 CẢNH BÁO ĐỎ - RỦI RO: {risk_score}/100<br>
                {risk.get('level', 'HIGH')} - {risk.get('warnings', [''])[0] if risk.get('warnings') else ''}
            </div>
        """, unsafe_allow_html=True)
        st.error("""
        **KHUYẾN NGHỊ: DỪNG CHƠI NGAY!**
        
        - Pattern bất thường, có dấu hiệu nhà cái điều khiển
        - Không nên vào tiền lúc này
        - Theo dõi thêm 10-15 kỳ nữa
        """)
    
    elif risk_score >= 40:
        st.markdown(f"""
            <div class='status-bar risk-med'>
                ⚠️ CẢNH BÁO VÀNG - RỦI RO: {risk_score}/100<br>
                {risk.get('level', 'MEDIUM')}
            </div>
        """, unsafe_allow_html=True)
        st.warning("""
        **KHUYẾN NGHỊ: CẨN THẬN**
        
        - Có dấu hiệu bất thường nhẹ
        - Nếu chơi: Đánh nhỏ, quản lý vốn chặt
        - Sẵn sàng dừng nếu có biến
        """)
    
    else:
        st.markdown(f"""
            <div class='status-bar risk-low'>
                ✅ TƯƠNG ĐỐI ỔN - RỦI RO: {risk_score}/100<br>
                {risk.get('level', 'NORMAL')}
            </div>
        """, unsafe_allow_html=True)
    
    # 🎯 Hiển thị kết quả dự đoán (chỉ khi risk < 60)
    if risk_score < 60:
        decision = res.get('decision', 'N/A')
        conf = res.get('confidence', 0)
        
        # Màu sắc theo decision
        colors = {
            "ĐÁNH": ("#238636", "✅"),
            "THEO DÕI": ("#d29922", "⏳"),
            "DỪNG": ("#da3633", "🛑")
        }
        bg, icon = colors.get(decision, ("#30363d", "❓"))
        
        st.markdown(f"""
            <div class='status-bar' style='background: {bg};'>
                {icon} KẾT LUẬN: {decision} (Độ tin: {conf}%)
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        
        # Hiển thị số dự đoán
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.markdown("<p style='color:#8b949e; text-align:center;'>🔥 3 SỐ CHÍNH</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='num-box'>{res.get('main_3', '???')}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<p style='color:#8b949e; text-align:center;'>🛡️ 4 SỐ LÓT</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='lot-box'>{res.get('support_4', '???')}</div>", unsafe_allow_html=True)
        
        st.divider()
        
        # Logic và tags thuật toán
        col_logic, col_copy = st.columns([2, 1])
        with col_logic:
            st.write(f"💡 **Logic:** {res.get('logic', 'N/A')}")
            
            # Hiển thị đóng góp thuật toán
            if 'algorithm_contributions' in res:
                st.write("🔧 **Thuật toán đóng góp:**")
                for algo, weight in res['algorithm_contributions'].items():
                    st.markdown(f"<span class='algo-tag'>{algo}: {weight}</span>", unsafe_allow_html=True)
            
            # Cảnh báo thêm nếu confidence thấp
            if conf < 75:
                st.markdown("<div class='warning-box'>⚠️ Độ tin cậy thấp - Nên đánh nhỏ hoặc chờ</div>", unsafe_allow_html=True)
        
        with col_copy:
            full_dan = "".join(sorted(set(str(res.get('main_3', '')) + str(res.get('support_4', '')))))
            st.text_input("📋 Dàn 7 số (Copy):", full_dan)
            if st.button("📋 Copy toàn bộ", key="copy_btn"):
                st.toast("Đã copy dàn số! 🎯")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ================= PHẦN 4: BIỂU ĐỒ & THỐNG KÊ =================
st.markdown("---")
with st.expander("📊 Thống kê chi tiết 50 kỳ gần"):
    if st.session_state.history:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("##### 🔢 Tần suất số (0-9)")
            all_d = "".join(st.session_state.history[-50:])
            df_freq = pd.Series(Counter(all_d)).sort_index()
            st.bar_chart(df_freq, color="#58a6ff")
        
        with col2:
            st.write("##### 📈 Tổng các kỳ")
            totals = [sum(int(d) for d in num) for num in st.session_state.history[-50:] if len(num)==5]
            st.line_chart(pd.Series(totals).tail(20))
        
        with col3:
            st.write("##### 🎯 Số nóng theo vị trí")
            if st.session_state.history:
                stats = advanced_stats(st.session_state.history)
                for pos_name in ['Chục ngàn', 'Ngàn', 'Trăm']:
                    if pos_name in stats:
                        st.write(f"{pos_name}: `{' | '.join(stats[pos_name])}`")

# ================= FOOTER =================
st.markdown("---")
st.caption("""
⚠️ **LƯU Ý QUAN TRỌNG:**
- Tool hỗ trợ tham khảo dựa trên thống kê + AI
- Nhà cái online CÓ THỂ điều khiển kết quả
- Quản lý vốn chặt, không all-in
- Dừng đúng lúc quan trọng hơn thắng

🔐 Bảo mật: API Key lưu trong Secrets, không lộ trên GitHub
☁️ Cloud-ready: Upload/Download DB để đồng bộ dữ liệu
""")
st.caption(f"🕐 Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')} | TITAN v28.0 PRO")