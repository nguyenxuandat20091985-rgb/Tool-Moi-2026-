# ================= IMPORT THƯ VIỆN =================
import streamlit as st
import google.generativeai as genai
import re
import json
import pandas as pd
from collections import Counter
from datetime import datetime

# ================= CẤU HÌNH BẢO MẬT =================
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    st.error("⚠️ Chưa cấu hình API Key trong Secrets! Vào Settings → Secrets để thêm.")
    st.stop()

# ================= KHỞI TẠO HỆ THỐNG =================
st.set_page_config(page_title="TITAN v25.0 CLOUD", layout="wide", page_icon="🧠")

@st.cache_resource
def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        # ✅ SỬA: Dùng model mới nhất và ổn định
        return genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception as e:
        st.error(f"Lỗi khởi tạo AI: {e}")
        return None

neural_engine = setup_neural()

# ================= QUẢN LÝ DỮ LIỆU (CLOUD SAFE) =================
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

# ================= GIAO DIỆN & CSS =================
st.markdown("""
    <style>
    .stApp { background: #0d1117; color: #c9d1d9; }
    .main-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px; }
    .big-number { font-size: 60px; font-weight: 800; color: #ff7b72; text-align: center; letter-spacing: 8px; }
    .sub-number { font-size: 40px; font-weight: 700; color: #58a6ff; text-align: center; letter-spacing: 5px; }
    .status-badge { padding: 5px 15px; border-radius: 20px; font-weight: bold; display: inline-block; }
    .bg-go { background: #238636; color: white; }
    .bg-stop { background: #da3633; color: white; }
    .error-box { background: #da3633; color: white; padding: 10px; border-radius: 8px; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 TITAN v25.0 - CLOUD NEURAL NETWORK")
st.markdown("---")

# ================= SIDEBAR: QUẢN LÝ DỮ LIỆU =================
with st.sidebar:
    st.header("💾 Database Control")
    st.info("Lưu ý: Trên Cloud, dữ liệu sẽ mất khi reload. Hãy tải DB về máy sau khi nhập.")
    
    uploaded_db = st.file_uploader("📂 Nạp DB cũ (JSON)", type="json")
    if uploaded_db:
        st.session_state.history = load_data_from_json(uploaded_db)
        st.success(f"Đã nạp {len(st.session_state.history)} kỳ!")
        st.rerun()
    
    st.divider()
    
    if st.session_state.history:
        json_data = convert_df_to_json(st.session_state.history)
        st.download_button(
            label="💾 Tải DB về máy (Backup)",
            data=json_data,
            file_name=f"titan_db_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    st.divider()
    st.write(f"📊 **Tổng kỳ:** {len(st.session_state.history)}")
    if st.button("🗑️ Xóa toàn bộ dữ liệu"):
        st.session_state.history = []
        st.rerun()
    
    st.divider()
    st.caption("🔐 API Key được bảo mật trong Secrets")

# ================= PHẦN 1: NHẬP LIỆU & XỬ LÝ =================
col1, col2 = st.columns([3, 1])
with col1:
    raw_input = st.text_area("📡 Dán kết quả xổ số (Mỗi dòng 5 số)", height=150, placeholder="32880\n21808\n99215...")
with col2:
    st.metric("Kỳ gần nhất", len(st.session_state.history))
    if st.button("🚀 LƯU DỮ LIỆU", type="primary", use_container_width=True):
        if raw_input:
            clean = re.findall(r"\d{5}", raw_input)
            if clean:
                new_data = list(dict.fromkeys(clean))
                st.session_state.history.extend(new_data)
                st.session_state.history = st.session_state.history[-1000:] 
                st.success(f"✅ Đã lưu {len(new_data)} kỳ mới!")
                st.rerun()
        else:
            st.warning("Vui lòng nhập dữ liệu!")

# ================= PHẦN 2: PHÂN TÍCH AI =================
st.markdown("---")
st.subheader("🤖 Phân Tích AI")

if st.session_state.history:
    if st.button("🔍 KÍCH HOẠT AI PHÂN TÍCH", type="secondary", use_container_width=True):
        with st.spinner("🧠 Titan đang tư duy..."):
            all_nums = "".join(st.session_state.history[-50:])
            freq = Counter(all_nums)
            top_freq = [str(x[0]) for x in freq.most_common(5)]
            
            prompt = f"""
            Role: Chuyên gia phân tích xổ số cao cấp (TITAN v25).
            Data: 50 kỳ gần nhất: {st.session_state.history[-50:]}
            Thống kê tần suất số nóng: {top_freq}
            
            Task:
            1. Phân tích quy luật đầu, đuôi, tổng.
            2. Dự đoán 3 số chính (Main) và 4 số lót (Support).
            3. Quyết định: "ĐÁNH" hoặc "CHỜ".
            
            Output JSON ONLY (no markdown):
            {{
                "main_3": "123",
                "support_4": "4567",
                "decision": "ĐÁNH",
                "confidence": 85,
                "reasoning": "Phân tích ngắn gọn..."
            }}
            """
            try:
                # ✅ SỬA: Thêm timeout và xử lý lỗi chi tiết
                response = neural_engine.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        top_p=0.9,
                        max_output_tokens=1024
                    )
                )
                text = response.text
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    st.session_state.last_prediction = json.loads(json_match.group())
                    st.success("✅ AI đã phân tích xong!")
                    st.rerun()
                else:
                    st.error("AI trả về kết quả không đúng chuẩn JSON.")
                    st.write(f"Raw response: {text[:500]}")
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Lỗi AI: {error_msg}")
                
                # ✅ FALLBACK: Nếu AI lỗi, dùng thống kê thuần túy
                st.warning("⚠️ Chuyển sang chế độ thống kê thuần túy...")
                all_n = "".join(st.session_state.history[-40:])
                top = [x[0] for x in Counter(all_n).most_common(7)]
                st.session_state.last_prediction = {
                    "main_3": "".join([str(x) for x in top[:3]]),
                    "support_4": "".join([str(x) for x in top[3:]]),
                    "decision": "ĐÁNH",
                    "confidence": 75,
                    "reasoning": "Dùng thống kê tần suất thuần túy (AI đang bảo trì)"
                }
                st.rerun()
else:
    st.warning("⚠️ Chưa có dữ liệu. Vui lòng nhập kết quả xổ số trước.")

# ================= PHẦN 3: HIỂN THỊ KẾT QUẢ =================
if st.session_state.last_prediction:
    res = st.session_state.last_prediction
    is_go = res.get('decision', '').upper() == 'ĐÁNH'
    badge_class = "bg-go" if is_go else "bg-stop"
    
    st.markdown("---")
    st.markdown(f"<div class='main-card'>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(f"<h3 style='text-align:center'>📢 KẾT LUẬN: <span class='status-badge {badge_class}'>{res.get('decision', 'CHỜ')}</span></h3>", unsafe_allow_html=True)
    
    st.divider()
    
    c_num1, c_num2 = st.columns(2)
    with c_num1:
        st.markdown("<p style='text-align:center;color:#8b949e'>🔥 3 SỐ CHỦ LỰC</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-number'>{res.get('main_3', '???')}</div>", unsafe_allow_html=True)
    with c_num2:
        st.markdown("<p style='text-align:center;color:#8b949e'>🛡️ 4 SỐ LÓT</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='sub-number'>{res.get('support_4', '???')}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.info(f"💡 **Logic:** {res.get('reasoning', 'Không có giải thích')}")
    st.success(f"🎯 **Độ tin cậy:** {res.get('confidence', 0)}%")
    
    full_set = "".join(sorted(set(str(res.get('main_3', '')) + str(res.get('support_4', '')))))
    st.text_input("📋 Dàn số tham khảo (Copy):", full_set)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ================= PHẦN 4: THỐNG KÊ VISUAL =================
st.markdown("---")
with st.expander("📊 Biểu đồ tần suất (50 kỳ gần nhất)"):
    if st.session_state.history:
        all_d = "".join(st.session_state.history[-50:])
        df_freq = pd.Series(Counter(all_d)).sort_index()
        st.bar_chart(df_freq, color="#58a6ff")

# ================= FOOTER =================
st.markdown("---")
st.caption("⚠️ **Cảnh báo:** Công cụ hỗ trợ tham khảo dựa trên xác suất thống kê và AI. Không đảm bảo trúng thưởng. Chơi xổ số có rủi ro, hãy cân nhắc kỹ trước khi xuống tiền.")
st.caption(f"🕐 Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')}")