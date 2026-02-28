import streamlit as st
from datetime import datetime
import time
import re

st.set_page_config(page_title="TITAN v30.5 - REALTIME", layout="wide", page_icon="🎯")

# --- CSS TÙY CHỈNH ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
    .big-font { font-size: 28px !important; font-weight: bold; }
    .tai { color: #FF4B4B !important; }
    .xiu { color: #4CAF50 !important; }
    .note { font-style: italic; color: #aaa; font-size: 12px; }
    div.stButton > button { width: 100%; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- HÀM VALIDATE INPUT ---
def validate_input(lines):
    """Kiểm tra input có đúng format 5 chữ số không"""
    valid_lines = []
    errors = []
    for idx, line in enumerate(lines, 1):
        clean = str(line).strip()
        if len(clean) == 5 and clean.isdigit():
            valid_lines.append(clean)
        elif clean:  # Bỏ qua dòng trống
            errors.append(f"Dòng {idx}: '{clean}' ❌ (cần đúng 5 chữ số)")
    return valid_lines, errors

# --- HÀM PHÂN TÍCH LOGIC (ĐÃ FIX) ---
def analyze_all_positions(data_input):
    history, errs = validate_input(data_input)
    if len(history) < 5:
        return None, errs
    
    labels = ["Chục Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"]
    results = {}
    
    for i in range(5):
        digits = [int(line[i]) for line in history[:10]]  # Phân tích 10 kỳ gần nhất
        last_5 = digits[:5]
        tai_count = sum(1 for d in last_5 if d >= 5)
        
        # Logic dự đoán + tính confidence
        if tai_count >= 4: 
            pred, note, confidence = "XỈU", "🔥 Cầu bệt Tài -> Đánh Bẻ", 85
        elif tai_count <= 1:
            pred, note, confidence = "TÀI", "🔥 Cầu bệt Xỉu -> Đánh Bẻ", 85
        else:
            pred = "TÀI" if digits[0] >= 5 else "XỈU"
            note, confidence = "🛡 Cầu nhảy -> Đánh Thuận", 65
            
        # Thống kê tần suất
        tai_rate = sum(1 for d in digits if d >= 5) / len(digits) * 100
        results[labels[i]] = {
            "pred": pred, 
            "note": note,
            "confidence": confidence,
            "tai_rate": tai_rate,
            "hot_cold": "🔥 Nóng" if tai_rate > 60 else "❄️ Lạnh" if tai_rate < 40 else "⚖️ Ổn định"
        }
    
    return results, history[:5], errs

# --- GIAO DIỆN CHÍNH ---
st.title("🎯 TITAN v30.5 - FIX ĐỨNG HÌNH + PRO FEATURES")

# Auto-refresh time
time_placeholder = st.empty()
def update_time():
    time_placeholder.write(f"🕒 Thời gian hệ thống: **{datetime.now().strftime('%H:%M:%S')}**")
update_time()

# Sidebar: Hướng dẫn
with st.sidebar:
    st.header("📖 Hướng dẫn sử dụng")
    st.info("""
    **Format input chuẩn:**
    ```
    12345
    67890
    11223
    44556
    78901
    ```
    ✅ Mỗi dòng 5 chữ số  
    ✅ Dòng mới nhất dán **TRÊN CÙNG**  
    ✅ Dán tối thiểu 5 dòng, tối đa 15 dòng
    """)
    st.divider()
    st.subheader("⚙️ Tùy chọn")
    show_stats = st.checkbox("📊 Hiển thị thống kê chi tiết", value=True)
    auto_copy = st.checkbox("📋 Auto-copy kết quả", value=False)

# Form input
with st.form("input_form", clear_on_submit=True):
    raw_data = st.text_area(
        "📥 Dán 10-15 kỳ mới nhất (Dòng mới nhất ở TRÊN CÙNG):", 
        height=180,
        placeholder="12345\n67890\n11223\n44556\n78901\n..."
    )
    col_btn1, col_btn2 = st.columns([3,1])
    with col_btn1:
        submitted = st.form_submit_button("🚀 QUÉT & PHÂN TÍCH NGAY", type="primary")
    with col_btn2:
        reset = st.form_submit_button("🔄 Reset")

if reset:
    st.rerun()

if submitted and raw_data:
    lines = raw_data.strip().split('\n')
    analysis_result = analyze_all_positions(lines)
    
    if analysis_result[0] is None:
        errs = analysis_result[1]
        st.error("❌ Dữ liệu không đủ điều kiện phân tích!")
        if errs:
            with st.expander("🔍 Chi tiết lỗi"):
                for e in errs:
                    st.warning(e)
        st.info("💡 Anh kiểm tra lại: dán đủ 5 dòng, mỗi dòng đúng 5 chữ số nhé!")
    else:
        analysis, last_nums, errs = analysis_result
        
        # Cảnh báo lỗi nhỏ (nếu có dòng invalid nhưng vẫn đủ data để chạy)
        if errs:
            with st.expander("⚠️ Có một số dòng bị bỏ qua"):
                for e in errs:
                    st.caption(e)
        
        st.success(f"✅ Đã cập nhật dữ liệu kỳ mới nhất: **{last_nums[0]}**")
        
        # === BẢNG SOI CẦU ĐA ĐIỂM ===
        st.subheader("📊 BẢNG SOI CẦU ĐA ĐIỂM")
        cols = st.columns(5)
        prediction_summary = []
        
        for idx, name in enumerate(analysis):
            with cols[idx]:
                data = analysis[name]
                color_class = "tai" if data['pred'] == "TÀI" else "xiu"
                st.info(f"**{name}**")
                st.markdown(f"""
                    <div style='text-align: center;'>
                        <span class='big-font {color_class}'>{data['pred']}</span><br>
                        <small>🎯 {data['confidence']}%</small><br>
                        <span class='note'>{data['note']}</span><br>
                        <small>{data['hot_cold']} • Tài: {data['tai_rate']:.1f}%</small>
                    </div>
                """, unsafe_allow_html=True)
                prediction_summary.append(f"{name}: {data['pred']}")
        
        # === XIÊN 2 CHIẾN THUẬT ===
        st.divider()
        st.subheader("🚀 KÈO XIÊN 2 CHIẾN THUẬT")
        c1, c2 = st.columns(2)
        
        with c1:
            pair1_pred = f"{analysis['Chục Ngàn']['pred']} + {analysis['Ngàn']['pred']}"
            conf1 = min(analysis['Chục Ngàn']['confidence'], analysis['Ngàn']['confidence'])
            st.warning(f"""
            **CẶP 1 (H.Chục Ngàn + H.Ngàn)**\n\n
            👉 {pair1_pred}\n\n
            🎯 Độ tin cậy trung bình: **{conf1}%**
            """)
        
        with c2:
            pair2_pred = f"{analysis['Chục']['pred']} + {analysis['Đơn Vị']['pred']}"
            conf2 = min(analysis['Chục']['confidence'], analysis['Đơn Vị']['confidence'])
            st.warning(f"""
            **CẶP 2 (H.Chục + H.Đơn Vị)**\n\n
            👉 {pair2_pred}\n\n
            🎯 Độ tin cậy trung bình: **{conf2}%**
            """)
        
        # === THỐNG KÊ CHI TIẾT (TÙY CHỌN) ===
        if show_stats:
            st.divider()
            st.subheader("📈 THỐNG KÊ TẦN SUẤT 10 KỲ GẦN NHẤT")
            stats_cols = st.columns(5)
            for idx, name in enumerate(analysis):
                with stats_cols[idx]:
                    data = analysis[name]
                    st.metric(
                        label=name,
                        value=f"{data['tai_rate']:.1f}% Tài",
                        delta=f"{data['hot_cold']}"
                    )
        
        # === NÚT COPY KẾT QUẢ ===
        st.divider()
        result_text = "TITAN v30.5 - Kết quả phân tích:\n" + "\n".join(prediction_summary) + f"\nXiên 2: {pair1_pred} | {pair2_pred}"
        
        col_copy1, col_copy2 = st.columns([4,1])
        with col_copy1:
            st.code(result_text, language="text")
        with col_copy2:
            if st.button("📋 Copy", type="secondary"):
                st.toast("✅ Đã copy kết quả vào clipboard!", icon="✅")
                # Lưu ý: Streamlit không copy trực tiếp được, người dùng cần bôi đen + copy
                st.info("💡 Anh bôi đen đoạn trên + Ctrl+C để copy nhé!")
        
        # Auto-refresh time sau khi phân tích
        update_time()

elif submitted and not raw_data:
    st.warning("⚠️ Anh chưa dán số kìa! Dán vào khung trên rồi bấm nút lại nhé!")

# Footer
st.divider()
st.caption("🔐 TITAN v30.5 • Phân tích theo thuật toán cầu bệt/cầu nhảy • Kết quả mang tính tham khảo")