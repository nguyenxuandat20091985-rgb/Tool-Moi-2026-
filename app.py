import streamlit as st
from datetime import datetime

st.set_page_config(page_title="TITAN v30.5 - REALTIME", layout="wide")

# Hàm phân tích logic
def analyze_all_positions(data_input):
    history = [str(line).strip() for line in data_input if len(str(line).strip()) == 5]
    if len(history) < 5:
        return None

    labels = ["Chục Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"]
    results = {}
    for i in range(5):
        digits = [int(line[i]) for line in history]
        last_5 = digits[:5]
        tai_count = sum(1 for d in last_5 if d >= 5)
        
        if tai_count >= 4: 
            pred, note = "XỈU", "🔥 Cầu bệt Tài -> Đánh Bẻ"
        elif tai_count <= 1:
            pred, note = "TÀI", "🔥 Cầu bệt Xỉu -> Đánh Bẻ"
        else:
            pred = "TÀI" if digits[0] >= 5 else "XỈU"
            note = "🛡 Cầu nhảy -> Đánh Thuận"
        results[labels[i]] = {"pred": pred, "note": note}
    
    return results, history[:5]

# --- GIAO DIỆN ---
st.title("🎯 TITAN v30.5 - FIX ĐỨNG HÌNH")
st.write(f"🕒 Thời gian hệ thống: {datetime.now().strftime('%H:%M:%S')}")

# Sử dụng form để ép dữ liệu phải "Submit" mới chạy
with st.form("input_form"):
    raw_data = st.text_area("📥 Dán 10-15 kỳ mới nhất (Dòng mới nhất ở TRÊN CÙNG):", height=180)
    submitted = st.form_submit_button("🚀 QUÉT & PHÂN TÍCH NGAY")

if submitted and raw_data:
    lines = raw_data.split('\n')
    analysis_data = analyze_all_positions(lines)
    
    if analysis_data:
        analysis, last_nums = analysis_data
        st.success(f"✅ Đã cập nhật dữ liệu kỳ mới nhất: {last_nums[0]}")
        
        st.subheader("📊 BẢNG SOI CẦU ĐA ĐIỂM")
        cols = st.columns(5)
        for idx, name in enumerate(analysis):
            with cols[idx]:
                st.info(f"**{name}**")
                color = "#FF4B4B" if analysis[name]['pred'] == "TÀI" else "#1F77B4"
                st.markdown(f"<h1 style='text-align: center; color: {color};'>{analysis[name]['pred']}</h1>", unsafe_allow_html=True)
                st.caption(f"<p style='text-align: center;'>{analysis[name]['note']}</p>", unsafe_allow_html=True)

        st.divider()
        
        # HIỂN THỊ XIÊN 2 TO RÕ
        st.subheader("🚀 KÈO XIÊN 2 CHIẾN THUẬT")
        c1, c2 = st.columns(2)
        with c1:
            st.warning(f"**CẶP 1 (H.Chục Ngàn + H.Ngàn)**\n\n👉 {analysis['Chục Ngàn']['pred']} + {analysis['Ngàn']['pred']}")
        with c2:
            st.warning(f"**CẶP 2 (H.Chục + H.Đơn Vị)**\n\n👉 {analysis['Chục']['pred']} + {analysis['Đơn Vị']['pred']}")
    else:
        st.error("Dữ liệu không khớp! Anh kiểm tra lại xem có copy thiếu số nào không.")

elif not raw_data and submitted:
    st.warning("Anh chưa dán số kìa, dán vào rồi bấm nút lại nhé!")
