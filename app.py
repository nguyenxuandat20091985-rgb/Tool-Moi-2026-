import streamlit as st
from datetime import datetime
import time

st.set_page_config(page_title="TITAN v30.6 - PRO", layout="wide", page_icon="🎯")

# --- SESSION STATE ---
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'last_input' not in st.session_state:
    st.session_state.last_input = ""

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
            errors.append(f"Dòng {idx}: '{clean}' ❌ (cần 5 chữ số)")
    return valid_lines, errors

# --- HÀM TÍNH STATISTICS ---
def calculate_stats(history, position_idx):
    """Tính thống kê cho 1 vị trí"""
    digits = [int(line[position_idx]) for line in history]
    return {
        'total': len(digits),
        'tai': sum(1 for d in digits if d >= 5),
        'xiu': sum(1 for d in digits if d < 5),
        'avg': sum(digits) / len(digits) if digits else 0,
        'last_3_trend': "TÀI" if sum(int(h[position_idx]) for h in history[:3]) >= 8 else "XỈU"
    }

# --- HÀM PHÂN TÍCH LOGIC (ĐÃ CẢI TIẾN) ---
def analyze_all_positions(data_input):
    history, errors = validate_input(data_input)
    
    if len(history) < 5:
        return None, "Cần ít nhất 5 kỳ hợp lệ!", errors
    
    labels = ["Chục Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"]
    results = {}
    
    for i in range(5):
        digits = [int(line[i]) for line in history]
        last_5 = digits[:5]
        last_3 = digits[:3]
        
        tai_count_5 = sum(1 for d in last_5 if d >= 5)
        tai_count_3 = sum(1 for d in last_3 if d >= 5)
        
        # Logic nâng cấp: kết hợp xu hướng ngắn + dài
        if tai_count_5 >= 4: 
            pred, note, confidence = "XỈU", "🔥 Cầu bệt Tài -> Bẻ cầu", "85%"
        elif tai_count_5 <= 1:
            pred, note, confidence = "TÀI", "🔥 Cầu bệt Xỉu -> Bẻ cầu", "85%"
        elif tai_count_3 == 3:
            pred, note, confidence = "XỈU", "📈 3 Tài liên tiếp -> Giảm", "70%"
        elif tai_count_3 == 0:
            pred, note, confidence = "TÀI", "📉 3 Xỉu liên tiếp -> Tăng", "70%"
        else:
            # Xu hướng trung bình
            avg = sum(last_5) / 5
            pred = "TÀI" if avg >= 4.5 else "XỈU"
            note = "🛡 Cầu nhảy -> Theo xu hướng"
            confidence = "60%"
            
        results[labels[i]] = {
            "pred": pred, 
            "note": note,
            "confidence": confidence,
            "stats": calculate_stats(history, i)
        }
    
    return results, history[:5], errors

# --- GIAO DIỆN CHÍNH ---
st.title("🎯 TITAN v30.6 - PRO EDITION")
st.write(f"🕒 Cập nhật: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}")

# Sidebar: Hướng dẫn
with st.sidebar:
    st.header("📖 Hướng dẫn sử dụng")
    st.info("""
    1. Dán 10-15 kỳ mới nhất  
    2. **Kỳ mới nhất ở TRÊN CÙNG**  
    3. Mỗi dòng = 5 chữ số (VD: 12345)  
    4. Bấm "🚀 QUÉT & PHÂN TÍCH"
    """)
    st.markdown("---")
    st.subheader("⚙️ Tuỳ chọn")
    auto_clear = st.checkbox("🗑️ Tự động xoá sau khi phân tích", value=False)

# Form nhập liệu
with st.form("input_form", clear_on_submit=auto_clear):
    raw_data = st.text_area(
        "📥 Dán dữ liệu tại đây:", 
        value=st.session_state.last_input,
        placeholder="95231\n18472\n03659\n...\n(Nhớ: kỳ mới nhất ở trên)",
        height=200
    )
    
    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn1:
        submitted = st.form_submit_button("🚀 QUÉT & PHÂN TÍCH NGAY", type="primary", use_container_width=True)
    with col_btn2:
        cleared = st.form_submit_button("🗑️ XOÁ TRỐNG", use_container_width=True)

# Xử lý clear
if cleared:
    st.session_state.last_input = ""
    st.session_state.analysis_result = None
    st.rerun()

# Xử lý phân tích
if submitted and raw_data:
    st.session_state.last_input = raw_data
    lines = raw_data.split('\n')
    
    with st.spinner("🔍 Đang phân tích dữ liệu..."):
        time.sleep(0.5)  # Hiệu ứng loading
        analysis_data, last_nums, errors = analyze_all_positions(lines)
    
    # Hiển thị warning nếu có lỗi input
    if errors:
        with st.expander("⚠️ Cảnh báo dữ liệu không hợp lệ", expanded=False):
            for err in errors:
                st.warning(err)
    
    if analysis_data:
        analysis, last_nums = analysis_data, last_nums
        st.session_state.analysis_result = analysis
        
        # ✅ Success message
        st.success(f"✅ Phân tích xong! Kỳ mới nhất: `{last_nums[0]}`")
        
        # 📊 BẢNG SOI CẦU
        st.subheader("📊 BẢNG DỰ ĐOÁN ĐA VỊ TRÍ")
        cols = st.columns(5)
        
        for idx, name in enumerate(analysis):
            with cols[idx]:
                item = analysis[name]
                is_tai = item['pred'] == "TÀI"
                color = "#FF4B4B" if is_tai else "#1F77B4"
                bg_color = "#FFE5E5" if is_tai else "#E5F0FF"
                
                st.markdown(f"""
                <div style='background:{bg_color}; padding:10px; border-radius:8px; text-align:center; border:1px solid {color}'>
                    <b>{name}</b><br>
                    <h2 style='color:{color}; margin:5px 0'>{item['pred']}</h2>
                    <small>🎯 Độ tin cậy: {item['confidence']}</small>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"_{item['note']}_")
                
                # Mini stats
                stats = item['stats']
                st.progress(int(stats['tai'] / stats['total'] * 100) if stats['total'] > 0 else 0)
                st.caption(f"Tài: {stats['tai']}/{stats['total']} | TB: {stats['avg']:.1f}")

        st.divider()
        
        # 🚀 KÈO XIÊN 2
        st.subheader("🚀 GỢI Ý XIÊN 2 CHIẾN THUẬT")
        c1, c2 = st.columns(2)
        
        with c1:
            pair1_pred = f"{analysis['Chục Ngàn']['pred']} + {analysis['Ngàn']['pred']}"
            conf1 = min(analysis['Chục Ngàn']['confidence'], analysis['Ngàn']['confidence'])
            st.metric(label="💎 CẶP 1: Chục Ngàn + Ngàn", value=pair1_pred, delta=f"🎯 {conf1}")
            st.info("👉 Phù hợp đánh lót ngược nếu cầu đang bệt")
            
        with c2:
            pair2_pred = f"{analysis['Chục']['pred']} + {analysis['Đơn Vị']['pred']}"
            conf2 = min(analysis['Chục']['confidence'], analysis['Đơn Vị']['confidence'])
            st.metric(label="💎 CẶP 2: Chục + Đơn Vị", value=pair2_pred, delta=f"🎯 {conf2}")
            st.info("👉 Phù hợp đánh theo xu hướng khi cầu nhảy")
        
        # 📈 Pattern Visualization
        st.subheader("📈 BIỂU ĐỒ XU HƯỚNG 5 KỲ GẦN")
        pattern_data = {name: [int(line[i] if line[i].isdigit() else 0) for line in last_nums] for i, name in enumerate(analysis)}
        
        for name in analysis:
            digits = pattern_data[name]
            trend_str = " → ".join([f"{'🔴' if d>=5 else '🔵'}{d}" for d in digits])
            st.caption(f"**{name}**: {trend_str}")

    else:
        st.error("❌ Dữ liệu không đủ điều kiện phân tích!")
        if errors:
            with st.expander("Xem chi tiết lỗi"):
                for e in errors:
                    st.write(f"• {e}")

elif not raw_data and submitted:
    st.warning("⚠️ Anh chưa dán số! Dán dữ liệu vào ô trên rồi bấm nút nhé 🔼")

# Footer
st.markdown("---")
st.caption("🔐 TITAN v30.6 | Phân tích theo thuật toán bẻ cầu + xu hướng | Kết quả mang tính tham khảo")