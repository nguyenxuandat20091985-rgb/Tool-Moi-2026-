import streamlit as st
from datetime import datetime
import time
import re

st.set_page_config(page_title="TITAN v30.7 - STABLE", layout="wide", page_icon="🎯")

# --- SESSION STATE ---
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'raw_input' not in st.session_state:
    st.session_state.raw_input = ""
if 'period_order' not in st.session_state:
    st.session_state.period_order = "newest_top"  # "newest_top" hoặc "newest_bottom"

# --- HÀM CLEAN & VALIDATE INPUT ---
def clean_and_parse_input(raw_text, order="newest_top"):
    """
    Làm sạch input và trả về list số + info lỗi
    order: "newest_top" = kỳ mới nhất ở dòng đầu tiên
    """
    lines = raw_text.strip().split('\n')
    valid_periods = []
    errors = []
    
    for idx, line in enumerate(lines, 1):
        # Remove ký tự không phải số hoặc khoảng trắng
        clean = re.sub(r'[^\d]', '', line.strip())
        
        if len(clean) == 5:
            period_num = len(valid_periods) + 1
            valid_periods.append({
                'period': period_num,
                'value': clean,
                'original_line': idx,
                'digits': [int(d) for d in clean]
            })
        elif clean:  # Có nội dung nhưng không đúng format
            errors.append(f"Dòng {idx}: '{line.strip()}' → Cần đúng 5 chữ số")
    
    # Đảo ngược nếu người dùng chọn newest ở dưới
    if order == "newest_bottom":
        valid_periods.reverse()
        for i, p in enumerate(valid_periods):
            p['period'] = i + 1  # Re-number sau khi đảo
    
    return valid_periods, errors

# --- HÀM TÍNH STATISTICS ---
def calculate_stats(periods, position_idx):
    if not periods:
        return {'tai': 0, 'xiu': 0, 'total': 0, 'avg': 0, 'trend': 'N/A'}
    
    digits = [p['digits'][position_idx] for p in periods]
    return {
        'total': len(digits),
        'tai': sum(1 for d in digits if d >= 5),
        'xiu': sum(1 for d in digits if d < 5),
        'avg': sum(digits) / len(digits),
        'trend': '📈 TĂNG' if digits[0] > digits[-1] else '📉 GIẢM' if digits[0] < digits[-1] else '➡️ ỔN ĐỊNH'
    }

# --- HÀM PHÂN TÍCH LOGIC ---
def analyze_positions(periods):
    if len(periods) < 5:
        return None
    
    labels = ["Chục Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"]
    results = {}
    
    for i in range(5):
        digits = [p['digits'][i] for p in periods]
        last_5 = digits[:5]
        last_3 = digits[:3]
        
        tai_5 = sum(1 for d in last_5 if d >= 5)
        tai_3 = sum(1 for d in last_3 if d >= 5)
        
        # Logic phân tích nâng cao
        if tai_5 >= 4:
            pred, note, conf = "XỈU", "🔥 Bệt Tài → Bẻ cầu", "85%"
        elif tai_5 <= 1:
            pred, note, conf = "TÀI", "🔥 Bệt Xỉu → Bẻ cầu", "85%"
        elif tai_3 == 3:
            pred, note, conf = "XỈU", "📈 3 Tài → Giảm nhiệt", "70%"
        elif tai_3 == 0:
            pred, note, conf = "TÀI", "📉 3 Xỉu → Bật tăng", "70%"
        else:
            avg = sum(last_5) / 5
            pred = "TÀI" if avg >= 4.5 else "XỈU"
            note = "🛡 Cầu nhảy → Theo xu hướng"
            conf = "60%"
        
        results[labels[i]] = {
            "pred": pred, "note": note, "confidence": conf,
            "stats": calculate_stats(periods, i)
        }
    
    return results

# --- GIAO DIỆN ---
st.title("🎯 TITAN v30.7 - FIX NHẢY KỲ")
st.write(f"🕒 {datetime.now().strftime('%H:%M:%S | %d/%m/%Y')}")

# Sidebar: Cài đặt
with st.sidebar:
    st.header("⚙️ Cài đặt nhập liệu")
    
    st.session_state.period_order = st.radio(
        "📌 Thứ tự kỳ:",
        options=["newest_top", "newest_bottom"],
        format_func=lambda x: "✅ Mới nhất ở TRÊN" if x == "newest_top" else "✅ Mới nhất ở DƯỚI",
        index=0 if st.session_state.period_order == "newest_top" else 1
    )
    
    st.info("""
    💡 Mẹo nhập nhanh:
    - Copy từ bảng kết quả
    - Mỗi dòng 1 kỳ (5 chữ số)
    - Ký tự khác số sẽ tự động lọc
    """)
    
    if st.button("🗑️ Reset toàn bộ", use_container_width=True):
        st.session_state.raw_input = ""
        st.session_state.analysis_result = None
        st.rerun()

# Form nhập liệu
with st.form("input_form"):
    st.subheader("📥 Nhập kết quả các kỳ")
    
    raw_data = st.text_area(
        "Dán dữ liệu tại đây:",
        value=st.session_state.raw_input,
        placeholder="Ví dụ:\n95231\n18472\n03659\n74125\n...\n(Lưu ý chọn đúng thứ tự kỳ ở sidebar ⬅️)",
        height=220,
        key="input_area"  # Key cố định tránh bị reset
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        submitted = st.form_submit_button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True)
    with col2:
        preview_btn = st.form_submit_button("👀 Xem trước", use_container_width=True)
    with col3:
        cleared = st.form_submit_button("🗑️ Xoá", use_container_width=True)

# Xử lý nút
if cleared:
    st.session_state.raw_input = ""
    st.session_state.analysis_result = None
    st.rerun()

# Preview dữ liệu (không cần submit)
if preview_btn or (submitted and raw_data):
    if raw_data:
        st.session_state.raw_input = raw_data
        periods, errors = clean_and_parse_input(raw_data, st.session_state.period_order)
        
        # Hiển thị preview bảng
        with st.expander("🔍 Xem trước dữ liệu đã parse", expanded=True):
            if periods:
                # Tạo bảng preview
                preview_data = {
                    "Kỳ #": [p['period'] for p in periods[:10]],  # Show 10 kỳ đầu
                    "Số": [p['value'] for p in periods[:10]],
                    "🔢 Dãy số": [" • ".join(str(d) for d in p['digits']) for p in periods[:10]]
                }
                st.dataframe(preview_data, use_container_width=True, hide_index=True)
                
                if len(periods) > 10:
                    st.caption(f"... và {len(periods) - 10} kỳ nữa")
            else:
                st.warning("Chưa có dữ liệu hợp lệ để xem trước")
            
            if errors:
                st.warning(f"⚠️ {len(errors)} dòng không hợp lệ:")
                for e in errors[:5]:
                    st.caption(f"• {e}")
                if len(errors) > 5:
                    st.caption(f"... và {len(errors) - 5} lỗi khác")
    
    # Nếu bấm Submit thì phân tích
    if submitted:
        periods, errors = clean_and_parse_input(raw_data, st.session_state.period_order)
        
        if len(periods) < 5:
            st.error(f"❌ Cần ít nhất 5 kỳ hợp lệ! Hiện có: {len(periods)}")
            if errors:
                with st.expander("Xem lỗi chi tiết"):
                    for e in errors:
                        st.warning(e)
        else:
            with st.spinner("🔄 Đang phân tích..."):
                time.sleep(0.3)
                analysis = analyze_positions(periods)
                st.session_state.analysis_result = {"analysis": analysis, "periods": periods}
            
            if analysis:
                st.success(f"✅ Phân tích xong {len(periods)} kỳ! Kỳ mới nhất: `{periods[0]['value']}`")
                
                # 📊 BẢNG DỰ ĐOÁN
                st.subheader("📊 KẾT QUẢ PHÂN TÍCH")
                cols = st.columns(5)
                
                labels = ["Chục Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"]
                for idx, name in enumerate(labels):
                    item = analysis[name]
                    is_tai = item['pred'] == "TÀI"
                    color = "#FF4B4B" if is_tai else "#1F77B4"
                    bg = "#FFE5E5" if is_tai else "#E5F0FF"
                    
                    with cols[idx]:
                        st.markdown(f"""
                        <div style='background:{bg}; padding:12px; border-radius:10px; 
                                  text-align:center; border:2px solid {color}; margin:5px'>
                            <b>{name}</b><br>
                            <h2 style='color:{color}; margin:8px 0'>{item['pred']}</h2>
                            <small>🎯 {item['confidence']}</small><br>
                            <small>{item['stats']['trend']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(f"_{item['note']}_")
                        
                        # Mini bar
                        stats = item['stats']
                        if stats['total'] > 0:
                            tai_pct = int(stats['tai'] / stats['total'] * 100)
                            st.progress(tai_pct, text=f"Tài {tai_pct}%")
                
                st.divider()
                
                # 🚀 XIÊN 2
                st.subheader("💎 GỢI Ý XIÊN 2")
                c1, c2 = st.columns(2)
                
                with c1:
                    p1 = f"{analysis['Chục Ngàn']['pred']}+{analysis['Ngàn']['pred']}"
                    c1_conf = min(analysis['Chục Ngàn']['confidence'], analysis['Ngàn']['confidence'])
                    st.metric("Cặp 1: Chục Ngàn + Ngàn", p1, delta=f"🎯 {c1_conf}")
                    st.caption("👉 Đánh khi cầu đang bệt, ưu tiên bẻ")
                
                with c2:
                    p2 = f"{analysis['Chục']['pred']}+{analysis['Đơn Vị']['pred']}"
                    c2_conf = min(analysis['Chục']['confidence'], analysis['Đơn Vị']['confidence'])
                    st.metric("Cặp 2: Chục + Đơn Vị", p2, delta=f"🎯 {c2_conf}")
                    st.caption("👉 Đánh khi cầu nhảy, theo xu hướng")
                
                # 📈 Xu hướng chi tiết
                with st.expander("📈 Xem chi tiết xu hướng từng vị trí"):
                    for name in labels:
                        item = analysis[name]
                        digits = [p['digits'][labels.index(name)] for p in periods[:10]]
                        trend_vis = " → ".join([f"{'🔴' if d>=5 else '🔵'}{d}" for d in digits])
                        st.write(f"**{name}**: {trend_vis}")
                        st.caption(f"Trung bình: {item['stats']['avg']:.2f} | {item['stats']['tai']} Tài / {item['stats']['xiu']} Xỉu")

            else:
                st.error("❌ Không thể phân tích. Kiểm tra lại dữ liệu nhập vào.")

elif not raw_data and submitted:
    st.warning("⚠️ Anh chưa nhập số! Dán dữ liệu vào ô trên rồi bấm nút nhé 🔼")

# Footer
st.markdown("---")
st.caption("🔐 TITAN v30.7 | Fix lỗi nhảy kỳ + Input thông minh | Kết quả tham khảo - Chơi có trách nhiệm 🙏")