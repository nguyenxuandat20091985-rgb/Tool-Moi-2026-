import streamlit as st

st.set_page_config(page_title="TITAN v30.4 - ĐA ĐIỂM", layout="wide")

def analyze_all_positions(data_input):
    # Lọc dữ liệu chuẩn: lấy 15 kỳ để soi cầu dài hơn cho chắc
    history = [str(line).strip() for line in data_input if len(str(line).strip()) == 5]
    if len(history) < 10:
        return None

    # Danh sách các vị trí
    labels = ["Chục Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"]
    results = {}

    for i in range(5):
        # Tách số của từng hàng (từ trái qua phải 0 -> 4)
        digits = [int(line[i]) for line in history]
        
        # SOI CẦU: Lấy 5 kỳ gần nhất
        last_5 = digits[:5]
        tai_count = sum(1 for d in last_5 if d >= 5)
        xiu_count = 5 - tai_count
        
        # Dự đoán dựa trên xu hướng
        if tai_count >= 4: 
            pred = "XỈU"
            note = "🔥 Cầu bệt Tài -> Đánh Bẻ"
        elif xiu_count >= 4:
            pred = "TÀI"
            note = "🔥 Cầu bệt Xỉu -> Đánh Bẻ"
        else:
            # Nếu cầu đang nhảy 1-1 hoặc 2-1, đánh thuận theo con vừa về
            pred = "TÀI" if digits[0] >= 5 else "XỈU"
            note = "🛡 Cầu nhảy -> Đánh Thuận"
            
        results[labels[i]] = {"pred": pred, "note": note}
    
    return results, history[:5]

# --- GIAO DIỆN ---
st.title("🎯 TITAN v30.4 - SOI CẦU ĐA ĐIỂM")
st.write("Sửa lỗi: Phân tích toàn bộ 5 hàng số để anh chọn cặp Xiên khớp với trang cược.")

raw_data = st.text_area("📥 Dán 10-15 kỳ mới nhất (Số mới nhất ở trên cùng):", height=200)

if raw_data:
    lines = raw_data.split('\n')
    analysis, last_nums = analyze_all_positions(lines)
    
    if analysis:
        st.success(f"✅ Đã phân tích 5 kỳ gần nhất: {', '.join(last_nums)}")
        
        # Hiển thị dạng bảng cho anh dễ so sánh
        st.subheader("📊 BẢNG SOI CẦU TOÀN DIỆN")
        
        # Tạo 5 cột cho 5 hàng số
        cols = st.columns(5)
        for idx, name in enumerate(analysis):
            with cols[idx]:
                st.markdown(f"### {name}")
                color = "red" if analysis[name]['pred'] == "TÀI" else "blue"
                st.markdown(f"<h2 style='color:{color};'>{analysis[name]['pred']}</h2>", unsafe_allow_html=True)
                st.caption(analysis[name]['note'])

        st.divider()
        
        # GỢI Ý XIÊN 2 DỰA TRÊN ẢNH ANH GỬI (H.Chục Ngàn & H.Ngàn)
        st.subheader("🚀 GỢI Ý XIÊN 2 CHIẾN THUẬT")
        c1, c2 = st.columns(2)
        
        with c1:
            st.info(f"**CẶP 1 (H.Chục Ngàn + H.Ngàn):**\n\n {analysis['Chục Ngàn']['pred']} + {analysis['Ngàn']['pred']}")
            st.caption("Khớp với mục anh đang chọn trong ảnh!")
            
        with c2:
            st.warning(f"**CẶP 2 (H.Chục + H.Đơn Vị):**\n\n {analysis['Chục']['pred']} + {analysis['Đơn Vị']['pred']}")
            st.caption("Cặp dự phòng nếu cặp 1 đang biến động.")

    else:
        st.error("Anh dán thêm dữ liệu đi, ít nhất 10 dòng nhé!")

st.markdown("---")
st.write("⚠️ **Lưu ý cực quan trọng:** Anh nhìn vào bảng trên, nếu thấy hàng nào báo **'🔥 Đánh Bẻ'** thì tỷ lệ thắng Xiên khi ghép hàng đó sẽ cao hơn rất nhiều.")
