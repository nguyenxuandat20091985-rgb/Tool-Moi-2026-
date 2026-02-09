import streamlit as st

st.set_page_config(page_title="TOOL THỰC CHIẾN 2026", layout="wide")

st.markdown("""
    <style>
    .win-box { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; font-weight: bold; }
    .loss-box { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; }
    .final-bt { font-size: 100px; color: yellow; background: black; text-align: center; border-radius: 20px; border: 5px solid red; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔥 TOOL THỰC CHIẾN: SOI CẦU & BÁO THẮNG THUA")

data_input = st.text_area("👇 Nhập kết quả (Ván mới nhất dán TRÊN CÙNG):", height=150)

if st.button("📊 KIỂM TRA & CHỐT SỐ"):
    lines = [l.strip() for l in data_input.split('\n') if len(l.strip()) == 5]
    
    if len(lines) < 5:
        st.warning("Anh nhập thêm tầm 5-10 ván để em check xem cầu đang chạy thế nào nhé!")
    else:
        # 1. BẢNG THỐNG KÊ THẮNG THUA THỰC TẾ
        st.subheader("📝 NHẬT KÝ THẮNG THUA (10 VÁN GẦN ĐÂY)")
        
        # Quy luật bóng số: 0-5, 1-6, 2-7, 3-8, 4-9
        bong_so = {0:5, 5:0, 1:6, 6:1, 2:7, 7:2, 3:8, 8:3, 4:9, 9:4}
        
        wins = 0
        total_check = min(10, len(lines)-1)
        
        for i in range(total_check):
            kq_that = int(lines[i][4]) # Số cuối ván này
            so_du_doan = bong_so[int(lines[i+1][4])] # Soi từ ván trước theo bóng
            
            col1, col2, col3 = st.columns([1,2,1])
            col1.write(f"Ván {i+1}")
            if kq_that == so_du_doan:
                col2.markdown(f"<div class='win-box'>Dự đoán: {so_du_doan} ⮕ THỰC TẾ: {kq_that} (TIỀN VỀ 💰)</div>", unsafe_allow_html=True)
                wins += 1
            else:
                col2.markdown(f"<div class='loss-box'>Dự đoán: {so_du_doan} ⮕ THỰC TẾ: {kq_that} (TRƯỢT 💀)</div>", unsafe_allow_html=True)
        
        st.write(f"### 📈 Hiệu suất cầu: {wins}/{total_check} ván thắng")

        # 2. CHỐT SỐ VÁN TIẾP THEO
        st.write("---")
        st.subheader("🎯 CHỐT BẠCH THỦ VÁN TỚI")
        
        last_num = int(lines[0][4])
        chot_bt = bong_so[last_num] # Chốt theo bóng của ván vừa xong nhất
        
        st.markdown(f"<div class='final-bt'>{chot_bt}</div>", unsafe_allow_html=True)
        st.write(f"💡 Giải mã: Ván vừa rồi về **{last_num}**, theo cầu bóng âm dương thì ván tới tỷ lệ nổ **{chot_bt}** cực cao.")

st.info("Anh để ý: Nếu bảng Nhật ký hiện toàn 'TRƯỢT 💀' thì là cầu bóng đang gãy, anh nghỉ ván này. Nếu thấy 'TIỀN VỀ 💰' đang thông thì cứ thế mà quất!")
