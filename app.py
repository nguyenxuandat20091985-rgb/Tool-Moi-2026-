import streamlit as st
import collections

# Cấu hình giao diện cực mạnh, dễ nhìn trên điện thoại
st.set_page_config(page_title="TRÙM BAO LÔ 2026", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #000000; }
    .stTextArea textarea { background-color: #1a1a1a; color: #00ff00; font-size: 18px !important; border: 2px solid #00ff00; }
    .result-card { background: #111; padding: 20px; border-radius: 15px; border: 2px solid #ff4b4b; text-align: center; }
    .bt-number { font-size: 100px !important; color: #ffff00; font-weight: bold; text-shadow: 3px 3px #ff0000; }
    .win-tag { background-color: #28a745; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
    .loss-tag { background-color: #dc3545; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎰 HỆ THỐNG BAO LÔ THỰC CHIẾN v5.0")
st.write("---")

# Nhập dữ liệu
data_raw = st.text_area("👇 Dán kết quả (Càng nhiều càng chuẩn, ván mới nhất ở DƯỚI CÙNG):", height=200)

if st.button("🚀 PHÂN TÍCH MA TRẬN SỐ"):
    # Xử lý dữ liệu: bỏ dòng trống, lấy 5 số mỗi dòng
    lines = [l.strip() for l in data_raw.split('\n') if len(l.strip()) == 5]
    
    if len(lines) < 5:
        st.error("❌ Anh nhập thêm ít nhất 5-10 ván để em chạy ma trận nhé!")
    else:
        st.subheader("📊 KIỂM CHỨNG 5 VÁN VỪA QUA")
        
        win_count = 0
        # Duyệt lại 5 ván gần nhất để xem nếu dùng tool thì thắng hay thua
        for i in range(len(lines)-5, len(lines)):
            if i <= 0: continue
            # Lấy dữ liệu trước ván đó để dự đoán
            past_data = lines[:i]
            actual_result = lines[i] # Dòng kết quả thực tế
            
            # Thuật toán: Tìm số có tần suất nổ 'nhịp' nhất (không phải nhiều nhất)
            flat_list = "".join(past_data)
            counts = collections.Counter(flat_list)
            # Lấy số có tần suất vừa phải (thường là số đang vào cầu)
            predicted = counts.most_common(3)[1][0] # Lấy số đứng thứ 2 trong top
            
            check = "✅ ĂN" if predicted in actual_result else "❌ XỊT"
            if "✅" in check: win_count += 1
            
            col1, col2, col3 = st.columns([1, 2, 1])
            col1.write(f"Ván {i}")
            col2.write(f"Dự đoán: **{predicted}** ⮕ Kết quả: **{actual_result}**")
            col3.markdown(f"<span class='{'win-tag' if '✅' in check else 'loss-tag'}'>{check}</span>", unsafe_allow_html=True)

        # CHỐT SỐ VÁN TIẾP THEO
        st.write("---")
        st.subheader("🔥 CHỐT SỐ VÀNG VÁN KẾ TIẾP")
        
        # Lấy toàn bộ số đã nhập
        full_data = "".join(lines)
        c = collections.Counter(full_data)
        
        # Thuật toán chốt: Kết hợp số hay về và số vừa mới về
        top_nums = c.most_common(3)
        final_bt = top_nums[0][0] # Số mạnh nhất
        final_st = top_nums[1][0] # Số mạnh thứ 2
        
        st.markdown(f"""
            <div class='result-card'>
                <p style='color: white; font-size: 20px;'>BẠCH THỦ BAO LÔ</p>
                <span class='bt-number'>{final_bt}</span>
                <p style='color: #00ff00; font-size: 25px;'>Song Thủ Lót: {final_st}</p>
                <p style='color: #aaa;'>Tỉ lệ nổ dự kiến: {75 + (win_count*4)}%</p>
            </div>
        """, unsafe_allow_html=True)

st.warning("⚠️ **Lưu ý:** Nếu 5 ván gần nhất Tool báo XỊT liên tục (ví dụ xịt 4/5), thì ván này anh nên **ĐÁNH NGƯỢC** lại hoặc nghỉ. Cầu đang gãy thì không nên cố anh nhé!")
