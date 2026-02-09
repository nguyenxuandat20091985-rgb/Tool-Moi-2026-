import streamlit as st
import collections

st.set_page_config(page_title="TOOL KIỂM CHỨNG KẾT QUẢ", layout="wide")

st.markdown("""
    <style>
    .win { color: #28a745; font-weight: bold; font-size: 20px; }
    .loss { color: #dc3545; font-weight: bold; font-size: 20px; }
    .big-box { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 2px solid #343a40; text-align: center; }
    .number-bt { font-size: 80px; color: #ff4b4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ HỆ THỐNG SOI CẦU & KIỂM CHỨNG THẮNG THUA")

# Ô nhập dữ liệu lịch sử
data_input = st.text_area("👇 Nhập kết quả (Ván mới nhất nằm TRÊN CÙNG):", height=200, 
                         placeholder="Ví dụ:\n12345 (Ván mới nhất)\n67890\n...")

if st.button("🚀 PHÂN TÍCH & ĐỐI CHIẾU"):
    lines = [l.strip() for l in data_input.split('\n') if len(l.strip()) == 5]
    
    if len(lines) < 5:
        st.error("❌ Nhập thêm ván đi anh, ít nhất 5 ván mới đối chiếu được!")
    else:
        # 1. PHẦN KIỂM CHỨNG (Check xem ván trước đoán đúng hay sai)
        st.subheader("📋 BẢNG THẨM ĐỊNH 5 VÁN GẦN NHẤT")
        
        win_count = 0
        check_data = []
        
        # Thử đối chiếu cầu hàng đơn vị (số cuối)
        for i in range(min(5, len(lines)-1)):
            current_win_num = lines[i][4] # Số thực tế ván này
            # Thuật toán ván trước đó đã dự đoán (giả định dựa trên nhịp)
            prev_data = lines[i+1:]
            predicted_num = collections.Counter([l[4] for l in prev_data]).most_common(1)[0][0]
            
            status = "✅ ĂN" if current_win_num == predicted_num else "❌ XỊT"
            if status == "✅ ĂN": win_count += 1
            
            check_data.append({"Ván": f"Ván {i+1}", "Số dự đoán": predicted_num, "Kết quả thật": current_win_num, "Trạng thái": status})
        
        st.table(check_data)
        st.write(f"📊 **Tỉ lệ thắng hiện tại của Tool:** {(win_count/5)*100}%")

        # 2. PHẦN CHỐT SỐ CHO VÁN TIẾP THEO
        st.write("---")
        st.markdown("<div class='big-box'>", unsafe_allow_html=True)
        st.write("🎯 **DỰ ĐOÁN VÁN TIẾP THEO (BẠCH THỦ ĐUÔI)**")
        
        # Thuật toán bắt nhịp rơi
        all_last_nums = [l[4] for l in lines]
        final_bt = collections.Counter(all_last_nums).most_common(1)[0][0]
        
        st.markdown(f"<span class='number-bt'>{final_bt}</span>", unsafe_allow_html=True)
        st.write("💡 *Nếu bảng trên đang XỊT nhiều, ván này anh nên nhẹ tay hoặc đánh đảo số!*")
        st.markdown("</div>", unsafe_allow_html=True)

st.warning("⚠️ Giải thích: Tool lấy dữ liệu anh nhập để tự 'soi gương' lại chính nó. Nếu anh thấy nó đang báo XỊT liên tục thì tức là cầu đang gãy, anh đừng theo!")
