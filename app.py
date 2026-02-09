import streamlit as st
import collections

st.set_page_config(page_title="TOOL CỨU CÁNH 2026", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #1a1a1a; color: white; }
    .chot-so { background-color: #ffeb3b; color: #000; padding: 20px; border-radius: 15px; text-align: center; font-size: 25px; font-weight: bold; border: 4px solid #f44336; }
    .so-vip { font-size: 90px !important; color: #d32f2f; display: block; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔥 HỆ THỐNG SOI CẦU NGƯỢC (CHỐT NHỊP GAN)")

data_input = st.text_area("👇 Nhập 10-15 ván gần nhất (5 số/dòng):", height=150)

if st.button("🚀 LỌC SỐ ĐIỂM RƠI"):
    lines = [l.strip() for l in data_input.split('\n') if len(l.strip()) == 5]
    
    if len(lines) < 10:
        st.error("❌ Anh ơi, nhập thêm ván đi! Dưới 10 ván AI không tính được nhịp rơi đâu.")
    else:
        st.subheader("🎯 BẢNG CHỐT GIỜ G")
        
        # Phân tích từng hàng
        final_numbers = []
        for i in range(5):
            digits = [int(line[i]) for line in lines]
            # Thuật toán tìm số "vắng mặt" lâu nhất nhưng có dấu hiệu quay lại
            counts = collections.Counter(digits)
            
            # Tìm những số chưa xuất hiện trong 3 ván gần đây nhưng có tổng tần suất ổn định
            recent_digits = digits[:3]
            candidates = [n for n in range(10) if n not in recent_digits]
            
            if not candidates: # Nếu ván nào cũng có thì lấy số ít về nhất
                best_n = sorted(counts, key=counts.get)[0]
            else:
                # Trong các con chưa về, chọn con có tần suất tổng cao nhất (sắp nổ)
                best_n = max(candidates, key=lambda x: counts[x])
            
            final_numbers.append(str(best_n))

        # Hiển thị Bạch Thủ và Song Thủ
        bt_lo = "".join(final_numbers[3:]) # Lấy 2 số cuối làm song thủ
        
        st.markdown(f"""
            <div class='chot-so'>
                <p>🌟 BẠCH THỦ (Hàng Đơn Vị) 🌟</p>
                <span class='so-vip'>{final_numbers[4]}</span>
            </div>
            <br>
            <div class='chot-so' style='background-color: #fff;'>
                <p>🎁 SONG THỦ LÔ (2 Số cuối) 🎁</p>
                <span class='so-vip' style='color: #2e7d32;'>{final_numbers[3]}{final_numbers[4]}</span>
            </div>
        """, unsafe_allow_html=True)

        st.write("---")
        st.write("📊 **Dàn giải mã 5 hàng:** " + " - ".join(final_numbers))

st.warning("⚠️ Chú ý: Tool này đánh theo kiểu 'Săn số sắp nổ'. Anh nên theo đều tay 2-3 ván nếu cầu đang nhịp ngắn nhé!")
