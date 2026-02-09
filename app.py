import streamlit as st
import collections

st.set_page_config(page_title="TOOL BẮT CẦU BỆT 2026", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #ffffff; }
    .bet-box { background: linear-gradient(90deg, #ff0000 0%, #000000 100%); padding: 20px; border-radius: 15px; border-left: 10px solid #ffff00; margin: 20px 0; }
    .so-chot { font-size: 120px !important; color: #ffff00; font-weight: bold; line-height: 1; text-shadow: 5px 5px #ff0000; }
    .detected-text { font-size: 24px; color: #00ff00; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏹 THẦN TOÁN v6.0: CHUYÊN SĂN CẦU BỆT & BAO LÔ")

# Nhập liệu - Ván mới nhất dán dưới cùng
data_input = st.text_area("👇 Dán danh sách kết quả (Ván mới nhất nằm ở dòng CUỐI CÙNG):", height=200)

if st.button("🚀 QUÉT CẦU & CHỐT SỐ"):
    lines = [l.strip() for l in data_input.split('\n') if len(l.strip()) == 5]
    
    if len(lines) < 5:
        st.error("❌ Anh dán thêm kết quả đi, ít nhất 5 ván em mới soi được cầu bệt!")
    else:
        # Lấy dữ liệu ván gần nhất để check bệt
        last_line = lines[-1]
        all_data_str = "".join(lines)
        
        # --- THUẬT TOÁN NHẬN DIỆN CẦU BỆT ---
        st.subheader("🕵️ PHÂN TÍCH NHỊP CẦU")
        
        # Tìm con số bệt mạnh nhất (vừa về kỳ trước và có tần suất cao)
        counts = collections.Counter(all_data_str)
        most_common_global = counts.most_common(5)
        
        # Kiểm tra xem có con nào trong ván vừa rồi đang bệt không
        bet_candidate = None
        for num in last_line:
            # Nếu số này vừa về và 3 ván gần đây nổ từ 2 lần trở lên -> CẦU BỆT
            recent_3_vans = "".join(lines[-3:])
            if recent_3_vans.count(num) >= 2:
                bet_candidate = num
                break
        
        # Nếu không thấy bệt, chọn số có nhịp rơi đẹp nhất (tránh con số 9 nếu nó đang 'ngáo')
        if bet_candidate:
            final_selection = bet_candidate
            status_msg = f"🔥 PHÁT HIỆN CẦU BỆT CON: {final_selection}"
        else:
            # Thuật toán lấy số 'Đang lên' (không lấy con cao nhất để tránh kẹt số)
            final_selection = most_common_global[1][0] if most_common_global[0][0] == '9' else most_common_global[0][0]
            status_msg = "📉 CẦU ĐANG ĐI NHỊP ĐẢO - CHỐT SỐ RƠI"

        # HIỂN THỊ KẾT QUẢ
        st.markdown(f"""
            <div class="bet-box">
                <p class="detected-text">{status_msg}</p>
                <div style="text-align: center;">
                    <span style="font-size: 20px;">BẠCH THỦ BAO LÔ (VỀ LÀ ĂN)</span><br>
                    <span class="so-chot">{final_selection}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # BẢNG ĐỐI CHIẾU NHANH
        st.write("---")
        st.write("📊 **Thống kê nhanh:**")
        cols = st.columns(5)
        for i, (num, freq) in enumerate(most_common_global):
            cols[i].metric(label=f"Số {num}", value=f"{freq} lần")

st.info("💡 **Kinh nghiệm:** Nếu anh thấy nó báo 'CẦU BỆT', anh có thể vào tiền mạnh tay hơn một chút. Nếu nó báo 'NHỊP ĐẢO', anh nên đánh nhẹ tay để thăm dò.")
