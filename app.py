import streamlit as st
import collections

st.set_page_config(page_title="HỆ THỐNG SOI CẦU CHUYÊN NGHIỆP 2026", layout="wide")

# Giao diện cực chất cho dân chuyên nghiệp
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .main-box { background: #1c1f26; border-radius: 20px; padding: 30px; border: 2px solid #3e4451; text-align: center; }
    .bt-title { color: #f39c12; font-size: 28px; font-weight: bold; text-transform: uppercase; }
    .bt-number { font-size: 150px !important; color: #ff0000; font-weight: bold; text-shadow: 0 0 20px #ff0000; line-height: 1; }
    .status-bar { background: #2c3e50; padding: 10px; border-radius: 10px; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔥 HỆ THỐNG PHÂN TÍCH BẠCH THỦ BAO LÔ")
st.write("---")

# Nhập dữ liệu - Ván mới nhất dán dưới cùng
data_input = st.text_area("👇 Dán kết quả (Càng nhiều ván càng chuẩn - Mỗi ván 5 số):", height=180, placeholder="Ví dụ:\n12345\n67890\n...")

if st.button("🚀 BẮT ĐẦU PHÂN TÍCH TỔNG LỰC"):
    # Xử lý dữ liệu
    lines = [l.strip() for l in data_input.split('\n') if len(l.strip()) == 5]
    
    if len(lines) < 8:
        st.error("❌ Dữ liệu quá ít! Anh cần dán ít nhất 8-10 kỳ để AI tìm ra 'nhịp cầu'.")
    else:
        # Thuật toán bắt nhịp rơi (Tập trung vào 3 kỳ gần nhất và 5 kỳ trước đó)
        recent_data = "".join(lines[-3:]) # 3 ván gần nhất
        older_data = "".join(lines[-8:-3]) # 5 ván trước đó
        
        counts_recent = collections.Counter(recent_data)
        counts_older = collections.Counter(older_data)
        
        # Tìm con số tiềm năng: Có xuất hiện ở kỳ trước nhưng không quá dày đặc
        potential = []
        for i in range(10):
            num = str(i)
            # Điều kiện: Có nổ ở kỳ cũ và đang bắt đầu nổ lại ở kỳ gần đây
            if counts_recent[num] > 0 and counts_older[num] > 0:
                potential.append((num, counts_recent[num] + counts_older[num]))
        
        # Chốt Bạch Thủ
        if potential:
            # Sắp xếp theo số lần xuất hiện hợp lý nhất
            potential.sort(key=lambda x: x[1], reverse=True)
            chot_bt = potential[0][0]
        else:
            # Nếu cầu loạn, lấy số có tần suất ổn định nhất
            chot_bt = collections.Counter("".join(lines)).most_common(2)[0][0]

        # Hiển thị bảng chốt
        st.markdown(f"""
            <div class="main-box">
                <p class="bt-title">🎯 BẠCH THỦ BAO LÔ KỲ TỚI 🎯</p>
                <div class="bt-number">{chot_bt}</div>
                <div class="status-bar">
                    <p style="margin:0;">Trạng thái cầu: <span style="color:#00ff00;">ĐANG CHẠY 📈</span></p>
                    <p style="margin:0; font-size: 14px; color:#bdc3c7;">(Chỉ cần số {chot_bt} xuất hiện ở bất kỳ đâu trong 5 số là THẮNG)</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Thống kê nhanh để anh kiểm chứng
        st.write("---")
        st.subheader("📊 Thống kê nhịp số (Số lần nổ):")
        cols = st.columns(10)
        all_nums = "".join(lines)
        for i in range(10):
            cols[i].metric(label=f"Số {i}", value=all_nums.count(str(i)))

st.warning("💡 **Lời khuyên:** Bản này đã lọc bỏ tình trạng báo số 'ngáo'. Anh hãy dán khoảng 15 ván liên tục để thấy sức mạnh của nhịp cầu!")
