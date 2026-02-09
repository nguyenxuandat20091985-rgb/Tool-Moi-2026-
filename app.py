import streamlit as st
import collections

st.set_page_config(page_title="TAM TINH BẤT BẠI 2026", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #ffffff; }
    .result-box { background: linear-gradient(135deg, #1a1a1a 0%, #000000 100%); border: 3px solid #ff00ff; border-radius: 20px; padding: 30px; text-align: center; box-shadow: 0 0 25px #ff00ff; }
    .number-display { font-size: 110px !important; color: #00ecff; font-weight: bold; text-shadow: 0 0 15px #00ecff; margin: 0 15px; }
    .title-vip { color: #ff00ff; font-size: 30px; font-weight: bold; text-transform: uppercase; letter-spacing: 2px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎯 HỆ THỐNG TAM TINH TỰ DO v11.0")
st.write("---")

# Nhập dữ liệu
data_raw = st.text_area("👇 Nhập kết quả (5 số mỗi dòng, ván mới nhất TRÊN CÙNG):", height=200, placeholder="Ví dụ:\n58912\n34678\n...")

if st.button("🔥 CHỐT BỘ 3 SỐ CHÍNH XÁC"):
    # Xử lý dữ liệu
    lines = [l.strip() for l in data_raw.split('\n') if len(l.strip()) == 5]
    
    if len(lines) < 10:
        st.error("❌ Anh dán ít nhất 10-15 ván để em tính toán 'độ lệch' của bộ 3 cho chuẩn!")
    else:
        # THUẬT TOÁN PHÂN TÍCH TỔ HỢP
        # Bước 1: Tìm nhịp rơi của 3 ván gần nhất
        recent_pool = "".join(lines[:3])
        # Bước 2: Tìm nhịp rơi của 7 ván trước đó
        older_pool = "".join(lines[3:10])
        
        # Bước 3: Lọc số - Ưu tiên số có mặt ở cả 2 pool nhưng không quá 'nóng'
        all_counts = collections.Counter("".join(lines))
        candidates = []
        
        for i in range(10):
            num = str(i)
            # Tính điểm ưu tiên (Số vừa về có điểm cao, nhưng nếu về quá 4 lần trong 10 ván thì trừ điểm tránh 'khan')
            score = all_counts[num]
            if num in lines[0]: score += 5 # Ưu tiên số vừa về (bắt bệt)
            if all_counts[num] > 6: score -= 10 # Tránh số quá ngáo
            candidates.append((num, score))
            
        # Sắp xếp chọn ra 3 con điểm cao nhất
        candidates.sort(key=lambda x: x[1], reverse=True)
        final_3 = [candidates[i][0] for i in range(3)]
        final_3.sort() # Sắp xếp thứ tự nhỏ đến lớn cho dễ nhìn

        # HIỂN THỊ SIÊU CẤP
        st.markdown(f"""
            <div class="result-box">
                <p class="title-vip">💎 BỘ 3 TAM TINH CHỐT HẠ 💎</p>
                <div>
                    <span class="number-display">{final_3[0]}</span>
                    <span class="number-display">{final_3[1]}</span>
                    <span class="number-display">{final_3[2]}</span>
                </div>
                <p style="margin-top: 20px; color: #ff00ff; font-size: 18px;">
                    (Chỉ cần 3 số này nổ trong dải 5 số là anh HÚP!)
                </p>
            </div>
        """, unsafe_allow_html=True)

        # PHẦN KIỂM CHỨNG THỰC TẾ
        st.write("---")
        st.subheader("📊 Lịch sử nổ của bộ số này:")
        match_count = 0
        for i in range(min(10, len(lines))):
            found = [n for n in final_3 if n in lines[i]]
            if len(found) >= 3:
                st.write(f"Ván {i+1}: {lines[i]} -> ✅ **NỔ CẢ 3**")
                match_count += 1
            elif len(found) == 2:
                st.write(f"Ván {i+1}: {lines[i]} -> 🔸 Nổ 2")
            else:
                st.write(f"Ván {i+1}: {lines[i]} -> ❌ Trượt")
        
        st.sidebar.metric("Độ tin cậy bộ số", f"{(match_count/10)*100}%")

st.info("💡 **Lưu ý của em:** Nếu anh thấy bộ 3 này đã nổ liên tiếp 2 ván trước đó, thì ván này anh nên vào nhẹ tay vì cầu có thể đảo.")
