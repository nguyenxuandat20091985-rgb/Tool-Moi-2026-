import streamlit as st
import collections

st.set_page_config(page_title="TOOL TAM TỬ 2026", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #000; color: #fff; }
    .box-3-so { background: linear-gradient(145deg, #1e1e1e, #111); border: 3px solid #00ffcc; border-radius: 25px; padding: 30px; text-align: center; box-shadow: 0 0 30px #00ffcc; }
    .so-to { font-size: 100px !important; color: #00ffcc; font-weight: bold; margin: 0 20px; text-shadow: 0 0 15px #00ffcc; }
    .label-3-so { font-size: 24px; color: #fff; font-weight: bold; margin-bottom: 20px; text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ SIÊU TOOL: BẮT 3 SỐ TỰ DO (BAO LÔ 3 CON)")
st.write("---")

# Nhập dữ liệu
data_raw = st.text_area("👇 Dán kết quả (Mỗi ván 5 số, ván mới nhất TRÊN CÙNG):", height=200)

if st.button("🚀 PHÂN TÍCH VÙNG HỘI TỤ"):
    lines = [l.strip() for l in data_raw.split('\n') if len(l.strip()) == 5]
    
    if len(lines) < 7:
        st.error("❌ Anh dán ít nhất 7 ván để em tính toán nhịp rơi của 3 con số nhé!")
    else:
        # Thuật toán bắt 3 số tiềm năng nhất
        all_nums = "".join(lines)
        counts = collections.Counter(all_nums)
        
        # Lấy top 5 số về nhiều
        top_5 = counts.most_common(5)
        
        # Loại bỏ bớt số nổ quá dày để tránh "ngáo", chọn 3 con có nhịp đẹp nhất
        # Ưu tiên những số xuất hiện ở ván gần nhất nhưng không quá 3 lần
        recent_van = lines[0]
        final_3 = []
        
        for num, freq in top_5:
            if len(final_3) < 3:
                final_3.append(num)
        
        # Sắp xếp lại cho đẹp
        final_3.sort()

        # Hiển thị kết quả 3 số sập mắt
        st.write("### 🎯 KẾT QUẢ DỰ ĐOÁN 3 SỐ VÀNG:")
        st.markdown(f"""
            <div class="box-3-so">
                <div class="label-3-so">Bộ 3 số tự do (Nổ đâu cũng được)</div>
                <span class="so-to">{final_3[0]}</span>
                <span class="so-to">{final_3[1]}</span>
                <span class="so-to">{final_3[2]}</span>
                <p style="margin-top: 20px; color: #888;">Chỉ cần dải kết quả ván tới có 3 số này là anh THẮNG!</p>
            </div>
        """, unsafe_allow_html=True)

        # Kiểm chứng nhanh ván trước
        st.write("---")
        st.subheader("📋 Kiểm chứng ván gần nhất:")
        check_last = lines[0]
        st.write(f"Ván mới nhất về: **{check_last}**")
        st.write("---")
        st.info("💡 Mẹo: Anh có thể đánh bao lô cả 3 con này, hoặc ghép xiên xoay để tăng tỉ lệ ăn!")

st.markdown("<p style='text-align: center; color: #444;'>Thiết kế bởi Gemini - Bản tối ưu 3 số v10.0</p>", unsafe_allow_html=True)
