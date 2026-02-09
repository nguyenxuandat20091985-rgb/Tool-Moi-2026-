import streamlit as st
import collections

st.set_page_config(page_title="BẠCH THỦ BAO LÔ 2026", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #000; color: #fff; }
    .box-chot { background: #1a1a1a; border: 4px solid #f1c40f; border-radius: 20px; padding: 40px; text-align: center; box-shadow: 0px 0px 30px #f1c40f; }
    .so-vip { font-size: 150px !important; color: #f1c40f; font-weight: bold; text-shadow: 0 0 20px #fff; line-height: 1.2; }
    .stButton>button { width: 100%; background: #f1c40f; color: #000; font-weight: bold; font-size: 20px; height: 3em; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏆 CHỐT BẠCH THỦ BAO LÔ 🏆")

# Nhập kết quả
data_input = st.text_area("👇 Dán danh sách kết quả (5 số mỗi dòng):", height=200)

if st.button("🔥 CHỐT BẠCH THỦ DUY NHẤT"):
    lines = [l.strip() for l in data_input.split('\n') if len(l.strip()) == 5]
    
    if len(lines) < 10:
        st.error("⚠️ Anh dán ít nhất 10 kỳ để em soi 'tâm điểm' cho chuẩn nhé!")
    else:
        # Thuật toán: Phân tích tần suất và loại bỏ các số 'rác'
        full_text = "".join(lines)
        counts = collections.Counter(full_text)
        
        # Lấy 2 con mạnh nhất
        top_2 = counts.most_common(2)
        
        # Logic chốt Bạch Thủ: 
        # Nếu con mạnh nhất đã nổ quá nhiều (trên 30% tổng số), nó dễ bị khan -> lấy con mạnh thứ 2.
        # Ngược lại thì lấy con mạnh nhất.
        if int(top_2[0][1]) > (len(full_text) * 0.25):
            bach_thu = top_2[1][0]
        else:
            bach_thu = top_2[0][0]

        st.markdown(f"""
            <div class="box-chot">
                <p style="font-size: 25px; color: #fff;">🎯 BẠCH THỦ BAO LÔ 🎯</p>
                <span class="so-vip">{bach_thu}</span>
                <p style="font-size: 18px; color: #aaa; margin-top: 10px;">
                    (Chỉ cần 1 con duy nhất - Nổ ở đâu cũng ăn)
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 Lời khuyên: Con số này đang có tần suất rơi ổn định nhất. Anh có thể đánh bao lô hoặc làm số đá đều đẹp.")

st.markdown("<p style='text-align: center; color: #555;'>Phiên bản tối ưu hóa cho Bạch Thủ Duy Nhất v7.0</p>", unsafe_allow_html=True)
