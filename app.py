import streamlit as st
import collections
import time
import numpy as np
import pandas as pd
from datetime import datetime

# =============== CLASS CHÍNH ===============
class LotteryAIAnalyzer:
    def __init__(self):
        self.history = []
        
    def eliminate_risk_numbers(self, data: str):
        """Loại 3 số rủi ro cao nhất"""
        nums = list(filter(str.isdigit, data))
        
        if len(nums) < 10:
            return [], []
            
        counts = collections.Counter(nums)
        
        # 1. Tìm số lạnh (xuất hiện ít nhất)
        cold_numbers = sorted([str(i) for i in range(10)], 
                             key=lambda x: counts.get(x, 0))[:3]
        
        # 2. Tính điểm rủi ro
        risk_scores = {}
        for num in range(10):
            num_str = str(num)
            freq = counts.get(num_str, 0)
            last_20 = nums[-20:] if len(nums) >= 20 else nums
            
            # Điểm rủi ro dựa trên:
            # - Tần suất thấp
            # - Không xuất hiện gần đây
            risk = 0
            if freq == 0:
                risk += 3
            elif freq <= 1:
                risk += 2
                
            if num_str not in last_20:
                risk += 2
                
            risk_scores[num_str] = risk
        
        # Lấy 3 số rủi ro cao nhất
        eliminated = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        eliminated_nums = [num for num, _ in eliminated]
        
        # 7 số còn lại
        remaining = [str(i) for i in range(10) if str(i) not in eliminated_nums]
        
        return eliminated_nums, remaining
    
    def select_top_three(self, remaining_nums: List[str], data: str):
        """Chọn 3 số tốt nhất từ 7 số còn lại"""
        nums = list(filter(str.isdigit, data))
        
        if not nums:
            return remaining_nums[:3]
            
        last_num = nums[-1]
        
        # Lý thuyết bóng đề
        bong_duong = {"0": "5", "1": "6", "2": "7", "3": "8", "4": "9",
                      "5": "0", "6": "1", "7": "2", "8": "3", "9": "4"}
        bong_am = {"0": "7", "1": "4", "2": "9", "3": "6", "4": "1",
                   "5": "8", "6": "3", "7": "0", "8": "5", "9": "2"}
        
        candidates = []
        
        # Ưu tiên bóng
        for bong_num in [bong_duong.get(last_num), bong_am.get(last_num)]:
            if bong_num and bong_num in remaining_nums and bong_num not in candidates:
                candidates.append(bong_num)
        
        # Ưu tiên số liền kề
        for adj_num in [str((int(last_num) + 1) % 10), str((int(last_num) - 1) % 10)]:
            if adj_num in remaining_nums and adj_num not in candidates:
                candidates.append(adj_num)
        
        # Nếu chưa đủ, lấy từ remaining
        for num in remaining_nums:
            if num not in candidates:
                candidates.append(num)
            if len(candidates) >= 3:
                break
        
        return candidates[:3]

# =============== GIAO DIỆN TỐI ƯU MOBILE ===============
st.set_page_config(
    page_title="AI 3-TINH MOBILE",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS tối ưu cho mobile
st.markdown("""
    <style>
    /* Reset mặc định */
    .stApp {
        background: #0f172a;
        color: #f8fafc;
        padding: 10px;
    }
    
    /* Tiêu đề nhỏ gọn */
    .main-title {
        font-size: 1.8rem !important;
        text-align: center;
        color: #38bdf8;
        margin: 10px 0;
        font-weight: 700;
    }
    
    /* Ô input */
    .stTextArea textarea {
        font-size: 16px !important;
        min-height: 80px !important;
        background: #1e293b !important;
        color: #cbd5e1 !important;
        border: 2px solid #38bdf8 !important;
        border-radius: 10px !important;
    }
    
    /* Nút bấm */
    .stButton button {
        width: 100% !important;
        height: 50px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #38bdf8, #3b82f6) !important;
        border: none !important;
        border-radius: 12px !important;
        margin: 15px 0 !important;
    }
    
    /* Kết quả chính - NHỎ HƠN */
    .result-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        border: 2px solid #38bdf8;
    }
    
    /* Hiển thị số - KÍCH THƯỚC VỪA PHẢI */
    .numbers-display {
        font-size: 3.5rem !important;
        color: #fbbf24;
        font-weight: 900;
        letter-spacing: 8px;
        text-align: center;
        margin: 10px 0;
        text-shadow: 0 0 10px rgba(251, 191, 36, 0.5);
    }
    
    /* Box loại số */
    .eliminated-box {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        font-size: 14px;
    }
    
    /* Box số an toàn */
    .safe-box {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid #22c55e;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        font-size: 14px;
    }
    
    /* Thông tin phụ */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid #3b82f6;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        font-size: 13px;
    }
    
    /* Ẩn các element không cần thiết trên mobile */
    @media (max-width: 768px) {
        .st-emotion-cache-1v0mbdj {
            padding: 5px !important;
        }
        
        .numbers-display {
            font-size: 2.8rem !important;
            letter-spacing: 5px;
        }
        
        .main-title {
            font-size: 1.5rem !important;
        }
    }
    
    /* Scrollbar tối giản */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 3px;
    }
    </style>
""", unsafe_allow_html=True)

# Header ngắn gọn
st.markdown("<h1 class='main-title'>🎯 AI 3-TINH MOBILE</h1>", unsafe_allow_html=True)

# Khởi tạo analyzer
analyzer = LotteryAIAnalyzer()

# Input đơn giản
st.markdown("### 📥 Nhập dãy số")
data_input = st.text_area(
    "",
    height=100,
    placeholder="Dán dãy số từ kết quả...\nVí dụ: 53829174625381920475",
    help="Cần ít nhất 10 số để phân tích"
)

# Nút phân tích
if st.button("🔍 PHÂN TÍCH NGAY", type="primary"):
    if len(data_input.strip()) < 10:
        st.warning("⚠️ Vui lòng nhập ít nhất 10 số!")
    else:
        with st.spinner('Đang tính toán...'):
            time.sleep(0.5)
            
            # Phân tích
            eliminated, remaining = analyzer.eliminate_risk_numbers(data_input)
            top_three = analyzer.select_top_three(remaining, data_input)
            
            # Hiển thị kết quả chính - GỌN HƠN
            st.markdown(f"""
                <div class='result-card'>
                    <div style='text-align: center; color: #38bdf8; font-size: 16px;'>
                        🎯 3 SỐ TỐT NHẤT
                    </div>
                    <div class='numbers-display'>
                        {" ".join(top_three)}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Hiển thị thông tin phụ
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                    <div class='eliminated-box'>
                        <div style='color: #ef4444; font-size: 14px;'>
                            ⛔ LOẠI 3 SỐ
                        </div>
                        <div style='font-size: 20px; font-weight: bold; color: #fca5a5;'>
                            {" ".join(eliminated)}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class='safe-box'>
                        <div style='color: #22c55e; font-size: 14px;'>
                            ✅ DÀN 7 SỐ
                        </div>
                        <div style='font-size: 16px; color: #86efac;'>
                            {", ".join(remaining)}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Thông tin bổ sung
            st.markdown(f"""
                <div class='info-box'>
                    <div style='color: #3b82f6;'>
                        📊 THÔNG TIN PHÂN TÍCH
                    </div>
                    <div style='font-size: 13px; margin-top: 8px;'>
                        • Số cuối cùng: <b>{data_input[-1] if data_input else 'N/A'}</b><br>
                        • Tổng số đã phân tích: <b>{len(list(filter(str.isdigit, data_input)))}</b><br>
                        • Thời gian: <b>{datetime.now().strftime('%H:%M')}</b>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Hướng dẫn ngắn
            st.markdown("""
                <div style='background: rgba(251, 191, 36, 0.1); 
                          border-radius: 10px; 
                          padding: 12px; 
                          margin-top: 10px;
                          border: 1px solid #fbbf24;'>
                    <div style='color: #fbbf24; font-size: 14px;'>
                        💡 CHIẾN THUẬT
                    </div>
                    <div style='font-size: 12px; color: #fde68a;'>
                        1. Chọn đủ 7 số từ dàn bên trên<br>
                        2. Tập trung vào 3 số được bôi vàng<br>
                        3. Tránh 3 số bị loại<br>
                        4. Vào tiền hợp lý
                    </div>
                </div>
            """, unsafe_allow_html=True)

# Footer nhỏ
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 12px; color: #64748b;'>
    AI 3-TINH MOBILE | Phiên bản v1.1
</div>
""", unsafe_allow_html=True)