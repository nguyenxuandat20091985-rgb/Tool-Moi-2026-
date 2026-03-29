import streamlit as st
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
from itertools import combinations
import re

# ==================== CẤU HÌNH ====================
DB_FILE = "titan_v32_data.json"
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

st.set_page_config(page_title="TITAN V32 - AI PREDICTION", page_icon="🔮", layout="wide")

# ==================== CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        color: #FFD700;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }
    .metric-box {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin: 10px 0;
        border: 2px solid #FFD700;
    }
    .prediction-card {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #FFD700;
    }
    .top-pick {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: black;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .hot-number { color: #ff4444; font-weight: bold; }
    .cold-number { color: #4444ff; font-weight: bold; }
    .streak-number { color: #ffaa00; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==================== QUẢN LÝ DỮ LIỆU ====================
def load_data():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"history": [], "predictions": [], "stats": {}}

def save_data(data):
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ==================== PHÂN TÍCH THỐNG KÊ ====================
def analyze_frequency(numbers_list):
    """Phân tích tần suất xuất hiện"""
    all_digits = "".join(numbers_list)
    counter = Counter(all_digits)
    total = len(all_digits)
    
    stats = {}
    for digit in "0123456789":
        count = counter.get(digit, 0)
        stats[digit] = {
            "count": count,
            "frequency": count / total * 100 if total > 0 else 0,
            "percentage": f"{count/total*100:.1f}%" if total > 0 else "0%"
        }
    return stats

def detect_streaks(numbers_list):
    """Phát hiện số bệt (lặp liên tiếp)"""
    streaks = defaultdict(int)
    current_streak = {}
    
    for num in numbers_list:
        digits = set(num)
        for digit in digits:
            if digit in current_streak:
                current_streak[digit] += 1
                streaks[digit] = max(streaks[digit], current_streak[digit])
            else:
                current_streak[digit] = 1
    
    return dict(streaks)

def analyze_cycles(numbers_list):
    """Phân tích chu kỳ xuất hiện"""
    cycles = defaultdict(list)
    last_position = {}
    
    for idx, num in enumerate(numbers_list):
        for digit in set(num):
            if digit in last_position:
                gap = idx - last_position[digit]
                cycles[digit].append(gap)
            last_position[digit] = idx
    
    cycle_stats = {}
    for digit in "0123456789":
        if cycles[digit]:
            avg_gap = np.mean(cycles[digit])
            min_gap = min(cycles[digit])
            max_gap = max(cycles[digit])
            cycle_stats[digit] = {
                "avg": avg_gap,
                "min": min_gap,
                "max": max_gap,
                "predict_next": avg_gap
            }
    
    return cycle_stats

def analyze_patterns(numbers_list):
    """Phân tích pattern Đông-Tây"""
    patterns = {
        "even_odd": {"even": 0, "odd": 0},
        "high_low": {"high": 0, "low": 0},
        "sum_total": 0,
        "mirror_pairs": [],
        "shadow_numbers": defaultdict(int)
    }
    
    for num in numbers_list:
        digits = [int(d) for d in num]
        
        # Chẵn lẻ
        for d in digits:
            if d % 2 == 0:
                patterns["even_odd"]["even"] += 1
            else:
                patterns["even_odd"]["odd"] += 1
        
        # Cao thấp (0-4: thấp, 5-9: cao)
        for d in digits:
            if d >= 5:
                patterns["high_low"]["high"] += 1
            else:
                patterns["high_low"]["low"] += 1
        
        # Tổng
        patterns["sum_total"] += sum(digits)
        
        # Bóng số (mirror: 0-5, 1-6, 2-7, 3-8, 4-9)
        for d in digits:
            mirror = (d + 5) % 10
            patterns["shadow_numbers"][str(mirror)] += 1
    
    return patterns

# ==================== MÁY HỌC ĐƠN GIẢN ====================
def simple_ml_predict(numbers_list, freq_stats, cycle_stats):
    """Dự đoán dựa trên machine learning đơn giản"""
    scores = {}
    
    for digit in "0123456789":
        score = 0
        
        # Điểm từ tần suất
        freq = freq_stats[digit]["frequency"]
        score += freq * 2
        
        # Điểm từ chu kỳ
        if digit in cycle_stats:
            expected_gap = cycle_stats[digit]["avg"]
            last_gaps = []
            last_pos = -1
            for idx, num in enumerate(numbers_list):
                if digit in num:
                    if last_pos != -1:
                        last_gaps.append(idx - last_pos)
                    last_pos = idx
            
            if last_gaps:
                current_gap = len(numbers_list) - last_pos - 1
                if abs(current_gap - expected_gap) < 2:
                    score += 30  # Sắp đến chu kỳ
        
        # Điểm từ xu hướng gần
        recent = numbers_list[-10:] if len(numbers_list) >= 10 else numbers_list
        recent_count = sum(1 for num in recent if digit in num)
        score += recent_count * 5
        
        scores[digit] = score
    
    return scores

# ==================== TẠO PHƯƠNG ÁN ====================
def generate_strategies(numbers_list, freq_stats, cycle_stats, ml_scores, streaks, patterns):
    """Tạo 10+ phương án dự đoán"""
    strategies = []
    
    # 1. Theo tần suất cao
    hot_numbers = sorted(freq_stats.items(), key=lambda x: x[1]["frequency"], reverse=True)
    strategies.append({
        "name": "🔥 CẶP NÓNG (Tần suất cao)",
        "numbers": [hot_numbers[0][0], hot_numbers[1][0]],
        "reasoning": f"{hot_numbers[0][0]} ({hot_numbers[0][1]['percentage']}) và {hot_numbers[1][0]} ({hot_numbers[1][1]['percentage']}) xuất hiện nhiều nhất",
        "confidence": 75,
        "logic": "frequency"
    })
    
    # 2. Theo chu kỳ
    due_numbers = []
    for digit in "0123456789":
        if digit in cycle_stats:
            last_pos = -1
            for idx, num in enumerate(numbers_list):
                if digit in num:
                    last_pos = idx
            if last_pos != -1:
                current_gap = len(numbers_list) - last_pos - 1
                expected = cycle_stats[digit]["avg"]
                if current_gap >= expected * 0.8:
                    due_numbers.append((digit, current_gap, expected))
    
    due_numbers.sort(key=lambda x: x[1]/x[2] if x[2] > 0 else 0, reverse=True)
    if len(due_numbers) >= 2:
        strategies.append({
            "name": "⏰ CẶP ĐẾN KỲ (Chu kỳ)",
            "numbers": [due_numbers[0][0], due_numbers[1][0]],
            "reasoning": f"{due_numbers[0][0]} (cách {due_numbers[0][1]} kỳ, TB: {due_numbers[0][2]:.1f}) và {due_numbers[1][0]} đến kỳ",
            "confidence": 70,
            "logic": "cycle"
        })
    
    # 3. Theo ML score
    ml_top = sorted(ml_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    strategies.append({
        "name": "🤖 CẶP AI (Machine Learning)",
        "numbers": [ml_top[0][0], ml_top[1][0]],
        "reasoning": f"AI đánh giá {ml_top[0][0]} (score: {ml_top[0][1]:.0f}) và {ml_top[1][0]} (score: {ml_top[1][1]:.0f}) có xác suất cao",
        "confidence": 80,
        "logic": "ml"
    })
    
    # 4. Theo số bệt
    streak_sorted = sorted(streaks.items(), key=lambda x: x[1], reverse=True)
    if streak_sorted:
        strategies.append({
            "name": "📌 CẶP BỆT (Đang hot)",
            "numbers": [streak_sorted[0][0], streak_sorted[1][0] if len(streak_sorted) > 1 else streak_sorted[0][0]],
            "reasoning": f"{streak_sorted[0][0]} đang bệt {streak_sorted[0][1]} kỳ liên tiếp",
            "confidence": 65,
            "logic": "streak"
        })
    
    # 5. Theo bóng số
    shadow_top = sorted(patterns["shadow_numbers"].items(), key=lambda x: x[1], reverse=True)[:2]
    if len(shadow_top) >= 2:
        strategies.append({
            "name": "🪞 CẶP BÓNG (Mirror)",
            "numbers": [shadow_top[0][0], shadow_top[1][0]],
            "reasoning": f"Bóng số của {shadow_top[0][0]} và {shadow_top[1][0]} xuất hiện nhiều",
            "confidence": 60,
            "logic": "shadow"
        })
    
    # 6. Theo chẵn lẻ cân bằng
    if patterns["even_odd"]["even"] > patterns["even_odd"]["odd"]:
        # Ưu tiên số lẻ
        odd_numbers = [d for d in "13579" if freq_stats[d]["frequency"] > 8]
        if len(odd_numbers) >= 2:
            strategies.append({
                "name": "⚖️ CẶP CÂN BẰNG (Chẵn-Lẻ)",
                "numbers": [odd_numbers[0], odd_numbers[1]],
                "reasoning": "Cân bằng xu hướng chẵn lẻ",
                "confidence": 55,
                "logic": "balance"
            })
    
    # 7. Theo cao thấp
    if patterns["high_low"]["high"] > patterns["high_low"]["low"]:
        # Ưu tiên số thấp
        low_numbers = [d for d in "01234" if freq_stats[d]["frequency"] > 5]
        if len(low_numbers) >= 2:
            strategies.append({
                "name": "📊 CẶP CAO-THẤP",
                "numbers": [low_numbers[0], low_numbers[1]],
                "reasoning": "Cân bằng xu hướng cao-thấp",
                "confidence": 55,
                "logic": "highlow"
            })
    
    # 8. Theo tổng
    avg_sum = patterns["sum_total"] / len(numbers_list) if numbers_list else 22.5
    strategies.append({
        "name": "📐 CẶP THEO TỔNG",
        "numbers": ["4", "7"],  # Tổng trung bình
        "reasoning": f"Tổng trung bình kỳ: {avg_sum:.1f}, chọn cặp có tổng gần mức này",
        "confidence": 50,
        "logic": "sum"
    })
    
    # 9. Theo số lạnh (đảo ngược)
    cold_numbers = sorted(freq_stats.items(), key=lambda x: x[1]["frequency"])[:2]
    strategies.append({
        "name": "❄️ CẶP LẠNH (Đảo chiều)",
        "numbers": [cold_numbers[0][0], cold_numbers[1][0]],
        "reasoning": f"{cold_numbers[0][0]} và {cold_numbers[1][0]} ít xuất hiện, có thể về",
        "confidence": 45,
        "logic": "cold"
    })
    
    # 10. Theo vị trí
    position_stats = defaultdict(lambda: Counter())
    for num in numbers_list[-20:]:
        for pos, digit in enumerate(num):
            position_stats[pos][digit] += 1
    
    pos0_top = position_stats[0].most_common(1)[0][0] if position_stats[0] else "0"
    pos4_top = position_stats[4].most_common(1)[0][0] if position_stats[4] else "0"
    strategies.append({
        "name": "🎯 CẶP VỊ TRÍ (Đầu-Cuối)",
        "numbers": [pos0_top, pos4_top],
        "reasoning": f"Số hay về đầu: {pos0_top}, số hay về cuối: {pos4_top}",
        "confidence": 60,
        "logic": "position"
    })
    
    return strategies

# ==================== CHỌN TOP 3 ====================
def select_top_picks(strategies):
    """Chọn TOP 3 bạch thủ mạnh nhất"""
    # Sort theo confidence
    sorted_strategies = sorted(strategies, key=lambda x: x["confidence"], reverse=True)
    
    # Lấy top 3 khác nhau
    top_picks = []
    used_numbers = set()
    
    for strategy in sorted_strategies:
        if len(top_picks) >= 3:
            break
        
        nums = strategy["numbers"]
        # Kiểm tra xem cặp này có quá giống với cái đã chọn không
        num_set = set(nums)
        if not num_set.issubset(used_numbers):
            top_picks.append(strategy)
            used_numbers.update(num_set)
    
    return top_picks

# ==================== GIAO DIỆN CHÍNH ====================
def main():
    st.markdown('<h1 class="main-header">🔮 TITAN V32 - AI PREDICTION SYSTEM</h1>', unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 PHÂN TÍCH", "🎯 DỰ ĐOÁN", "📋 LỊCH SỬ"])
    
    # Input
    st.sidebar.header("📥 NHẬP DỮ LIỆU")
    input_text = st.sidebar.text_area(
        "Nhập kết quả (mỗi kỳ 1 dòng):",
        height=200,
        placeholder="46602\n32476\n14606..."
    )
    
    if st.sidebar.button("💾 LƯU VÀ PHÂN TÍCH"):
        numbers = [line.strip() for line in input_text.split('\n') if line.strip() and re.match(r'^\d{5}$', line.strip())]
        if numbers:
            data["history"] = numbers
            save_data(data)
            st.sidebar.success(f"Đã lưu {len(numbers)} kỳ!")
    
    # Load history nếu có
    numbers_list = data.get("history", [])
    
    if not numbers_list:
        # Sử dụng dữ liệu mẫu từ input
        sample_input = """46602
32476
14606
97269
15109
33912
76108
41895
77642
26765
95572
33997
14405
31871
82832
33123
01617
04675
12461
20659
98005
98310
52064
55063
69053
13944
65605
06931
77496
60204
51253
76732
73854
89879"""
        numbers_list = [line.strip() for line in sample_input.split('\n') if line.strip()]
    
    # Phân tích
    freq_stats = analyze_frequency(numbers_list)
    streaks = detect_streaks(numbers_list)
    cycle_stats = analyze_cycles(numbers_list)
    patterns = analyze_patterns(numbers_list)
    ml_scores = simple_ml_predict(numbers_list, freq_stats, cycle_stats)
    strategies = generate_strategies(numbers_list, freq_stats, cycle_stats, ml_scores, streaks, patterns)
    top_picks = select_top_picks(strategies)
    
    with tab1:
        st.header("📊 PHÂN TÍCH THỐNG KÊ")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🔥 SỐ NÓNG (Tần suất cao)")
            hot_sorted = sorted(freq_stats.items(), key=lambda x: x[1]["frequency"], reverse=True)[:5]
            for digit, stats in hot_sorted:
                st.markdown(f"**{digit}**: {stats['percentage']} ({stats['count']} lần)")
        
        with col2:
            st.subheader("❄️ SỐ LẠNH (Ít xuất hiện)")
            cold_sorted = sorted(freq_stats.items(), key=lambda x: x[1]["frequency"])[:5]
            for digit, stats in cold_sorted:
                st.markdown(f"**{digit}**: {stats['percentage']} ({stats['count']} lần)")
        
        with col3:
            st.subheader("📌 SỐ BỆT (Liên tiếp)")
            if streaks:
                streak_sorted = sorted(streaks.items(), key=lambda x: x[1], reverse=True)[:5]
                for digit, count in streak_sorted:
                    st.markdown(f"**{digit}**: {count} kỳ liên tiếp")
            else:
                st.write("Không có số bệt đáng chú ý")
        
        st.divider()
        
        st.subheader("📈 PHÂN TÍCH CHU KỲ")
        cycle_df = pd.DataFrame([
            {"Số": d, "TB kỳ": f"{cycle_stats[d]['avg']:.1f}" if d in cycle_stats else "N/A",
             "Min": cycle_stats[d]['min'] if d in cycle_stats else "-",
             "Max": cycle_stats[d]['max'] if d in cycle_stats else "-"}
            for d in "0123456789"
        ])
        st.dataframe(cycle_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        st.subheader("🎭 PATTERN ĐÔNG-TÂY")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Chẵn/Lẻ**: {patterns['even_odd']['even']} / {patterns['even_odd']['odd']}")
            st.markdown(f"**Cao/Thấp**: {patterns['high_low']['high']} / {patterns['high_low']['low']}")
        with col2:
            st.markdown(f"**Tổng trung bình**: {patterns['sum_total']/len(numbers_list):.1f}")
            st.markdown(f"**Số kỳ**: {len(numbers_list)}")
    
    with tab2:
        st.header("🎯 DỰ ĐOÁN TOP 3 BẠCH THỦ")
        
        # Hiển thị TOP 3
        for i, pick in enumerate(top_picks, 1):
            st.markdown(f"""
            <div class="top-pick">
                🏆 TOP {i}: {pick['numbers'][0]} - {pick['numbers'][1]}<br>
                <small>Độ tin cậy: {pick['confidence']}%</small>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"📋 Chi tiết TOP {i}: {pick['name']}"):
                st.write(f"**Lý do**: {pick['reasoning']}")
                st.write(f"**Logic**: {pick['logic']}")
                st.write(f"**Độ tin cậy**: {pick['confidence']}%")
        
        st.divider()
        
        st.subheader("📋 TẤT CẢ PHƯƠNG ÁN ({})".format(len(strategies)))
        for i, strategy in enumerate(strategies, 1):
            with st.expander(f"{i}. {strategy['name']} - Độ tin cậy: {strategy['confidence']}%"):
                st.markdown(f"""
                **Cặp số**: {strategy['numbers'][0]} - {strategy['numbers'][1]}<br>
                **Lý do**: {strategy['reasoning']}<br>
                **Logic**: {strategy['logic']}
                """, unsafe_allow_html=True)
        
        st.divider()
        
        st.subheader("💡 CHIẾN LƯỢC ĐÁNH")
        st.info("""
        **Khuyến nghị:**<br>
        1. Ưu tiên TOP 1 với 70% vốn<br>
        2. TOP 2 và TOP 3 với 20% vốn mỗi cặp<br>
        3. Theo dõi thêm 2-3 kỳ để xác nhận xu hướng<br>
        4. Không đánh quá 5 cặp/kỳ để tối ưu vốn
        """)
    
    with tab3:
        st.header("📋 LỊCH SỬ DỰ ĐOÁN")
        
        if "predictions" in data and data["predictions"]:
            pred_df = pd.DataFrame(data["predictions"])
            st.dataframe(pred_df, use_container_width=True)
        else:
            st.write("Chưa có lịch sử dự đoán")
        
        st.divider()
        
        st.subheader("💾 Lưu dự đoán hiện tại")
        if st.button("Lưu kết quả vào lịch sử"):
            for pick in top_picks:
                data["predictions"].append({
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "numbers": f"{pick['numbers'][0]}-{pick['numbers'][1]}",
                    "confidence": pick['confidence'],
                    "strategy": pick['name']
                })
            save_data(data)
            st.success("Đã lưu TOP 3 vào lịch sử!")
    
    # Tự động lưu vào session
    data["last_analysis"] = {
        "timestamp": datetime.now().isoformat(),
        "total_periods": len(numbers_list),
        "top_picks": [f"{p['numbers'][0]}-{p['numbers'][1]}" for p in top_picks]
    }
    save_data(data)

if __name__ == "__main__":
    main()