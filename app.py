import streamlit as st
import re, pandas as pd, numpy as np, math
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import random

# === CẤU HÌNH ===
LUCKY_OX = [0, 2, 5, 6, 7, 8]
st.set_page_config(page_title="TITAN V52 - OMNI PREDICTOR", page_icon="🎯", layout="centered")

# === CSS GOD MODE ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #000000 0%, #0a0a1a 50%, #000000 100%);
        color: #FFD700;
        font-family: 'Orbitron', monospace;
    }
    
    .god-header {
        font-size: 38px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #FFD700, #FF00FF, #00FFFF, #FFD700);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite;
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .method-card {
        background: linear-gradient(135deg, #1a1a2e, #0f0f1a);
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
        border-left: 4px solid;
    }
    
    .method-hot { border-color: #ff0040; }
    .method-cold { border-color: #0066ff; }
    .method-gan { border-color: #ff9900; }
    .method-shadow { border-color: #9900ff; }
    .method-complement { border-color: #00ff99; }
    .method-pattern { border-color: #ffcc00; }
    .method-frequency { border-color: #ff6600; }
    .method-entropy { border-color: #cc00ff; }
    .method-position { border-color: #00ccff; }
    .method-advanced { border-color: #ff00cc; }
    
    .pair-display {
        font-size: 42px;
        font-weight: 900;
        letter-spacing: 10px;
        text-align: center;
        padding: 15px;
        margin: 10px 0;
        border-radius: 12px;
        background: linear-gradient(135deg, #1a1a2e, #0f0f1a);
    }
    
    .final-pick {
        font-size: 64px;
        font-weight: 900;
        letter-spacing: 15px;
        text-align: center;
        padding: 30px;
        margin: 20px 0;
        border-radius: 20px;
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #000;
        box-shadow: 0 0 40px rgba(255, 215, 0, 0.6);
        animation: pulse-gold 2s infinite;
    }
    
    @keyframes pulse-gold {
        0%, 100% { transform: scale(1); box-shadow: 0 0 40px rgba(255, 215, 0, 0.6); }
        50% { transform: scale(1.02); box-shadow: 0 0 60px rgba(255, 215, 0, 0.8); }
    }
    
    .top3-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 3px solid;
        text-align: center;
    }
    
    .top3-gold { border-color: #FFD700; box-shadow: 0 0 20px rgba(255, 215, 0, 0.4); }
    .top3-silver { border-color: #C0C0C0; box-shadow: 0 0 20px rgba(192, 192, 192, 0.4); }
    .top3-bronze { border-color: #CD7F32; box-shadow: 0 0 20px rgba(205, 127, 50, 0.4); }
    
    .tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 10px;
        font-weight: bold;
        margin: 2px;
    }
    
    .tag-strong { background: #00ff40; color: #000; }
    .tag-medium { background: #ffff00; color: #000; }
    .tag-weak { background: #ff0040; color: #fff; }
    .tag-method { background: #9900ff; color: #fff; }
    
    .metric-box {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin: 5px 0;
        border: 1px solid #333;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #3a1a1a, #2a0a0a);
        border: 2px solid #ff0040;
        color: #ff6680;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #1a3a1a, #0a2a0a);
        border: 2px solid #00ff40;
        color: #66ff80;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .history-win { color: #00ff40; font-weight: 900; }
    .history-lose { color: #ff0040; font-weight: 900; }
    
    .progress-bar {
        height: 20px;
        background: #1a1a1a;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00);
        transition: width 0.5s;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: bold;
        color: #000;
    }
    
    .explanation-box {
        background: #0f0f1a;
        border-left: 4px solid #00ffff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
        font-size: 12px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# === OMNI PREDICTOR ENGINE ===

class OmniPredictor:
    """Hệ thống dự đoán đa phương pháp - Top 1%"""
    
    def __init__(self, db, history=None):
        self.db = db
        self.history = history or []
        self.all_pairs = []
        self.methods_results = {}
        
    def method_1_frequency(self):
        """Phương pháp 1: Tần suất thuần túy"""
        results = []
        recent_str = "".join(self.db[-50:])
        single_pool = Counter(recent_str)
        
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            freq = single_pool[p[0]] + single_pool[p[1]]
            results.append({'pair': pair, 'score': freq * 5, 'method': 'FREQUENCY'})
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:20]
    
    def method_2_gan(self):
        """Phương pháp 2: Số gan (vàng 5-12 kỳ)"""
        results = []
        
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            gan = 0
            for num in reversed(self.db):
                if not set(p).issubset(set(num)):
                    gan += 1
                else:
                    break
            
            # Điểm cao nhất cho gan 5-12
            if 5 <= gan <= 12:
                score = 100 - abs(gan - 8.5) * 5  # Đỉnh ở gan 8-9
            elif 2 <= gan <= 4:
                score = 50
            elif gan > 15:
                score = 10
            else:
                score = 30
            
            results.append({'pair': pair, 'score': score, 'method': 'GAN', 'gan': gan})
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:20]
    
    def method_3_streak(self):
        """Phương pháp 3: Bệt (nhưng tránh bệt quá dài)"""
        results = []
        
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            streak = 0
            for num in reversed(self.db):
                if set(p).issubset(set(num)):
                    streak += 1
                else:
                    break
            
            # Ưu tiên streak 1-2, tránh >= 3
            if streak == 1:
                score = 80
            elif streak == 2:
                score = 60
            elif streak >= 3:
                score = 10  # Bẫy
            else:
                score = 40
            
            results.append({'pair': pair, 'score': score, 'method': 'STREAK', 'streak': streak})
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:20]
    
    def method_4_shadow(self):
        """Phương pháp 4: Shadow number (số đi sau)"""
        if len(self.db) < 10:
            return []
        
        shadow_map = defaultdict(Counter)
        for idx in range(len(self.db) - 1):
            current = self.db[idx]
            next_num = self.db[idx + 1]
            for d in current:
                for nd in next_num:
                    shadow_map[d][nd] += 1
        
        results = []
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            score = 0
            # Check nếu p[1] là shadow của p[0] hoặc ngược lại
            if p[1] in shadow_map[p[0]]:
                score += shadow_map[p[0]][p[1]] * 10
            if p[0] in shadow_map[p[1]]:
                score += shadow_map[p[1]][p[0]] * 10
            
            results.append({'pair': pair, 'score': score, 'method': 'SHADOW'})
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:20]
    
    def method_5_complement(self):
        """Phương pháp 5: Số bù (9-x)"""
        results = []
        
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            # Check nếu là cặp bù
            is_complement = (int(p[0]) + int(p[1])) == 9
            
            if is_complement:
                score = 70
            else:
                # Check tần suất cặp bù xuất hiện cùng
                complement_count = 0
                for num in self.db[-30:]:
                    has_p0 = p[0] in num
                    has_comp_p1 = str(9 - int(p[1])) in num
                    if has_p0 and has_comp_p1:
                        complement_count += 1
                
                score = complement_count * 15
            
            results.append({'pair': pair, 'score': score, 'method': 'COMPLEMENT'})
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:20]
    
    def method_6_pattern(self):
        """Phương pháp 6: Pattern lặp"""
        results = []
        
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            score = 0
            
            # Tìm pattern lặp
            for i in range(len(self.db) - 5):
                if set(p).issubset(set(self.db[i])):
                    # Check nếu xuất hiện lại sau 3-7 kỳ
                    for j in range(i+3, min(i+8, len(self.db))):
                        if set(p).issubset(set(self.db[j])):
                            score += 20
            
            results.append({'pair': pair, 'score': score, 'method': 'PATTERN'})
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:20]
    
    def method_7_position(self):
        """Phương pháp 7: Phân tích vị trí"""
        positions = {i: Counter() for i in range(5)}
        for num in self.db[-30:]:
            for i, d in enumerate(num):
                positions[i][d] += 1
        
        results = []
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            score = 0
            
            # Check nếu 2 số này hot ở các vị trí khác nhau
            for pos in positions:
                if p[0] in [x[0] for x in positions[pos].most_common(3)]:
                    score += 10
                if p[1] in [x[0] for x in positions[pos].most_common(3)]:
                    score += 10
            
            results.append({'pair': pair, 'score': score, 'method': 'POSITION'})
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:20]
    
    def method_8_entropy(self):
        """Phương pháp 8: Entropy cân bằng"""
        recent_str = "".join(self.db[-20:])
        counter = Counter(recent_str)
        total = len(recent_str)
        
        # Tính entropy từng số
        entropy_scores = {}
        for d in "0123456789":
            count = counter.get(d, 0)
            if count > 0:
                p = count / total
                entropy_scores[d] = -p * math.log2(p) if p > 0 else 0
            else:
                entropy_scores[d] = 0.5  # Số lạnh có tiềm năng
        
        results = []
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            score = (entropy_scores.get(p[0], 0) + entropy_scores.get(p[1], 0)) * 50
            results.append({'pair': pair, 'score': score, 'method': 'ENTROPY'})
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:20]
    
    def method_9_lucky(self):
        """Phương pháp 9: Tuổi Sửu + Ngũ hành"""
        results = []
        
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            score = 0
            
            # Tuổi Sửu
            lucky_count = sum(1 for d in p if int(d) in LUCKY_OX)
            score += lucky_count * 20
            
            # Chẵn lẻ cân bằng
            even_count = sum(1 for d in p if int(d) % 2 == 0)
            if even_count == 1:  # 1 chẵn 1 lẻ
                score += 30
            
            # Cao thấp cân bằng
            high_count = sum(1 for d in p if int(d) >= 5)
            if high_count == 1:  # 1 cao 1 thấp
                score += 25
            
            results.append({'pair': pair, 'score': score, 'method': 'LUCKY'})
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:20]
    
    def method_10_advanced(self):
        """Phương pháp 10: AI tổng hợp nâng cao"""
        results = []
        
        # Kết hợp nhiều yếu tố
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            score = 0
            
            # 1. Tần suất 30 kỳ
            freq = sum(1 for n in self.db[-30:] if set(p).issubset(set(n)))
            if 3 <= freq <= 7:
                score += 40
            
            # 2. Gan zone
            gan = 0
            for num in reversed(self.db):
                if not set(p).issubset(set(num)):
                    gan += 1
                else:
                    break
            
            if 5 <= gan <= 12:
                score += 50
            
            # 3. Tránh bệt >= 3
            streak = 0
            for num in reversed(self.db):
                if set(p).issubset(set(num)):
                    streak += 1
                else:
                    break
            
            if streak >= 3:
                score -= 60
            elif streak == 1:
                score += 30
            
            # 4. History learning
            if self.history:
                pair_losses = sum(1 for h in self.history if h.get('Dự đoán') == pair and '❌' in h.get('KQ', ''))
                if pair_losses >= 2:
                    score -= 40
            
            results.append({'pair': pair, 'score': score, 'method': 'ADVANCED'})
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:20]
    
    def generate_all_candidates(self):
        """Tạo tất cả ứng viên từ 10 phương pháp"""
        self.methods_results = {
            'FREQUENCY': self.method_1_frequency(),
            'GAN': self.method_2_gan(),
            'STREAK': self.method_3_streak(),
            'SHADOW': self.method_4_shadow(),
            'COMPLEMENT': self.method_5_complement(),
            'PATTERN': self.method_6_pattern(),
            'POSITION': self.method_7_position(),
            'ENTROPY': self.method_8_entropy(),
            'LUCKY': self.method_9_lucky(),
            'ADVANCED': self.method_10_advanced()
        }
        
        # Gom tất cả vào
        all_candidates = defaultdict(lambda: {'score': 0, 'methods': [], 'details': {}})
        
        for method_name, results in self.methods_results.items():
            for i, item in enumerate(results):
                pair = item['pair']
                score = item['score']
                
                # Normalize score theo rank
                rank_score = max(0, 20 - i) * (score / 100)
                
                all_candidates[pair]['score'] += rank_score
                all_candidates[pair]['methods'].append(method_name)
                all_candidates[pair]['details'][method_name] = score
        
        # Convert to list
        self.all_pairs = [
            {'pair': pair, 'total_score': data['score'], 'methods': data['methods'], 'details': data['details']}
            for pair, data in all_candidates.items()
        ]
        
        self.all_pairs.sort(key=lambda x: x['total_score'], reverse=True)
    
    def select_top_3(self):
        """Chọn top 3 từ danh sách đã tổng hợp"""
        if not self.all_pairs:
            self.generate_all_candidates()
        
        top_3 = self.all_pairs[:3]
        
        # Gắn nhãn
        labels = ['🥇 GOLD', '🥈 SILVER', '🥉 BRONZE']
        for i, pair in enumerate(top_3):
            pair['label'] = labels[i] if i < len(labels) else f'TOP {i+1}'
        
        return top_3
    
    def create_final_pick(self, top_3):
        """Tạo 1 cặp FINAL tối ưu nhất"""
        if not top_3:
            return None
        
        # Phân tích sâu top 3
        final_analysis = []
        
        for pair_data in top_3:
            pair = pair_data['pair']
            score = pair_data['total_score']
            methods = pair_data['methods']
            
            # Tính thêm các yếu tố bổ sung
            gan = 0
            for num in reversed(self.db):
                if not set(pair).issubset(set(num)):
                    gan += 1
                else:
                    break
            
            streak = 0
            for num in reversed(self.db):
                if set(pair).issubset(set(num)):
                    streak += 1
                else:
                    break
            
            freq = sum(1 for n in self.db[-30:] if set(pair).issubset(set(n)))
            
            # Final score với trọng số tối ưu
            final_score = score
            
            # Bonus cho gan vàng
            if 5 <= gan <= 12:
                final_score *= 1.3
            
            # Penalty cho bệt dài
            if streak >= 3:
                final_score *= 0.6
            
            # Bonus cho nhiều phương pháp đồng thuận
            if len(methods) >= 7:
                final_score *= 1.2
            
            final_analysis.append({
                'pair': pair,
                'final_score': final_score,
                'original_score': score,
                'methods_count': len(methods),
                'gan': gan,
                'streak': streak,
                'frequency': freq,
                'methods': methods
            })
        
        # Sort theo final score
        final_analysis.sort(key=lambda x: x['final_score'], reverse=True)
        
        return final_analysis[0]
    
    def get_explanation(self, final_pick):
        """Giải thích chi tiết vì sao chọn"""
        if not final_pick:
            return ""
        
        explanations = []
        
        # 1. Phương pháp đồng thuận
        methods_str = ", ".join(final_pick['methods'][:5])
        explanations.append(f"✅ **Đa phương pháp đồng thuận:** {final_pick['methods_count']}/10 phương pháp chọn cặp này ({methods_str})")
        
        # 2. Nhịp gan
        if 5 <= final_pick['gan'] <= 12:
            explanations.append(f"✅ **Nhịp gan vàng:** {final_pick['gan']} kỳ (vùng tối ưu 5-12)")
        else:
            explanations.append(f"⚠️ **Nhịp gan:** {final_pick['gan']} kỳ")
        
        # 3. Tần suất
        if 3 <= final_pick['frequency'] <= 7:
            explanations.append(f"✅ **Tần suất lý tưởng:** {final_pick['frequency']} lần/30 kỳ")
        else:
            explanations.append(f"📊 **Tần suất:** {final_pick['frequency']} lần/30 kỳ")
        
        # 4. Bệt
        if final_pick['streak'] == 0:
            explanations.append(f"✅ **Không bệt:** An toàn, không phải bẫy")
        elif final_pick['streak'] == 1:
            explanations.append(f"✅ **Bệt 1 kỳ:** Có thể rơi lại")
        elif final_pick['streak'] >= 3:
            explanations.append(f"❌ **Bệt {final_pick['streak']} kỳ:** Rủi ro cao, có thể là bẫy")
        
        # 5. Score
        explanations.append(f" **Điểm tổng hợp:** {final_pick['final_score']:.1f} (cao nhất)")
        
        return "\n\n".join(explanations)
    
    def get_risk_warning(self, final_pick):
        """Cảnh báo rủi ro"""
        warnings = []
        
        if final_pick['streak'] >= 3:
            warnings.append("⚠️ **RỦI RO CAO:** Đang bệt dài, có thể là bẫy nhà cái")
        
        if final_pick['gan'] > 15:
            warnings.append("⚠️ **GAN QUÁ SÂU:** Số này lâu chưa về, rủi ro cao")
        
        if final_pick['methods_count'] < 5:
            warnings.append("⚠️ **ÍT ĐỒNG THUẬN:** Chỉ {final_pick['methods_count']} phương pháp chọn, độ tin cậy thấp")
        
        # Check history
        if self.history:
            recent_losses = sum(1 for h in self.history[:5] if '❌' in h.get('KQ', ''))
            if recent_losses >= 4:
                warnings.append("⚠️ **THUA NHIỀU:** 4/5 kỳ gần thua, nên nghỉ hoặc đánh nhỏ")
        
        if not warnings:
            warnings.append("✅ **RỦI RO THẤP:** Các chỉ số đều trong vùng an toàn")
        
        return "\n\n".join(warnings)
    
    def predict(self):
        """Dự đoán cuối cùng"""
        if len(self.db) < 15:
            return None
        
        self.generate_all_candidates()
        top_3 = self.select_top_3()
        final_pick = self.create_final_pick(top_3)
        
        return {
            'all_pairs': self.all_pairs[:10],  # Top 10
            'top_3': top_3,
            'final': final_pick,
            'explanation': self.get_explanation(final_pick) if final_pick else "",
            'risk_warning': self.get_risk_warning(final_pick) if final_pick else "",
            'methods_results': self.methods_results
        }

# === XỬ LÝ DỮ LIỆU ===

def get_nums(text):
    clean_text = re.sub(r'\s+', '', text)
    nums = [n for n in re.findall(r"\d{5}", clean_text) if n]
    seen = set()
    return [n for n in nums if not (n in seen or seen.add(n))]

# === GIAO DIỆN ===

st.markdown('<h1 class="god-header">🎯 TITAN V52 - OMNI PREDICTOR</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888; font-size:11px;">10 Phương Pháp | Top 3 Selection | Final Optimization | God Mode</p>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_area("📥 Kết quả (kỳ mới nhất ở dưới):", height=100, 
                          placeholder="84890\n07119\n33627")

col1, col2 = st.columns(2)
with col1:
    if st.button("🎯 KÍCH HOẠT OMNI", type="primary", use_container_width=True):
        nums = get_nums(user_input)
        if len(nums) >= 15:
            if "last_pred" in st.session_state and nums:
                lp = st.session_state.last_pred
                last = nums[-1]
                if lp and lp.get('final'):
                    best = lp['final']['pair']
                    win = all(d in last for d in best)
                    st.session_state.history.insert(0, {
                        'Kỳ': last,
                        'Dự đoán': best,
                        'KQ': '🔥' if win else '❌'
                    })
            
            predictor = OmniPredictor(nums, st.session_state.history)
            st.session_state.last_pred = predictor.predict()
            st.rerun()
        else:
            st.warning(f"Cần 15+ kỳ (có {len(nums)})")

with col2:
    if st.button("🗑️ Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# === HIỂN THỊ KẾT QUẢ ===

if "last_pred" in st.session_state:
    res = st.session_state.last_pred
    
    # === FINAL PICK ===
    st.markdown("""<div style="text-align:center; margin:20px 0; font-size:18px;"> BẠCH THỦ FINAL</div>""", unsafe_allow_html=True)
    
    if res['final']:
        st.markdown(f"""
        <div class="final-pick">
            {res['final']['pair'][0]} - {res['final']['pair'][1]}
        </div>
        """, unsafe_allow_html=True)
        
        # Giải thích
        st.markdown("""<div style="text-align:center; margin:20px 0; font-size:16px;">📖 VÌ SAO CHỌN?</div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        {res['explanation'].replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
        
        # Cảnh báo rủi ro
        st.markdown("""<div style="text-align:center; margin:20px 0; font-size:16px;">⚠️ RỦI RO</div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="warning-box">
        {res['risk_warning'].replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Không thể tạo dự đoán. Kiểm tra lại dữ liệu.")
    
    # === TOP 3 ===
    st.markdown("""<div style="text-align:center; margin:25px 0; font-size:18px;">🥇 TOP 3 MẠNH NHẤT</div>""", unsafe_allow_html=True)
    
    if res['top_3']:
        cols = st.columns(3)
        for i, pair_data in enumerate(res['top_3']):
            with cols[i]:
                card_class = "top3-gold" if i == 0 else ("top3-silver" if i == 1 else "top3-bronze")
                st.markdown(f"""
                <div class="top3-card {card_class}">
                    <div style="font-size:14px; margin-bottom:10px;">{pair_data['label']}</div>
                    <div style="font-size:36px; font-weight:900; color:#FFD700; letter-spacing:6px;">
                        {pair_data['pair'][0]} - {pair_data['pair'][1]}
                    </div>
                    <div style="font-size:12px; margin-top:10px; color:#888;">
                        Score: {pair_data['total_score']:.1f}<br>
                        Methods: {len(pair_data['methods'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # === TOP 10 ===
    st.markdown("""<div style="text-align:center; margin:25px 0; font-size:18px;">📊 TOP 10 ỨNG VIÊN</div>""", unsafe_allow_html=True)
    
    for i, pair_data in enumerate(res['all_pairs'][:10]):
        methods_str = ", ".join(pair_data['methods'][:3])
        
        st.markdown(f"""
        <div class="method-card method-advanced">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:12px; color:#888;">#{i+1}</span>
                <span style="font-size:24px; font-weight:900; color:#FFD700; letter-spacing:4px;">
                    {pair_data['pair'][0]} - {pair_data['pair'][1]}
                </span>
                <span style="font-size:16px; color:#00ff40;">{pair_data['total_score']:.1f}</span>
            </div>
            <div style="font-size:10px; color:#888; margin-top:5px;">
                {methods_str}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # === HISTORY ===
    if st.session_state.history:
        st.markdown("""<div style="text-align:center; margin:25px 0; font-size:18px;">📋 LỊCH SỬ</div>""", unsafe_allow_html=True)
        
        df = pd.DataFrame(st.session_state.history[:15])
        
        def color_kq(val):
            return 'color: #00ff40; font-weight: 900' if '🔥' in val else 'color: #ff0040; font-weight: 900'
        
        st.dataframe(df.style.applymap(color_kq, subset=['KQ']), 
                     use_container_width=True, hide_index=True)
        
        # Win rate
        wins = sum(1 for h in st.session_state.history if '🔥' in h['KQ'])
        total = len(st.session_state.history)
        rate = (wins/total*100) if total > 0 else 0
        
        color_rate = "#00ff40" if rate >= 40 else ("#ffff00" if rate >= 30 else "#ff0040")
        
        st.markdown(f"""
        <div class="metric-box" style="border: 2px solid {color_rate}; margin-top:15px;">
            <div style="font-size:14px;">TỶ LỆ THẮNG</div>
            <div style="font-size:32px; font-weight:900; color:{color_rate};">{rate:.1f}% ({wins}/{total})</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width:{rate}%; background:{color_rate};">{rate:.0f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# === FOOTER ===
st.markdown("""
<div style="text-align:center; color:#444; font-size:10px; margin-top:30px; padding-top:15px; border-top:1px solid #333;">
    🎯 TITAN V52 - OMNI PREDICTOR | 10 Phương Pháp | Top 3 Selection | Final Optimization<br>
    <i>Tool hỗ trợ phân tích - Không đảm bảo 100% - Quản lý vốn thông minh</i>
</div>
""", unsafe_allow_html=True)