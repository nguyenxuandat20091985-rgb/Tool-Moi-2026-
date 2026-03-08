# ==============================================================================
# TITAN AI v5.0 - HOUSE PATTERN DETECTOR
# Chuyên phát hiện: Đảo cầu | Bệt cầu | Xoay cầu | Bẫy nhà cái
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import random
import re
import math
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CSS STYLING
# ==============================================================================

st.set_page_config(page_title="🎯 TITAN AI v5.0 | House Pattern", page_icon="🔍", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%); color: #e6edf3; }
    #MainMenu, footer, header { visibility: hidden; }
    
    .header-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%);
        border-radius: 20px; padding: 30px; text-align: center; margin-bottom: 25px;
        border: 2px solid rgba(124,58,237,0.3);
    }
    .header-title { font-size: 32px; font-weight: 900; color: white; margin: 0; }
    .header-subtitle { font-size: 14px; color: rgba(255,255,255,0.8); margin-top: 8px; }
    
    .pattern-alert {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white; padding: 15px 25px; border-radius: 12px;
        text-align: center; font-weight: 700; margin: 20px 0;
        border: 2px solid #fca5a5; animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 10px rgba(220,38,38,0.5); }
        50% { box-shadow: 0 0 20px rgba(220,38,38,0.8); }
    }
    
    .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }
    .stat-box {
        background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px;
        text-align: center; border: 1px solid rgba(255,255,255,0.1);
    }
    .stat-value { font-size: 32px; font-weight: 800; color: #60a5fa; }
    .stat-label { font-size: 12px; color: #94a3b8; margin-top: 8px; text-transform: uppercase; }
    
    .status-card {
        padding: 15px 25px; border-radius: 12px; text-align: center;
        font-weight: 700; font-size: 15px; margin: 20px 0;
    }
    .status-ok { background: linear-gradient(135deg, #059669, #10b981); color: white; border: 2px solid #34d399; }
    .status-warn { background: linear-gradient(135deg, #d97706, #f59e0b); color: white; border: 2px solid #fbbf24; }
    .status-stop { background: linear-gradient(135deg, #dc2626, #ef4444); color: white; border: 2px solid #f87171; }
    
    .numbers-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 20px 0; }
    .num-card {
        background: linear-gradient(135deg, #1e293b, #334155); border: 3px solid #60a5fa;
        border-radius: 16px; padding: 25px 20px; text-align: center;
    }
    .num-value { font-size: 56px; font-weight: 900; color: #60a5fa; line-height: 1; }
    .num-label { font-size: 12px; color: #94a3b8; margin-top: 10px; text-transform: uppercase; }
    
    .support-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 20px 0; }
    .support-card {
        background: linear-gradient(135deg, #1e293b, #334155); border: 2px solid #34d399;
        border-radius: 12px; padding: 18px 12px; text-align: center;
    }
    .support-value { font-size: 36px; font-weight: 800; color: #34d399; }
    
    .info-box {
        background: rgba(96,165,250,0.1); border-left: 4px solid #60a5fa;
        border-radius: 10px; padding: 15px 20px; margin: 15px 0;
    }
    
    .pattern-box {
        background: rgba(239,68,68,0.1); border: 2px solid #ef4444;
        border-radius: 12px; padding: 15px; margin: 15px 0;
    }
    .pattern-title { font-weight: 700; color: #ef4444; margin-bottom: 10px; }
    
    .stButton > button {
        background: linear-gradient(135deg, #1e3a8a, #7c3aed); color: white !important;
        border: none; border-radius: 12px; font-weight: 700; padding: 14px 32px; font-size: 15px;
    }
    
    .stTextArea textarea, .stTextInput input {
        background-color: #1e293b !important; color: #ffffff !important;
        border: 2px solid #475569 !important; border-radius: 12px;
    }
    
    @media (max-width: 600px) {
        .stats-grid { grid-template-columns: repeat(2, 1fr); }
        .num-value { font-size: 42px; }
        .support-value { font-size: 28px; }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. HOUSE PATTERN DETECTION ENGINE
# ==============================================================================

class HousePatternDetector:
    """
    Specialized detector for house manipulation patterns:
    - Bệt cầu (streaks)
    - Đảo cầu (reversals)
    - Xoay cầu (rotations)
    - Bẫy nhịp (rhythm traps)
    """
    
    def __init__(self):
        self.detected_patterns = []
        self.risk_level = 0
    
    def detect_all_patterns(self, data):
        """Run all pattern detection algorithms."""
        patterns = {}
        patterns['bet_cau'] = self._detect_bet_cau(data)  # Bệt cầu
        patterns['dao_cau'] = self._detect_dao_cau(data)  # Đảo cầu
        patterns['xoay_cau'] = self._detect_xoay_cau(data)  # Xoay cầu
        patterns['nhip_bay'] = self._detect_nhip_bay(data)  # Bẫy nhịp
        patterns['tong_control'] = self._detect_sum_control(data)  # Kiểm soát tổng
        
        # Calculate overall house control risk
        self.risk_level = self._calculate_house_control_risk(patterns)
        self.detected_patterns = patterns
        
        return patterns
    
    def _detect_bet_cau(self, data):
        """
        Phát hiện bệt cầu (streaks) - Nhà cái ra cùng số nhiều kỳ liên tiếp
        """
        if len(data) < 20:
            return {'detected': False, 'patterns': [], 'risk': 0}
        
        recent = data[:50] if len(data) >= 50 else data
        patterns = []
        risk = 0
        
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            i = 0
            while i < len(seq) - 2:
                if seq[i] == seq[i+1] == seq[i+2]:
                    d = seq[i]
                    streak = 3
                    j = i + 3
                    while j < len(seq) and seq[j] == d:
                        streak += 1
                        j += 1
                    
                    if streak >= 3:
                        patterns.append({
                            'type': 'Bệt cầu',
                            'position': pos,
                            'digit': d,
                            'streak': streak,
                            'description': f'Vị {pos}: Số {d} bệt {streak} kỳ'
                        })
                        # Risk increases with streak length
                        if streak >= 5:
                            risk += 40
                        elif streak >= 4:
                            risk += 25
                        else:
                            risk += 15
                    i = j
                else:
                    i += 1
        
        return {
            'detected': len(patterns) > 0,
            'patterns': patterns,
            'risk': min(100, risk),
            'max_streak': max([p['streak'] for p in patterns]) if patterns else 0
        }
    
    def _detect_dao_cau(self, data):
        """
        Phát hiện đảo cầu (reversals) - AB → BA → AB
        """
        if len(data) < 20:
            return {'detected': False, 'patterns': [], 'risk': 0}
        
        recent = data[:50] if len(data) >= 50 else data
        patterns = []
        risk = 0
        
        # Check 2-digit reversals
        for i in range(len(recent) - 3):
            a, b = recent[i], recent[i+1]
            c, d = recent[i+2], recent[i+3]
            
            if len(a) >= 2 and len(b) >= 2:
                # AB → BA pattern
                if a[0:2] == b[1::-1]:
                    patterns.append({
                        'type': 'Đảo cầu 2 số',
                        'position': i,
                        'pattern': f'{a[0:2]} → {b[0:2]}',
                        'description': f'Kỳ {i}: {a[0:2]} đảo thành {b[0:2]}'
                    })
                    risk += 10
                
                # AB → BA → AB pattern (full reversal cycle)
                if len(c) >= 2 and len(d) >= 2:
                    if a[0:2] == b[1::-1] == c[0:2]:
                        patterns.append({
                            'type': 'Đảo cầu hoàn chỉnh',
                            'position': i,
                            'pattern': f'{a[0:2]} → {b[0:2]} → {c[0:2]}',
                            'description': f'Chu kỳ đảo: {a[0:2]} → {b[0:2]} → {c[0:2]}'
                        })
                        risk += 25
        
        # Check position swaps
        for pos1 in range(4):
            for pos2 in range(pos1+1, 5):
                seq1 = [n[pos1] if len(n) > pos1 else '0' for n in recent[:20]]
                seq2 = [n[pos2] if len(n) > pos2 else '0' for n in recent[:20]]
                
                # Check if digits swap between positions
                swaps = 0
                for i in range(len(seq1) - 1):
                    if seq1[i] == seq2[i+1] and seq2[i] == seq1[i+1]:
                        swaps += 1
                
                if swaps >= 3:
                    patterns.append({
                        'type': 'Đảo vị trí',
                        'positions': [pos1, pos2],
                        'swaps': swaps,
                        'description': f'Vị {pos1} và {pos2} đảo {swaps} lần'
                    })
                    risk += 15
        
        return {
            'detected': len(patterns) > 0,
            'patterns': patterns[:10],
            'risk': min(100, risk)
        }
    
    def _detect_xoay_cau(self, data):
        """
        Phát hiện xoay cầu (rotations) - Số xoay vòng theo chu kỳ
        """
        if len(data) < 30:
            return {'detected': False, 'patterns': [], 'risk': 0}
        
        recent = data[:60] if len(data) >= 60 else data
        patterns = []
        risk = 0
        
        # Check for rotating digits across positions
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            
            # Check for repeating cycles (3-period, 4-period, 5-period)
            for cycle_len in [3, 4, 5]:
                if len(seq) >= cycle_len * 3:
                    cycle_matches = 0
                    for i in range(len(seq) - cycle_len * 3):
                        base = seq[i:i+cycle_len]
                        match = True
                        for j in range(1, 3):
                            if seq[i+j*cycle_len:i+(j+1)*cycle_len] != base:
                                match = False
                                break
                        if match:
                            cycle_matches += 1
                    
                    if cycle_matches >= 2:
                        patterns.append({
                            'type': 'Xoay cầu chu kỳ',
                            'position': pos,
                            'cycle_length': cycle_len,
                            'matches': cycle_matches,
                            'description': f'Vị {pos}: Chu kỳ {cycle_len} kỳ lặp {cycle_matches} lần'
                        })
                        risk += 20
        
        # Check for digit rotation (0→1→2→3...)
        for pos in range(5):
            seq = [int(n[pos]) if len(n) > pos and n[pos].isdigit() else 0 for n in recent[:30]]
            increment_matches = 0
            for i in range(len(seq) - 1):
                diff = (seq[i+1] - seq[i]) % 10
                if diff == 1 or diff == 9:  # +1 or -1 rotation
                    increment_matches += 1
            
            if increment_matches >= 15:
                patterns.append({
                    'type': 'Xoay số liên tiếp',
                    'position': pos,
                    'matches': increment_matches,
                    'description': f'Vị {pos}: {increment_matches} lần xoay +1/-1'
                })
                risk += 15
        
        return {
            'detected': len(patterns) > 0,
            'patterns': patterns[:10],
            'risk': min(100, risk)
        }
    
    def _detect_nhip_bay(self, data):
        """
        Phát hiện bẫy nhịp (rhythm traps) - Nhà cái tạo nhịp để dụ
        """
        if len(data) < 20:
            return {'detected': False, 'patterns': [], 'risk': 0}
        
        recent = data[:50] if len(data) >= 50 else data
        patterns = []
        risk = 0
        
        # Check for rhythm-2 that breaks (trap)
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            for i in range(len(seq) - 5):
                # X _ X _ X pattern
                if seq[i] == seq[i+2] == seq[i+4] and seq[i] != seq[i+1]:
                    d = seq[i]
                    # Check if next breaks the pattern (trap)
                    if i+5 < len(seq) and seq[i+5] != d:
                        patterns.append({
                            'type': 'Bẫy nhịp 2',
                            'position': pos,
                            'digit': d,
                            'description': f'Vị {pos}: Số {d} nhịp 2 bị gãy ở kỳ {i+5}'
                        })
                        risk += 15
        
        # Check for false streaks (3-4 then breaks)
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            i = 0
            while i < len(seq) - 3:
                if seq[i] == seq[i+1] == seq[i+2]:
                    d = seq[i]
                    # Check if streak breaks at 3-4 (common trap)
                    if i+3 < len(seq) and seq[i+3] != d:
                        patterns.append({
                            'type': 'Bẫy bệt ngắn',
                            'position': pos,
                            'digit': d,
                            'description': f'Vị {pos}: Số {d} bệt 3 kỳ rồi gãy (bẫy)'
                        })
                        risk += 12
                    i += 3
                else:
                    i += 1
        
        return {
            'detected': len(patterns) > 0,
            'patterns': patterns[:10],
            'risk': min(100, risk)
        }
    
    def _detect_sum_control(self, data):
        """
        Phát hiện kiểm soát tổng (sum control) - Nhà cái kiểm soát tổng các số
        """
        if len(data) < 20:
            return {'detected': False, 'patterns': [], 'risk': 0}
        
        recent = data[:50] if len(data) >= 50 else data
        sums = [sum(int(d) for d in n) for n in recent if len(n) == 5 and n.isdigit()]
        
        if len(sums) < 10:
            return {'detected': False, 'patterns': [], 'risk': 0}
        
        patterns = []
        risk = 0
        
        # Check for unusually stable sums
        sum_std = np.std(sums)
        if sum_std < 2.5:
            patterns.append({
                'type': 'Kiểm soát tổng',
                'std_dev': round(sum_std, 2),
                'avg_sum': round(np.mean(sums), 1),
                'description': f'Độ lệch chuẩn tổng: {sum_std:.2f} (quá ổn định)'
            })
            risk += 30
        
        # Check for sum range control
        sum_range = max(sums) - min(sums)
        if sum_range < 10 and len(sums) >= 30:
            patterns.append({
                'type': 'Giới hạn tổng',
                'range': sum_range,
                'description': f'Tổng chỉ dao động trong khoảng {sum_range} (bất thường)'
            })
            risk += 20
        
        # Check for repeating sums
        sum_freq = Counter(sums)
        most_common_sum, most_common_count = sum_freq.most_common(1)[0]
        if most_common_count > len(sums) * 0.2:  # Same sum > 20% of time
            patterns.append({
                'type': 'Tổng lặp',
                'sum': most_common_sum,
                'count': most_common_count,
                'description': f'Tổng {most_common_sum} xuất hiện {most_common_count} lần ({most_common_count/len(sums)*100:.0f}%)'
            })
            risk += 15
        
        return {
            'detected': len(patterns) > 0,
            'patterns': patterns,
            'risk': min(100, risk)
        }
    
    def _calculate_house_control_risk(self, patterns):
        """Calculate overall house control risk level."""
        total_risk = 0
        
        if patterns['bet_cau']['detected']:
            total_risk += patterns['bet_cau']['risk'] * 0.3
        if patterns['dao_cau']['detected']:
            total_risk += patterns['dao_cau']['risk'] * 0.2
        if patterns['xoay_cau']['detected']:
            total_risk += patterns['xoay_cau']['risk'] * 0.2
        if patterns['nhip_bay']['detected']:
            total_risk += patterns['nhip_bay']['risk'] * 0.15
        if patterns['tong_control']['detected']:
            total_risk += patterns['tong_control']['risk'] * 0.15
        
        return min(100, int(total_risk))
    
    def get_house_control_level(self):
        """Get house control level description."""
        if self.risk_level >= 70:
            return 'RẤT CAO', '🚫 Nhà cái đang điều khiển mạnh - NÊN DỪNG'
        elif self.risk_level >= 50:
            return 'CAO', '⚠️ Có dấu hiệu điều khiển - CẨN THẬN'
        elif self.risk_level >= 30:
            return 'TRUNG BÌNH', '⚠️ Một số pattern bất thường'
        else:
            return 'THẤP', '✅ Nhịp số tự nhiên'

# ==============================================================================
# 3. ENHANCED AI ENGINE WITH PATTERN AWARENESS
# ==============================================================================

class TitanAI:
    def __init__(self):
        self.weights = {
            'frequency': 25, 'gap': 20, 'markov': 20,
            'monte_carlo': 15, 'pattern': 12, 'hot_cold': 8
        }
        self.accuracy_history = []
        self.pattern_detector = HousePatternDetector()
    
    def analyze(self, history, max_simulations=2000):
        if not history or len(history) < 15:
            return self._fallback()
        
        clean_data = self._clean_history(history)
        if len(clean_data) < 15:
            return self._fallback("Cần ít nhất 15 kỳ")
        
        # Detect house patterns FIRST
        house_patterns = self.pattern_detector.detect_all_patterns(clean_data)
        house_control_level, house_warning = self.pattern_detector.get_house_control_level()
        
        # Run standard analysis
        results = {}
        results['frequency'] = self._analyze_frequency(clean_data)
        results['gap'] = self._analyze_gap(clean_data)
        results['markov'] = self._analyze_markov(clean_data)
        results['monte_carlo'] = self._analyze_monte_carlo(clean_data, max_simulations)
        results['pattern'] = self._analyze_pattern_advanced(clean_data)
        results['hot_cold'] = self._analyze_hot_cold(clean_data)
        
        # Adjust weights based on house patterns
        if house_patterns['bet_cau']['detected'] and house_patterns['bet_cau']['risk'] >= 40:
            # Reduce frequency weight during strong streaks (likely to break)
            self.weights['frequency'] = 15
            self.weights['pattern'] = 25
        
        ensemble = self._ensemble_vote(results, house_patterns)
        stats_df = self._build_stats_df(clean_data, results)
        risk = self._calculate_risk(clean_data, house_patterns)
        
        return {
            'main_3': ensemble['main_3'],
            'support_4': ensemble['support_4'],
            'stats_df': stats_df,
            'risk': risk,
            'confidence': ensemble['confidence'],
            'logic': self._build_logic(results, ensemble, house_patterns),
            'house_patterns': house_patterns,
            'house_control_level': house_control_level,
            'house_warning': house_warning
        }
    
    def record_accuracy(self, prediction, actual_result, won):
        self.accuracy_history.append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'actual': actual_result,
            'won': won,
            'confidence': prediction.get('confidence', 0)
        })
        if len(self.accuracy_history) > 100:
            self.accuracy_history.pop(0)
    
    def get_accuracy_stats(self):
        if not self.accuracy_history:
            return {'total': 0, 'wins': 0, 'win_rate': 0, 'avg_confidence': 0, 'by_confidence': {}}
        
        total = len(self.accuracy_history)
        wins = sum(1 for h in self.accuracy_history if h['won'])
        win_rate = wins / total * 100 if total > 0 else 0
        avg_conf = sum(h['confidence'] for h in self.accuracy_history) / total
        
        by_confidence = {}
        for bracket in ['50-69', '70-84', '85+']:
            if bracket == '50-69':
                subset = [h for h in self.accuracy_history if 50 <= h['confidence'] < 70]
            elif bracket == '70-84':
                subset = [h for h in self.accuracy_history if 70 <= h['confidence'] < 85]
            else:
                subset = [h for h in self.accuracy_history if h['confidence'] >= 85]
            
            if subset:
                w = sum(1 for h in subset if h['won'])
                by_confidence[bracket] = {'count': len(subset), 'win_rate': round(w / len(subset) * 100, 1)}
        
        return {
            'total': total, 'wins': wins, 'win_rate': round(win_rate, 1),
            'avg_confidence': round(avg_conf, 1), 'by_confidence': by_confidence
        }
    
    def _clean_history(self, history):
        cleaned = []
        for item in history:
            s = str(item).strip()
            match = re.search(r'\d{5}', s)
            if match:
                cleaned.append(match.group())
        return cleaned
    
    def _analyze_frequency(self, data):
        weighted = Counter()
        n = len(data)
        for idx, num in enumerate(data):
            weight = 3.0 - 2.0 * (idx / max(n, 1))
            for d in num:
                weighted[d] += weight
        scores = {d: weighted.get(d, 0) for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        return {'scores': scores, 'top_3': top_3}
    
    def _analyze_gap(self, data):
        last_seen = {d: -1 for d in '0123456789'}
        for idx, num in enumerate(data):
            for d in num:
                if last_seen[d] == -1:
                    last_seen[d] = idx
        scores = {}
        for d in '0123456789':
            gap = last_seen[d] if last_seen[d] >= 0 else len(data)
            scores[d] = gap * 2.5 if gap <= 15 else max(0, 37.5 - (gap - 15) * 0.5)
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        return {'scores': scores, 'top_3': top_3}
    
    def _analyze_markov(self, data):
        if len(data) < 20:
            return {'scores': {d: 10 for d in '0123456789'}, 'top_3': ['1','5','9']}
        transitions = defaultdict(Counter)
        for i in range(len(data) - 1):
            curr, next_num = data[i], data[i + 1]
            for pos in range(5):
                if pos < len(curr) and pos < len(next_num):
                    transitions[curr[pos]][next_num[pos]] += 1
        last_num = data[0] if data else '00000'
        next_prob = Counter()
        for pos, last_d in enumerate(last_num[:5]):
            if last_d in transitions and transitions[last_d]:
                total = sum(transitions[last_d].values())
                for next_d, count in transitions[last_d].items():
                    pos_weight = 1.0 + 0.25 * (2 - abs(pos - 2))
                    next_prob[next_d] += (count / total) * pos_weight
        scores = {d: next_prob.get(d, 0) * 10 for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        while len(top_3) < 3:
            for d in '0123456789':
                if d not in top_3:
                    top_3.append(d)
                    break
        return {'scores': scores, 'top_3': top_3}
    
    def _analyze_monte_carlo(self, data, n_simulations):
        if len(data) < 20:
            return {'scores': {d: 10 for d in '0123456789'}, 'top_3': ['2','4','6']}
        recent = data[:80] if len(data) >= 80 else data
        pool = []
        for idx, num in enumerate(recent):
            weight = max(1, 4 - idx // 20)
            for d in num:
                pool.extend([d] * weight)
        if not pool:
            return {'scores': {d: 10 for d in '0123456789'}, 'top_3': ['0','1','2']}
        sim_count = Counter()
        for _ in range(n_simulations):
            sample = random.choices(pool, k=3)
            for d in sample:
                sim_count[d] += 1
        total = sum(sim_count.values()) or 1
        scores = {d: sim_count.get(d, 0) / total * 100 for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        return {'scores': scores, 'top_3': top_3}
    
    def _analyze_pattern_advanced(self, data):
        if len(data) < 25:
            return {'scores': {d: 10 for d in '0123456789'}, 'top_3': ['3','5','7'], 'patterns': []}
        recent = data[:50] if len(data) >= 50 else data
        candidates = Counter()
        patterns_found = []
        avoid = []
        
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            i = 0
            while i < len(seq) - 2:
                if seq[i] == seq[i+1] == seq[i+2]:
                    d = seq[i]
                    streak_len = 3
                    j = i + 3
                    while j < len(seq) and seq[j] == d:
                        streak_len += 1
                        j += 1
                    patterns_found.append(f'Bệt vị {pos}: {d} ({streak_len} kỳ)')
                    if streak_len >= 5:
                        avoid.append(d)
                    elif streak_len >= 3:
                        candidates[d] += 5
                    i = j
                else:
                    i += 1
        
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            for i in range(len(seq) - 4):
                if seq[i] == seq[i+2] == seq[i+4] and seq[i] != seq[i+1]:
                    d = seq[i]
                    patterns_found.append(f'Nhịp-2 vị {pos}: {d}')
                    candidates[d] += 4
        
        if not candidates:
            all_digits = ''.join(recent)
            freq = Counter(all_digits)
            for d, c in freq.most_common(3):
                candidates[d] += 3
        
        scores = {d: candidates.get(d, 0) * 2 for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        while len(top_3) < 3:
            for d in '0123456789':
                if d not in top_3:
                    top_3.append(d)
                    break
        
        return {'scores': scores, 'top_3': top_3, 'patterns': patterns_found[:10], 'avoid': list(set(avoid))}
    
    def _analyze_hot_cold(self, data):
        recent = data[:15] if len(data) >= 15 else data
        older = data[15:45] if len(data) >= 45 else data
        recent_count = Counter(''.join(recent))
        older_count = Counter(''.join(older)) if older else Counter()
        scores = {}
        for d in '0123456789':
            r = recent_count.get(d, 0)
            o = older_count.get(d, 0)
            if r >= 4:
                scores[d] = 25 + r * 3
            elif r == 0 and o >= 3:
                scores[d] = 20 + o * 2
            elif r >= 2:
                scores[d] = 15 + r * 2
            else:
                scores[d] = 10 + r
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        return {'scores': scores, 'top_3': top_3}
    
    def _ensemble_vote(self, results, house_patterns):
        votes = Counter()
        avoid_votes = []
        
        # If house control is high, avoid streaking numbers
        if house_patterns['bet_cau']['detected'] and house_patterns['bet_cau']['risk'] >= 40:
            for p in house_patterns['bet_cau']['patterns']:
                if p['streak'] >= 4:
                    avoid_votes.append(p['digit'])
        
        for algo_name, result in results.items():
            weight = self.weights.get(algo_name, 10)
            for d in result.get('top_3', []):
                votes[d] += weight
            if result.get('avoid'):
                avoid_votes.extend(result['avoid'])
        
        avoid_set = set(avoid_votes)
        main_3 = [d for d, _ in votes.most_common(3) if d not in avoid_set]
        
        while len(main_3) < 3:
            for d in '0123456789':
                if d not in main_3 and d not in avoid_set:
                    main_3.append(d)
                    break
        
        remaining = [d for d, _ in votes.most_common(10) if d not in main_3 and d not in avoid_set]
        support_4 = remaining[:4]
        while len(support_4) < 4:
            for d in '0123456789':
                if d not in main_3 and d not in support_4 and d not in avoid_set:
                    support_4.append(d)
                    break
        
        if votes:
            top_votes = [c for _, c in votes.most_common(3)]
            confidence = min(95, 55 + sum(top_votes) / 3)
        else:
            confidence = 50
        
        return {'main_3': main_3, 'support_4': support_4, 'confidence': int(confidence), 'avoid': list(avoid_set)}
    
    def _build_stats_df(self, data, results):
        rows = []
        for d in '0123456789':
            row = {'Digit': d}
            row['Frequency'] = results['frequency']['scores'].get(d, 0)
            row['Gap'] = results['gap']['scores'].get(d, 0)
            row['Markov'] = results['markov']['scores'].get(d, 0)
            row['Monte_Carlo'] = results['monte_carlo']['scores'].get(d, 0)
            row['Pattern'] = results['pattern']['scores'].get(d, 0)
            row['Hot_Cold'] = results['hot_cold']['scores'].get(d, 0)
            ai_score = (row['Frequency'] * 0.25 + row['Gap'] * 0.20 + 
                       row['Markov'] * 0.20 + row['Monte_Carlo'] * 0.15 + 
                       row['Pattern'] * 0.12 + row['Hot_Cold'] * 0.08)
            row['AI_Score'] = round(ai_score, 1)
            rows.append(row)
        df = pd.DataFrame(rows)
        return df.sort_values('AI_Score', ascending=False).reset_index(drop=True)
    
    def _calculate_risk(self, data, house_patterns):
        base_risk = 0
        reasons = []
        
        # Add house control risk
        house_risk = self.pattern_detector.risk_level
        if house_risk >= 50:
            base_risk += house_risk * 0.5
            reasons.append(f'Nhà cái điều khiển: {house_risk}%')
        
        # Standard risk calculation
        if len(data) < 20:
            return {'score': 30, 'level': 'MEDIUM', 'reason': 'Dữ liệu ít'}
        
        all_digits = ''.join(data[:50])
        counts = Counter(all_digits)
        total = len(all_digits)
        
        entropy = sum(- (c/total) * math.log2(c/total) for c in counts.values() if c > 0)
        if entropy < 2.8:
            base_risk += 25
            reasons.append('Kết quả quá đều')
        elif entropy > 3.4:
            base_risk += 15
            reasons.append('Biến động mạnh')
        
        base_risk = min(100, int(base_risk))
        level = 'HIGH' if base_risk >= 50 else 'MEDIUM' if base_risk >= 25 else 'OK'
        
        return {'score': base_risk, 'level': level, 'reason': '; '.join(reasons) if reasons else 'Ổn định'}
    
    def _build_logic(self, results, ensemble, house_patterns):
        parts = []
        freq_top = [d for d, _ in sorted(results['frequency']['scores'].items(), key=lambda x: -x[1])[:2]]
        if freq_top:
            parts.append(f"Tần suất: {','.join(freq_top)}")
        
        if house_patterns['bet_cau']['detected']:
            parts.append(f"Bệt: {house_patterns['bet_cau']['max_streak']} kỳ")
        
        if ensemble['confidence'] >= 75:
            parts.append('Đồng thuận cao')
        
        if ensemble.get('avoid'):
            parts.append(f"Tránh: {','.join(ensemble['avoid'][:2])}")
        
        return ' | '.join(parts) if parts else 'Phân tích AI'
    
    def _fallback(self, msg="Chưa đủ dữ liệu"):
        return {
            'main_3': ['?', '?', '?'], 'support_4': ['0', '0', '0', '0'],
            'stats_df': pd.DataFrame({'Digit': list('0123456789'), 'AI_Score': [0]*10}),
            'risk': {'score': 0, 'level': 'LOW', 'reason': msg},
            'confidence': 0, 'logic': msg,
            'house_patterns': {}, 'house_control_level': 'N/A', 'house_warning': ''
        }

# ==============================================================================
# 4. MAIN APPLICATION
# ==============================================================================

def main():
    if 'db' not in st.session_state:
        st.session_state.db = []
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'ai' not in st.session_state:
        st.session_state.ai = TitanAI()
    if 'test_log' not in st.session_state:
        st.session_state.test_log = []
    
    # Header
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🎯 TITAN AI v5.0</div>
        <div class="header-subtitle">House Pattern Detector | Phát hiện bẫy nhà cái</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Độ Chính Xác")
        acc_stats = st.session_state.ai.get_accuracy_stats()
        st.metric("Tổng lần test", acc_stats['total'])
        st.metric("Trúng", acc_stats['wins'])
        st.metric("Win Rate", f"{acc_stats['win_rate']}%")
        
        st.markdown("---")
        if st.button("🗑️ Reset"):
            st.session_state.test_log = []
            st.session_state.ai.accuracy_history = []
            st.success("✅ Đã reset!")
            time.sleep(0.5)
            st.rerun()
    
    # Stats Overview
    acc_stats = st.session_state.ai.get_accuracy_stats()
    st.markdown("### 📊 Thống kê")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Tổng kỳ", len(st.session_state.db))
    with col2:
        st.metric("🎯 Đã test", acc_stats['total'])
    with col3:
        color = "🟢" if acc_stats['win_rate'] >= 40 else "🟡" if acc_stats['win_rate'] >= 25 else "🔴"
        st.metric("Win Rate", f"{color} {acc_stats['win_rate']}%")
    with col4:
        if st.session_state.result:
            st.metric("🏠 House Control", f"{st.session_state.ai.pattern_detector.risk_level}%")
    
    # Input
    st.markdown("### 📥 Nhập kết quả")
    raw_input = st.text_area(
        "Dán kết quả (mỗi kỳ 1 dòng, 5 chữ số):",
        height=150,
        placeholder="09215\n23823\n45976\n...",
        key="data_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        analyze_btn = st.button("🚀 PHÂN TÍCH", type="primary", use_container_width=True)
    with col2:
        if st.button("📋 Demo", use_container_width=True):
            demo = "\n".join(["87746", "56421", "69137", "00443", "04475",
                            "64472", "16755", "58569", "62640", "99723"] * 5)
            st.session_state.data_input = demo
            st.rerun()
    with col3:
        if st.button("🔄 Mới", use_container_width=True):
            st.rerun()
    
    # Process
    if analyze_btn and raw_input.strip():
        with st.spinner("🧠 Đang phân tích pattern nhà cái..."):
            numbers = re.findall(r'\d{5}', raw_input)
            if not numbers:
                st.error("❌ Không tìm thấy số 5 chữ số!")
            else:
                existing_set = set(st.session_state.db)
                new_numbers = [n for n in numbers if n not in existing_set]
                if new_numbers:
                    st.session_state.db = new_numbers + st.session_state.db
                    if len(st.session_state.db) > 500:
                        st.session_state.db = st.session_state.db[:500]
                    st.success(f"✅ Đã thêm {len(new_numbers)} số mới")
                
                if len(st.session_state.db) >= 15:
                    st.session_state.result = st.session_state.ai.analyze(st.session_state.db)
                    st.rerun()
                else:
                    st.warning(f"⚠️ Cần ít nhất 15 kỳ (hiện có: {len(st.session_state.db)})")
    
    # Display Results
    if st.session_state.result:
        res = st.session_state.result
        risk = res['risk']
        
        # House Control Warning
        if st.session_state.ai.pattern_detector.risk_level >= 50:
            st.markdown(f"""
            <div class="pattern-alert">
                🚨 CẢNH BÁO: Nhà cái đang điều khiển ({st.session_state.ai.pattern_detector.risk_level}%)<br>
                <small>{res['house_warning']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Status
        if risk['level'] == 'OK':
            status_class, status_text = 'status-ok', '✅ CÓ THỂ TEST'
        elif risk['level'] == 'MEDIUM':
            status_class, status_text = 'status-warn', '⚠️ CẨN THẬN'
        else:
            status_class, status_text = 'status-stop', '🛑 RỦI RO CAO'
        
        st.markdown(f"""
        <div class="status-card {status_class}">
            {status_text} | Risk: {risk['score']}/100 | {risk['reason']}
        </div>
        """, unsafe_allow_html=True)
        
        # House Patterns Detail
        if res.get('house_patterns'):
            st.markdown("### 🔍 Pattern Nhà Cái Phát Hiện")
            
            hp = res['house_patterns']
            
            if hp['bet_cau']['detected']:
                st.markdown(f"""
                <div class="pattern-box">
                    <div class="pattern-title">📊 Bệt Cầu ({hp['bet_cau']['risk']}% risk)</div>
                """, unsafe_allow_html=True)
                for p in hp['bet_cau']['patterns'][:5]:
                    st.markdown(f"- {p['description']}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            if hp['dao_cau']['detected']:
                st.markdown(f"""
                <div class="pattern-box">
                    <div class="pattern-title">🔄 Đảo Cầu ({hp['dao_cau']['risk']}% risk)</div>
                """, unsafe_allow_html=True)
                for p in hp['dao_cau']['patterns'][:5]:
                    st.markdown(f"- {p['description']}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            if hp['nhip_bay']['detected']:
                st.markdown(f"""
                <div class="pattern-box">
                    <div class="pattern-title">⚠️ Bẫy Nhịp ({hp['nhip_bay']['risk']}% risk)</div>
                """, unsafe_allow_html=True)
                for p in hp['nhip_bay']['patterns'][:5]:
                    st.markdown(f"- {p['description']}")
                st.markdown("</div>", unsafe_allow_html=True)
        
        # 3 Main Numbers
        st.markdown("### 🔮 3 SỐ DỰ ĐOÁN")
        main_3 = res['main_3']
        st.markdown(f"""
        <div class="numbers-grid">
            <div class="num-card"><div class="num-value">{main_3[0]}</div><div class="num-label">Số 1</div></div>
            <div class="num-card"><div class="num-value">{main_3[1]}</div><div class="num-label">Số 2</div></div>
            <div class="num-card"><div class="num-value">{main_3[2]}</div><div class="num-label">Số 3</div></div>
        </div>
        """, unsafe_allow_html=True)
        
        # 4 Support Numbers
        st.markdown("### 🎲 4 SỐ THAM KHẢO")
        support_4 = res['support_4']
        st.markdown(f"""
        <div class="support-grid">
            <div class="support-card"><div class="support-value">{support_4[0]}</div></div>
            <div class="support-card"><div class="support-value">{support_4[1]}</div></div>
            <div class="support-card"><div class="support-value">{support_4[2]}</div></div>
            <div class="support-card"><div class="support-value">{support_4[3]}</div></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.code(','.join(main_3 + support_4), language=None)
        
        if res['logic']:
            st.markdown(f'<div class="info-box">💡 <strong>Logic:</strong> {res["logic"]}</div>', unsafe_allow_html=True)
        
        # Test Verification
        st.markdown("---")
        st.markdown("### ✅ Test Độ Chính Xác")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            actual = st.text_input("Kết quả thực tế:", key="actual_result", placeholder="12864")
        with col2:
            if st.button("✅ GHI NHẬN", type="primary", use_container_width=True):
                if actual and len(actual) == 5 and actual.isdigit():
                    is_win = set(main_3).issubset(set(actual))
                    st.session_state.ai.record_accuracy(res, actual, is_win)
                    st.session_state.test_log.append({
                        'timestamp': datetime.now().isoformat(),
                        'prediction': main_3,
                        'actual': actual,
                        'won': is_win,
                        'confidence': res['confidence'],
                        'house_risk': st.session_state.ai.pattern_detector.risk_level
                    })
                    if is_win:
                        st.success(f"🎉 TRÚNG! (Conf: {res['confidence']}%, House: {st.session_state.ai.pattern_detector.risk_level}%)")
                    else:
                        missing = set(main_3) - set(actual)
                        st.warning(f"❌ Trượt! Thiếu: {', '.join(missing)}")
                    st.rerun()
    
    # Test History
    if st.session_state.test_log:
        st.markdown("---")
        st.markdown("### 📜 Lịch sử Test")
        df_test = pd.DataFrame(st.session_state.test_log[-20:])
        df_test['timestamp'] = pd.to_datetime(df_test['timestamp']).dt.strftime('%H:%M %d/%m')
        df_test['prediction'] = df_test['prediction'].apply(lambda x: ','.join(x))
        df_test['status'] = df_test['won'].apply(lambda x: '✅' if x else '❌')
        display_cols = ['timestamp', 'prediction', 'actual', 'status', 'confidence', 'house_risk']
        st.dataframe(df_test[display_cols], hide_index=True, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; padding: 20px; font-size: 12px;">
        🎯 TITAN AI v5.0 | House Pattern Detector<br>
        🔍 Phát hiện: Bệt cầu | Đảo cầu | Xoay cầu | Bẫy nhịp<br>
        ⚠️ Khi House Control >= 50%: Nên dừng
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()