# ==============================================================================
# TITAN AI v5.0 - AI Algorithms
# ==============================================================================

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import random
import math
import re
from typing import Dict, List, Tuple, Optional
from config import Config

class HousePatternDetector:
    """House pattern detector."""
    
    def __init__(self):
        self.detected_patterns = {}
        self.risk_level = 0
    
    def detect_all_patterns(self, data):
        """Detect all patterns."""
        try:
            patterns = {}
            patterns['bet_cau'] = self._detect_bet_cau(data)
            patterns['dao_cau'] = self._detect_dao_cau(data)
            patterns['xoay_cau'] = self._detect_xoay_cau(data)
            patterns['nhip_bay'] = self._detect_nhip_bay(data)
            patterns['tong_control'] = self._detect_sum_control(data)
            
            self.risk_level = self._calculate_risk(patterns)
            self.detected_patterns = patterns
            
            return patterns
        except:
            return {
                'bet_cau': {'detected': False, 'patterns': [], 'risk': 0, 'max_streak': 0},
                'dao_cau': {'detected': False, 'patterns': [], 'risk': 0},
                'xoay_cau': {'detected': False, 'patterns': [], 'risk': 0},
                'nhip_bay': {'detected': False, 'patterns': [], 'risk': 0},
                'tong_control': {'detected': False, 'patterns': [], 'risk': 0}
            }
    
    def _detect_bet_cau(self, data):
        """Detect streaks."""
        if len(data) < 20:
            return {'detected': False, 'patterns': [], 'risk': 0, 'max_streak': 0}
        
        recent = data[:50] if len(data) >= 50 else data
        patterns = []
        risk = 0
        max_streak = 0
        
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
                            'description': f'Vị {pos}: {d} bệt {streak} kỳ'
                        })
                        max_streak = max(max_streak, streak)
                        
                        if streak >= 5:
                            risk += 40
                        elif streak >= 4:
                            risk += 25
                        else:
                            risk += 15
                    i = j
                else:
                    i += 1
        
        return {'detected': len(patterns) > 0, 'patterns': patterns, 'risk': min(100, risk), 'max_streak': max_streak}
    
    def _detect_dao_cau(self, data):
        """Detect reversals."""
        if len(data) < 20:
            return {'detected': False, 'patterns': [], 'risk': 0}
        
        recent = data[:50] if len(data) >= 50 else data
        patterns = []
        risk = 0
        
        for i in range(len(recent) - 3):
            a, b = recent[i], recent[i+1]
            if len(a) >= 2 and len(b) >= 2:
                if a[0:2] == b[1::-1]:
                    patterns.append({
                        'type': 'Đảo cầu',
                        'pattern': f'{a[0:2]} → {b[0:2]}',
                        'description': f'{a[0:2]} → {b[0:2]}'
                    })
                    risk += 10
        
        return {'detected': len(patterns) > 0, 'patterns': patterns[:10], 'risk': min(100, risk)}
    
    def _detect_xoay_cau(self, data):
        """Detect rotations."""
        return {'detected': False, 'patterns': [], 'risk': 0}
    
    def _detect_nhip_bay(self, data):
        """Detect rhythm traps."""
        if len(data) < 20:
            return {'detected': False, 'patterns': [], 'risk': 0}
        
        recent = data[:50] if len(data) >= 50 else data
        patterns = []
        risk = 0
        
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            for i in range(len(seq) - 5):
                if seq[i] == seq[i+2] == seq[i+4] and seq[i] != seq[i+1]:
                    d = seq[i]
                    if i+5 < len(seq) and seq[i+5] != d:
                        patterns.append({
                            'type': 'Bẫy nhịp',
                            'digit': d,
                            'description': f'Vị {pos}: {d} gãy nhịp'
                        })
                        risk += 15
        
        return {'detected': len(patterns) > 0, 'patterns': patterns[:10], 'risk': min(100, risk)}
    
    def _detect_sum_control(self, data):
        """Detect sum control."""
        if len(data) < 20:
            return {'detected': False, 'patterns': [], 'risk': 0}
        
        recent = data[:50] if len(data) >= 50 else data
        sums = [sum(int(d) for d in n) for n in recent if len(n) == 5 and n.isdigit()]
        
        if len(sums) < 10:
            return {'detected': False, 'patterns': [], 'risk': 0}
        
        patterns = []
        risk = 0
        
        sum_std = float(np.std(sums))
        if sum_std < 2.5:
            patterns.append({'type': 'Kiểm soát tổng', 'std': sum_std})
            risk += 30
        
        return {'detected': len(patterns) > 0, 'patterns': patterns, 'risk': min(100, risk)}
    
    def _calculate_risk(self, patterns):
        """Calculate total risk."""
        total = 0
        if patterns.get('bet_cau', {}).get('detected'):
            total += patterns['bet_cau']['risk'] * 0.30
        if patterns.get('dao_cau', {}).get('detected'):
            total += patterns['dao_cau']['risk'] * 0.20
        if patterns.get('nhip_bay', {}).get('detected'):
            total += patterns['nhip_bay']['risk'] * 0.15
        if patterns.get('tong_control', {}).get('detected'):
            total += patterns['tong_control']['risk'] * 0.15
        
        return min(100, int(total))
    
    def get_house_control_level(self):
        """Get house control level."""
        if self.risk_level >= 70:
            return 'RẤT CAO', '🚫 Nhà cái điều khiển mạnh'
        elif self.risk_level >= 50:
            return 'CAO', '⚠️ Có dấu hiệu điều khiển'
        elif self.risk_level >= 30:
            return 'TRUNG BÌNH', '⚠️ Pattern bất thường'
        else:
            return 'THẤP', '✅ Ổn định'


class TitanAI:
    """AI prediction engine."""
    
    def __init__(self):
        self.weights = Config.ALGORITHM_WEIGHTS.copy()
        self.accuracy_history = []
        self.pattern_detector = HousePatternDetector()
    
    def analyze(self, history, max_simulations=None):
        """Main analysis."""
        if max_simulations is None:
            max_simulations = Config.DEFAULT_SIMULATIONS
        
        if not history or len(history) < Config.MIN_HISTORY_LENGTH:
            return self._fallback(f"Cần {Config.MIN_HISTORY_LENGTH}+ kỳ")
        
        clean_data = self._clean_history(history)
        if len(clean_data) < Config.MIN_HISTORY_LENGTH:
            return self._fallback("Data không hợp lệ")
        
        house_patterns = self.pattern_detector.detect_all_patterns(clean_data)
        house_level, house_warning = self.pattern_detector.get_house_control_level()
        
        results = {}
        results['frequency'] = self._frequency(clean_data)
        results['gap'] = self._gap(clean_data)
        results['markov'] = self._markov(clean_data)
        results['monte_carlo'] = self._monte_carlo(clean_data, max_simulations)
        results['pattern'] = self._pattern(clean_data)
        results['hot_cold'] = self._hot_cold(clean_data)
        
        ensemble = self._ensemble(results, house_patterns)
        stats_df = self._stats_df(clean_data, results)
        risk = self._risk(clean_data, house_patterns)
        
        return {
            'main_3': ensemble['main_3'],
            'support_4': ensemble['support_4'],
            'stats_df': stats_df,
            'risk': risk,
            'confidence': ensemble['confidence'],
            'logic': self._logic(results, ensemble, house_patterns),
            'house_patterns': house_patterns,
            'house_control_level': house_level,
            'house_warning': house_warning,
            'success': True
        }
    
    def _clean_history(self, history):
        """Clean data."""
        cleaned = []
        for item in history:
            s = str(item).strip()
            match = re.search(r'\d{5}', s)
            if match:
                cleaned.append(match.group())
        return cleaned
    
    def _frequency(self, data):
        """Frequency analysis."""
        weighted = Counter()
        n = len(data)
        
        for idx, num in enumerate(data):
            weight = 3.0 - 2.0 * (idx / max(n, 1))
            for d in num:
                if d.isdigit():
                    weighted[d] += weight
        
        scores = {d: weighted.get(d, 0) for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        
        while len(top_3) < 3:
            for d in '0123456789':
                if d not in top_3:
                    top_3.append(d)
                    break
        
        return {'scores': scores, 'top_3': top_3}
    
    def _gap(self, data):
        """Gap analysis."""
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
    
    def _markov(self, data):
        """Markov analysis."""
        if len(data) < 20:
            return {'scores': {d: 10.0 for d in '0123456789'}, 'top_3': ['1', '5', '9']}
        
        transitions = defaultdict(Counter)
        
        for i in range(len(data) - 1):
            curr = data[i]
            next_num = data[i + 1]
            for pos in range(5):
                if pos < len(curr) and pos < len(next_num):
                    transitions[curr[pos]][next_num[pos]] += 1
        
        last_num = data[0] if data else '00000'
        next_prob = Counter()
        
        for pos, last_d in enumerate(last_num[:5]):
            if last_d in transitions and transitions[last_d]:
                total = sum(transitions[last_d].values())
                if total > 0:
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
    
    def _monte_carlo(self, data, n_sim):
        """Monte Carlo simulation."""
        if len(data) < 20:
            return {'scores': {d: 10.0 for d in '0123456789'}, 'top_3': ['2', '4', '6']}
        
        recent = data[:80] if len(data) >= 80 else data
        pool = []
        
        for idx, num in enumerate(recent):
            weight = max(1, 4 - idx // 20)
            for d in num:
                if d.isdigit():
                    pool.extend([d] * weight)
        
        if not pool:
            return {'scores': {d: 10.0 for d in '0123456789'}, 'top_3': ['0', '1', '2']}
        
        sim_count = Counter()
        for _ in range(min(n_sim, 2000)):
            sample = random.choices(pool, k=3)
            for d in sample:
                sim_count[d] += 1
        
        total = sum(sim_count.values()) or 1
        scores = {d: sim_count.get(d, 0) / total * 100 for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        
        return {'scores': scores, 'top_3': top_3}
    
    def _pattern(self, data):
        """Pattern detection."""
        if len(data) < 25:
            return {'scores': {d: 10.0 for d in '0123456789'}, 'top_3': ['3', '5', '7'], 'patterns': [], 'avoid': []}
        
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
                    streak = 3
                    j = i + 3
                    while j < len(seq) and seq[j] == d:
                        streak += 1
                        j += 1
                    
                    if streak >= 3:
                        patterns_found.append(f'Bệt {pos}: {d} ({streak})')
                        if streak >= 5:
                            avoid.append(d)
                        else:
                            candidates[d] += 5
                    i = j
                else:
                    i += 1
        
        scores = {d: candidates.get(d, 0) * 2 for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        
        while len(top_3) < 3:
            for d in '0123456789':
                if d not in top_3:
                    top_3.append(d)
                    break
        
        return {'scores': scores, 'top_3': top_3, 'patterns': patterns_found[:10], 'avoid': list(set(avoid))}
    
    def _hot_cold(self, data):
        """Hot/cold analysis."""
        recent = data[:15] if len(data) >= 15 else data
        older = data[15:45] if len(data) >= 45 else []
        
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
    
    def _ensemble(self, results, house_patterns):
        """Ensemble voting."""
        votes = Counter()
        avoid_votes = []
        
        if house_patterns['bet_cau']['detected'] and house_patterns['bet_cau']['risk'] >= 40:
            for p in house_patterns['bet_cau']['patterns']:
                if p.get('streak', 0) >= 4:
                    avoid_votes.append(p['digit'])
        
        for algo, result in results.items():
            weight = self.weights.get(algo, 10)
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
    
    def _stats_df(self, data, results):
        """Build stats DataFrame."""
        rows = []
        for d in '0123456789':
            row = {'Digit': d}
            row['Frequency'] = float(results['frequency']['scores'].get(d, 0))
            row['Gap'] = float(results['gap']['scores'].get(d, 0))
            row['Markov'] = float(results['markov']['scores'].get(d, 0))
            row['Monte_Carlo'] = float(results['monte_carlo']['scores'].get(d, 0))
            row['Pattern'] = float(results['pattern']['scores'].get(d, 0))
            row['Hot_Cold'] = float(results['hot_cold']['scores'].get(d, 0))
            
            ai_score = (row['Frequency'] * 0.25 + row['Gap'] * 0.20 + 
                       row['Markov'] * 0.20 + row['Monte_Carlo'] * 0.15 + 
                       row['Pattern'] * 0.12 + row['Hot_Cold'] * 0.08)
            row['AI_Score'] = round(ai_score, 1)
            rows.append(row)
        
        return pd.DataFrame(rows).sort_values('AI_Score', ascending=False).reset_index(drop=True)
    
    def _risk(self, data, house_patterns):
        """Calculate risk."""
        base_risk = 0
        reasons = []
        
        house_risk = self.pattern_detector.risk_level
        if house_risk >= 50:
            base_risk += house_risk * 0.5
            reasons.append(f'House: {house_risk}%')
        
        if len(data) < 20:
            return {'score': 30, 'level': 'MEDIUM', 'reason': 'Ít data'}
        
        all_digits = ''.join(data[:50])
        counts = Counter(all_digits)
        total = len(all_digits)
        
        if total > 0:
            entropy = sum(- (c/total) * math.log2(c/total) for c in counts.values() if c > 0)
            if entropy < 2.8:
                base_risk += 25
                reasons.append('Quá đều')
            elif entropy > 3.4:
                base_risk += 15
                reasons.append('Biến động')
        
        base_risk = min(100, int(base_risk))
        level = 'HIGH' if base_risk >= 50 else 'MEDIUM' if base_risk >= 25 else 'OK'
        
        return {'score': base_risk, 'level': level, 'reason': '; '.join(reasons) if reasons else 'Ổn'}
    
    def _logic(self, results, ensemble, house_patterns):
        """Build logic string."""
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
        
        return ' | '.join(parts) if parts else 'AI Analysis'
    
    def _fallback(self, msg):
        """Fallback prediction."""
        return {
            'main_3': ['?', '?', '?'],
            'support_4': ['0', '0', '0', '0'],
            'stats_df': pd.DataFrame({'Digit': list('0123456789'), 'AI_Score': [0.0]*10}),
            'risk': {'score': 0, 'level': 'LOW', 'reason': msg},
            'confidence': 0,
            'logic': msg,
            'house_patterns': {},
            'house_control_level': 'N/A',
            'house_warning': '',
            'success': False
        }