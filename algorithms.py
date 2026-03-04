# ==============================================================================
# TITAN v35.0 - Prediction Algorithms
# Multi-Algorithm Ensemble Engine
# ==============================================================================

import numpy as np
from collections import Counter, defaultdict
from datetime import datetime

class PredictionEngine:
    """Multi-algorithm prediction engine for 5D bet."""
    
    def __init__(self):
        self.weights = {
            'frequency': 40,
            'pattern': 30,
            'hotcold': 20,
            'markov': 10
        }
        self.risk_threshold = 70
    
    def predict(self, history):
        """Main prediction method - Ensemble of all algorithms."""
        if len(history) < 20:
            return self._error_prediction("Cần ít nhất 20 kỳ dữ liệu")
        
        # Run all algorithms
        freq_result = self._frequency_analysis(history)
        pattern_result = self._pattern_recognition(history)
        hotcold_result = self._hotcold_analysis(history)
        markov_result = self._markov_chain(history)
        
        # Ensemble voting
        all_votes = []
        avoid_votes = []
        
        # Frequency (40%)
        for num in freq_result['top_3']:
            all_votes.extend([num] * self.weights['frequency'])
        
        # Pattern (30%)
        for num in pattern_result['likely']:
            all_votes.extend([num] * self.weights['pattern'])
        for num in pattern_result.get('avoid', []):
            avoid_votes.append(num)
        
        # Hot/Cold (20%)
        for num in hotcold_result['hot'][:3]:
            all_votes.extend([num] * self.weights['hotcold'])
        for num in hotcold_result.get('due', []):
            all_votes.extend([num] * (self.weights['hotcold'] + 5))
        
        # Markov (10%)
        for num in markov_result['top_3']:
            all_votes.extend([num] * self.weights['markov'])
        
        # Vote counting
        vote_count = Counter(all_votes)
        avoid_set = set(avoid_votes)
        
        # Get top 3 (excluding avoid)
        final_3 = []
        for num, count in vote_count.most_common():
            if num not in final_3 and num not in avoid_set:
                final_3.append(num)
            if len(final_3) == 3:
                break
        
        # Fill if needed
        while len(final_3) < 3:
            for i in range(10):
                if str(i) not in final_3 and str(i) not in avoid_set:
                    final_3.append(str(i))
                    break
        
        # Support 4
        remaining = [n for n, c in vote_count.most_common(10) if n not in final_3 and n not in avoid_set]
        support_4 = remaining[:4]
        while len(support_4) < 4:
            for i in range(10):
                if str(i) not in final_3 and str(i) not in support_4:
                    support_4.append(str(i))
                    break
        
        # Confidence calculation
        if vote_count:
            top_vote = vote_count.most_common(1)[0][1]
            confidence = min(95, 60 + top_vote * 2)
        else:
            confidence = 50
        
        # Build logic
        logic_parts = []
        if freq_result['top_3']:
            logic_parts.append(f"Tần suất: {','.join(freq_result['top_3'])}")
        if pattern_result['detected']:
            logic_parts.append(f"{len(pattern_result['detected'])} pattern")
        if hotcold_result.get('due'):
            logic_parts.append(f"Đến kỳ: {','.join(hotcold_result['due'][:2])}")
        
        return {
            'main_3': final_3,
            'support_4': support_4,
            'confidence': confidence,
            'algorithm': 'Ensemble (5 algorithms)',
            'logic': ' | '.join(logic_parts) if logic_parts else 'Phân tích đa thuật toán',
            'avoid': list(avoid_set),
            'details': {
                'frequency': freq_result,
                'pattern': pattern_result,
                'hotcold': hotcold_result,
                'markov': markov_result,
                'votes': dict(vote_count.most_common(10))
            }
        }
    
    def calculate_risk(self, history):
        """Calculate risk score 0-100."""
        if len(history) < 20:
            return 0, "LOW", []
        
        recent = history[-50:] if len(history) >= 50 else history
        all_digits = ''.join(recent)
        freq = Counter(all_digits)
        reasons = []
        risk = 0
        
        # 1. Over-represented numbers
        total_slots = len(all_digits)
        if total_slots > 0:
            for num, count in freq.most_common(3):
                rate = count / total_slots
                if rate > 0.25:
                    risk += 20
                    reasons.append(f"Số '{num}' xuất hiện {rate*100:.0f}%")
        
        # 2. Abnormal streaks
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            max_streak = 1
            current = 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i-1]:
                    current += 1
                    max_streak = max(max_streak, current)
                else:
                    current = 1
            if max_streak >= 5:
                risk += 30
                reasons.append(f"Cầu bệt {max_streak} kỳ vị trí {pos}")
        
        # 3. Entropy
        if len(all_digits) > 0:
            entropy = -sum((c/len(all_digits)) * np.log2(c/len(all_digits)) 
                          for c in freq.values() if c > 0)
            if entropy < 2.8:
                risk += 25
                reasons.append(f"Entropy thấp ({entropy:.2f})")
        
        # 4. Stable sums
        totals = [sum(int(d) for d in n) for n in recent if len(n) == 5]
        if len(totals) > 10:
            std_dev = np.std(totals)
            if std_dev < 2.5:
                risk += 15
                reasons.append(f"Tổng quá ổn định (σ={std_dev:.2f})")
        
        risk = min(100, risk)
        
        if risk >= 70:
            level = "HIGH"
        elif risk >= 40:
            level = "MEDIUM"
        else:
            level = "LOW"
        
        return risk, level, reasons
    
    def detect_patterns(self, history):
        """Detect patterns in history."""
        recent = history[-30:] if len(history) >= 30 else history
        
        patterns = {
            'bet': [],
            'nhip2': [],
            'nhip3': [],
            'detected': [],
            'likely': [],
            'avoid': []
        }
        
        # Cầu bệt
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            for i in range(len(seq) - 2):
                if seq[i] == seq[i+1] == seq[i+2]:
                    digit = seq[i]
                    streak_len = 3
                    for j in range(i+3, len(seq)):
                        if seq[j] == digit:
                            streak_len += 1
                        else:
                            break
                    
                    if digit not in patterns['bet']:
                        patterns['bet'].append(digit)
                        patterns['detected'].append(f'Bệt {streak_len} kỳ vị {pos}: {digit}')
                        
                        if streak_len >= 4:
                            if digit not in patterns['avoid']:
                                patterns['avoid'].append(digit)
                        else:
                            if digit not in patterns['likely']:
                                patterns['likely'].append(digit)
        
        # Cầu nhịp 2
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            for i in range(len(seq) - 4):
                if seq[i] == seq[i+2] == seq[i+4] and seq[i] != seq[i+1]:
                    digit = seq[i]
                    if digit not in patterns['nhip2']:
                        patterns['nhip2'].append(digit)
                        patterns['detected'].append(f'Nhịp-2 vị {pos}: {digit}')
                        if digit not in patterns['likely']:
                            patterns['likely'].append(digit)
        
        return patterns
    
    def _frequency_analysis(self, history):
        """Algorithm 1: Frequency analysis with exponential weighting."""
        recent = history[-100:] if len(history) >= 100 else history
        
        weighted_freq = defaultdict(float)
        
        for idx, num in enumerate(recent):
            # Exponential decay: recent = 5.0, older = 1.0
            weight = 5.0 - 4.0 * (idx / max(len(recent), 1))
            for digit in num:
                if digit.isdigit():
                    weighted_freq[digit] += weight
        
        sorted_items = sorted(weighted_freq.items(), key=lambda x: x[1], reverse=True)
        top_3 = [str(x[0]) for x in sorted_items[:3]]
        
        while len(top_3) < 3:
            for i in range(10):
                if str(i) not in top_3:
                    top_3.append(str(i))
                    break
        
        return {
            'top_3': top_3,
            'scores': {k: round(v, 2) for k, v in sorted_items[:10]}
        }
    
    def _pattern_recognition(self, history):
        """Algorithm 2: Pattern recognition."""
        return self.detect_patterns(history)
    
    def _hotcold_analysis(self, history):
        """Algorithm 3: Hot/Cold analysis."""
        recent = history[-10:] if len(history) >= 10 else history
        older = history[-20:-10] if len(history) >= 20 else []
        
        recent_digits = Counter(''.join(recent))
        older_digits = Counter(''.join(older)) if older else Counter()
        
        hot = [str(x[0]) for x in recent_digits.most_common(5)]
        
        all_recent = ''.join(recent)
        cold = [str(i) for i in range(10) if str(i) not in all_recent]
        
        due = []
        for num in cold:
            if older_digits.get(num, 0) >= 4:
                due.append(num)
        
        return {
            'hot': hot,
            'cold': cold,
            'due': due
        }
    
    def _markov_chain(self, history):
        """Algorithm 4: Markov Chain prediction."""
        if len(history) < 30:
            return {'top_3': ['0', '1', '2'], 'transition_matrix': {}}
        
        # Build transition matrix (last digit → next digits)
        transition = defaultdict(lambda: defaultdict(int))
        
        for num in history[:-1]:
            if len(num) >= 5:
                last_digit = num[-1]
                next_num = history[history.index(num) + 1] if num in history else None
                if next_num:
                    for digit in next_num:
                        transition[last_digit][digit] += 1
        
        # Get most likely next digits
        all_next = defaultdict(int)
        for last_d, next_dict in transition.items():
            for next_d, count in next_dict.items():
                all_next[next_d] += count
        
        sorted_items = sorted(all_next.items(), key=lambda x: x[1], reverse=True)
        top_3 = [str(x[0]) for x in sorted_items[:3]]
        
        while len(top_3) < 3:
            for i in range(10):
                if str(i) not in top_3:
                    top_3.append(str(i))
                    break
        
        return {
            'top_3': top_3,
            'transition_matrix': dict(transition)
        }
    
    def _error_prediction(self, error_msg):
        """Return error prediction."""
        return {
            'main_3': ['?', '?', '?'],
            'support_4': ['0', '0', '0', '0'],
            'confidence': 0,
            'algorithm': 'Error',
            'logic': error_msg,
            'avoid': [],
            'details': {}
        }