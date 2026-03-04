# ==============================================================================
# TITAN v37.5 ULTRA - Multi-Layer AI Prediction Engine
# FIXED: All layer return formats, error handling, self-learning
# ==============================================================================

import numpy as np
from collections import Counter, defaultdict
import math
import random
from datetime import datetime

class PredictionEngine:
    """
    Multi-Layer AI Engine with:
    - Frequency Analysis (Exponential Weighted)
    - Markov Chain (Transition Probability)
    - Monte Carlo (Weighted Simulation)
    - Pattern Recognition (Advanced Rules)
    - Ensemble Voting (Adaptive Weights)
    """
    
    def __init__(self):
        # Algorithm weights (auto-adjusted by self-learning)
        self.weights = {
            'frequency': 30,      # Statistical frequency
            'markov': 25,          # Markov chain transitions
            'monte_carlo': 25,     # Probabilistic simulation
            'pattern': 20          # Rule-based patterns
        }
        
        # Learning parameters
        self.learning_rate = 0.15
        self.win_history = []  # Track last 30 predictions
        self.max_history = 30
        
        # Pattern memory for advanced learning
        self.pattern_memory = defaultdict(lambda: {'wins': 0, 'losses': 0})
        
        # Risk configuration
        self.risk_config = {
            'entropy_min': 2.8,
            'entropy_max': 3.3,
            'max_streak': 5,
            'min_std_sum': 2.0
        }
    
    def predict(self, history):
        """
        Main prediction: Run all AI layers and ensemble results.
        FIXED: Consistent return format for all layers.
        """
        # Validate input
        if not history or len(history) < 10:
            return self._fallback_prediction("⚠️ Cần tối thiểu 10 kỳ dữ liệu")
        
        # Clean history - ensure all are 5-digit strings
        clean_history = [str(h).zfill(5)[-5:] for h in history if str(h).isdigit() and len(str(h)) >= 3]
        if len(clean_history) < 10:
            return self._fallback_prediction("⚠️ Dữ liệu không hợp lệ")
        
        try:
            # Run each AI layer (all return dict with 'top_3', 'score', 'method')
            freq_res = self._layer_frequency(clean_history)
            markov_res = self._layer_markov(clean_history)
            monte_res = self._layer_monte_carlo(clean_history)
            pattern_res = self._layer_pattern(clean_history)
            
            # Ensemble voting
            all_layers = {
                'frequency': freq_res,
                'markov': markov_res,
                'monte_carlo': monte_res,
                'pattern': pattern_res
            }
            
            ensemble = self._ensemble_vote(all_layers)
            risk_metrics = self.calculate_risk(clean_history)
            
            # Calculate confidence based on layer agreement
            confidence = self._calculate_confidence(all_layers, ensemble)
            
            # Build logic explanation
            logic = self._build_logic(all_layers, ensemble)
            
            return {
                'main_3': ensemble['main_3'],
                'support_4': ensemble['support_4'],
                'confidence': confidence,
                'logic': logic,
                'risk_metrics': risk_metrics,
                'layer_scores': {k: v.get('score', 0) for k, v in all_layers.items()},
                'algorithm': 'Multi-Layer AI v37.5'
            }
            
        except Exception as e:
            return self._fallback_prediction(f"⚠️ Lỗi AI: {str(e)[:50]}")
    
    def calculate_risk(self, history):
        """
        Multi-factor risk assessment.
        FIXED: Proper entropy calculation, multiple risk factors.
        """
        if not history or len(history) < 10:
            return {'score': 0, 'level': 'LOW', 'reasons': ['Chưa đủ dữ liệu']}
        
        try:
            recent = history[-30:] if len(history) >= 30 else history
            all_digits = "".join(recent)
            
            if not all_digits:
                return {'score': 50, 'level': 'MEDIUM', 'reasons': ['Dữ liệu trống']}
            
            counts = Counter(all_digits)
            total = len(all_digits)
            
            # Factor 1: Entropy (randomness measure)
            entropy = 0
            for c in counts.values():
                if c > 0:
                    p = c / total
                    entropy -= p * math.log2(p)
            
            # Factor 2: Over-represented numbers
            over_represented = False
            for num, count in counts.most_common(2):
                if count / total > 0.25:  # >25% appearance
                    over_represented = True
                    break
            
            # Factor 3: Streak detection
            max_streak = 0
            for pos in range(5):
                seq = [n[pos] if len(n) > pos else '0' for n in recent]
                streak = 1
                for i in range(1, len(seq)):
                    if seq[i] == seq[i-1]:
                        streak += 1
                        max_streak = max(max_streak, streak)
                    else:
                        streak = 1
            
            # Factor 4: Sum stability
            totals = [sum(int(d) for d in n) for n in recent if len(n) == 5 and n.isdigit()]
            sum_std = np.std(totals) if len(totals) > 2 else 10
            
            # Calculate risk score
            score = 0
            reasons = []
            
            # Entropy risk
            if entropy < self.risk_config['entropy_min']:
                score += 25
                reasons.append(f"Entropy thấp ({entropy:.2f}) - Kết quả không ngẫu nhiên")
            elif entropy > self.risk_config['entropy_max']:
                score += 15
                reasons.append(f"Entropy cao ({entropy:.2f}) - Biến động mạnh")
            
            # Over-represented risk
            if over_represented:
                score += 20
                reasons.append("Có số xuất hiện quá nhiều (bất thường)")
            
            # Streak risk
            if max_streak >= self.risk_config['max_streak']:
                score += 30
                reasons.append(f"Cầu bệt {max_streak} kỳ - Nhà cái có thể điều khiển")
            
            # Sum stability risk
            if sum_std < self.risk_config['min_std_sum']:
                score += 15
                reasons.append(f"Tổng số quá ổn định (σ={sum_std:.2f})")
            
            score = min(100, score)
            level = "HIGH" if score >= 50 else "MEDIUM" if score >= 25 else "LOW"
            
            return {
                'score': score,
                'level': level,
                'reasons': reasons if reasons else ['Nhịp số tự nhiên']
            }
            
        except Exception as e:
            return {'score': 50, 'level': 'MEDIUM', 'reasons': [f'Lỗi tính risk: {str(e)[:30]}']}
    
    def _layer_frequency(self, history):
        """
        Layer 1: Exponential-weighted frequency analysis.
        FIXED: Returns dict with consistent format.
        """
        try:
            # Use last 60 periods with exponential decay
            recent = history[-60:] if len(history) >= 60 else history
            
            weighted_freq = defaultdict(float)
            
            for idx, num in enumerate(recent):
                # Exponential decay: recent = 4.0, older = 1.0
                weight = 4.0 - 3.0 * (idx / max(len(recent), 1))
                for digit in num:
                    if digit.isdigit():
                        weighted_freq[digit] += weight
            
            # Get top 3
            sorted_items = sorted(weighted_freq.items(), key=lambda x: x[1], reverse=True)
            top_3 = [str(x[0]) for x in sorted_items[:3]]
            
            # Fill if needed
            while len(top_3) < 3:
                for i in range(10):
                    if str(i) not in top_3:
                        top_3.append(str(i))
                        break
            
            score = sum(weighted_freq.get(n, 0) for n in top_3) / 3 if top_3 else 0
            
            return {
                'top_3': top_3,
                'score': min(100, score),
                'method': 'Exponential Frequency',
                'details': dict(sorted_items[:10])
            }
            
        except:
            return {'top_3': ['1', '5', '9'], 'score': 30, 'method': 'Frequency (error)'}
    
    def _layer_markov(self, history):
        """
        Layer 2: Markov Chain with position-aware transitions.
        FIXED: Proper transition matrix, position tracking.
        """
        try:
            if len(history) < 15:
                return {'top_3': ['2', '5', '8'], 'score': 25, 'method': 'Markov (insufficient)'}
            
            # Build position-aware transition matrix
            # Key: (position, last_digit) → Counter of next_digits
            transitions = defaultdict(lambda: defaultdict(Counter))
            
            for i in range(len(history) - 1):
                curr = history[i]
                next_num = history[i + 1]
                
                if len(curr) >= 5 and len(next_num) >= 5:
                    for pos in range(5):
                        last_d = curr[pos]
                        next_d = next_num[pos]
                        transitions[pos][last_d][next_d] += 1
            
            # Predict based on last number's digits
            last_num = history[-1] if len(history[-1]) == 5 else '00000'
            
            # Aggregate predictions across positions
            next_prob = Counter()
            for pos in range(5):
                last_d = last_num[pos]
                if last_d in transitions[pos]:
                    total = sum(transitions[pos][last_d].values())
                    if total > 0:
                        for next_d, count in transitions[pos][last_d].items():
                            # Weight by position importance (middle positions more important)
                            pos_weight = 1.0 + 0.3 * (2 - abs(pos - 2))
                            next_prob[next_d] += (count / total) * pos_weight
            
            # Get top 3
            sorted_items = sorted(next_prob.items(), key=lambda x: x[1], reverse=True)
            top_3 = [str(x[0]) for x in sorted_items[:3]]
            
            while len(top_3) < 3:
                for i in range(10):
                    if str(i) not in top_3:
                        top_3.append(str(i))
                        break
            
            score = sum(next_prob.get(n, 0) for n in top_3) / 3 if top_3 else 0
            
            return {
                'top_3': top_3,
                'score': min(100, score * 10),
                'method': 'Position-Aware Markov',
                'transitions': {pos: dict(transitions[pos]) for pos in range(5)}
            }
            
        except:
            return {'top_3': ['3', '6', '9'], 'score': 20, 'method': 'Markov (error)'}
    
    def _layer_monte_carlo(self, history):
        """
        Layer 3: Weighted Monte Carlo Simulation.
        FIXED: Proper weighted sampling, noise filtering.
        """
        try:
            if len(history) < 15:
                return {'top_3': ['0', '4', '8'], 'score': 25, 'method': 'Monte Carlo (insufficient)'}
            
            # Build weighted pool from recent history
            recent = history[-40:] if len(history) >= 40 else history
            
            # Create weighted digit pool (recent digits weighted higher)
            weighted_pool = []
            for idx, num in enumerate(recent):
                weight = int(5 - idx / 8)  # Weight: 5 → 1
                weight = max(1, weight)
                for digit in num:
                    if digit.isdigit():
                        weighted_pool.extend([digit] * weight)
            
            if not weighted_pool:
                return {'top_3': ['1', '2', '3'], 'score': 10, 'method': 'Monte Carlo (empty)'}
            
            # Run simulations with noise filtering
            sim_results = Counter()
            num_simulations = min(5000, len(weighted_pool) * 10)
            
            for _ in range(num_simulations):
                # Weighted random sample of 3 digits
                sample = random.choices(weighted_pool, k=3)
                
                # Apply noise filter: prefer diverse samples
                if len(set(sample)) >= 2:  # At least 2 unique digits
                    for n in sample:
                        sim_results[n] += 1
            
            # Get top 3
            sorted_items = sorted(sim_results.items(), key=lambda x: x[1], reverse=True)
            top_3 = [str(x[0]) for x in sorted_items[:3]]
            
            while len(top_3) < 3:
                for i in range(10):
                    if str(i) not in top_3:
                        top_3.append(str(i))
                        break
            
            total_votes = sum(sim_results.values())
            score = sum(sim_results.get(n, 0) for n in top_3) / total_votes * 100 if total_votes > 0 else 0
            
            return {
                'top_3': top_3,
                'score': min(100, score),
                'method': 'Weighted Monte Carlo',
                'simulations': num_simulations
            }
            
        except:
            return {'top_3': ['2', '4', '6'], 'score': 15, 'method': 'Monte Carlo (error)'}
    
    def _layer_pattern(self, history):
        """
        Layer 4: Advanced Pattern Recognition.
        FIXED: Detect streaks, rhythms, reversals, triangles.
        """
        try:
            if len(history) < 15:
                return {'top_3': ['1', '3', '7'], 'score': 20, 'method': 'Pattern (insufficient)'}
            
            recent = history[-30:] if len(history) >= 30 else history
            patterns = {'likely': [], 'avoid': [], 'detected': []}
            
            # 1. Streak detection (cầu bệt)
            for pos in range(5):
                seq = [n[pos] if len(n) > pos else '0' for n in recent]
                i = 0
                while i < len(seq) - 2:
                    if seq[i] == seq[i+1] == seq[i+2]:
                        digit = seq[i]
                        streak_len = 3
                        j = i + 3
                        while j < len(seq) and seq[j] == digit:
                            streak_len += 1
                            j += 1
                        
                        if streak_len >= 4:
                            # Long streak likely to break → avoid
                            if digit not in patterns['avoid']:
                                patterns['avoid'].append(digit)
                                patterns['detected'].append(f'Bệt {streak_len} kỳ vị {pos}: {digit} (TRÁNH)')
                        elif streak_len == 3:
                            # Medium streak may continue → likely
                            if digit not in patterns['likely']:
                                patterns['likely'].append(digit)
                                patterns['detected'].append(f'Bệt 3 kỳ vị {pos}: {digit}')
                        i = j
                    else:
                        i += 1
            
            # 2. Rhythm-2 detection (X _ X _ X)
            for pos in range(5):
                seq = [n[pos] if len(n) > pos else '0' for n in recent]
                for i in range(len(seq) - 4):
                    if seq[i] == seq[i+2] == seq[i+4] and seq[i] != seq[i+1]:
                        digit = seq[i]
                        if digit not in patterns['likely'] and digit not in patterns['avoid']:
                            patterns['likely'].append(digit)
                            patterns['detected'].append(f'Nhịp-2 vị {pos}: {digit}')
            
            # 3. Rhythm-3 detection (X _ _ X _ _ X)
            for pos in range(5):
                seq = [n[pos] if len(n) > pos else '0' for n in recent]
                for i in range(len(seq) - 6):
                    if seq[i] == seq[i+3] == seq[i+6]:
                        digit = seq[i]
                        if digit not in patterns['likely'] and digit not in patterns['avoid']:
                            patterns['likely'].append(digit)
                            patterns['detected'].append(f'Nhịp-3 vị {pos}: {digit}')
            
            # 4. Reversal detection (AB → BA)
            for i in range(len(recent) - 1):
                curr, next_num = recent[i], recent[i+1]
                if len(curr) >= 2 and len(next_num) >= 2:
                    if curr[0:2] == next_num[1::-1]:  # AB → BA
                        for d in curr[0:2]:
                            if d not in patterns['likely'] and d not in patterns['avoid']:
                                patterns['likely'].append(d)
                                patterns['detected'].append(f'Đảo cặp: {curr[0:2]} → {next_num[0:2]}')
            
            # Build candidates from patterns
            candidates = Counter()
            for digit in patterns['likely']:
                candidates[digit] += 3
            for digit in patterns['avoid']:
                candidates[digit] -= 2  # Penalize avoided numbers
            
            # Add fallback if no patterns found
            if not candidates:
                # Use simple frequency as fallback
                all_digits = ''.join(recent)
                freq = Counter(all_digits)
                for num, count in freq.most_common(3):
                    candidates[num] += 2
            
            top_3 = [str(x[0]) for x in candidates.most_common(3)]
            
            while len(top_3) < 3:
                for i in range(10):
                    if str(i) not in top_3:
                        top_3.append(str(i))
                        break
            
            score = len(patterns['detected']) * 5 + sum(candidates.get(n, 0) for n in top_3)
            
            return {
                'top_3': top_3,
                'score': min(100, score),
                'method': 'Advanced Pattern AI',
                'patterns_found': patterns['detected'],
                'likely': patterns['likely'],
                'avoid': patterns['avoid']
            }
            
        except:
            return {'top_3': ['0', '5', '9'], 'score': 15, 'method': 'Pattern (error)'}
    
    def _ensemble_vote(self, layers):
        """
        Ensemble voting with adaptive weights and avoid filtering.
        FIXED: Proper dict handling, avoid number filtering.
        """
        votes = Counter()
        avoid_votes = []
        
        for layer_name, layer_result in layers.items():
            if not isinstance(layer_result, dict):
                continue
                
            weight = self.weights.get(layer_name, 10)
            top_3 = layer_result.get('top_3', [])
            
            # Add votes for top candidates
            for num in top_3:
                if isinstance(num, str) and num.isdigit():
                    votes[num] += weight
            
            # Collect avoid numbers
            avoid = layer_result.get('avoid', [])
            avoid_votes.extend(avoid)
        
        # Get top candidates excluding avoided numbers
        avoid_set = set(avoid_votes)
        
        # Get main_3 (top 3 excluding avoid)
        main_3 = []
        for num, count in votes.most_common():
            if num not in main_3 and num not in avoid_set:
                main_3.append(num)
            if len(main_3) == 3:
                break
        
        # Fill if needed
        while len(main_3) < 3:
            for i in range(10):
                if str(i) not in main_3 and str(i) not in avoid_set:
                    main_3.append(str(i))
                    break
        
        # Get support_4 (next best excluding main_3 and avoid)
        remaining = [n for n, c in votes.most_common(10) if n not in main_3 and n not in avoid_set]
        support_4 = remaining[:4]
        
        while len(support_4) < 4:
            for i in range(10):
                if str(i) not in main_3 and str(i) not in support_4 and str(i) not in avoid_set:
                    support_4.append(str(i))
                    break
        
        return {
            'main_3': main_3,
            'support_4': support_4,
            'avoid': list(avoid_set),
            'vote_counts': dict(votes.most_common(10))
        }
    
    def _calculate_confidence(self, layers, ensemble):
        """Calculate confidence based on layer agreement."""
        try:
            # Check agreement among layers
            all_picks = []
            for layer_result in layers.values():
                if isinstance(layer_result, dict):
                    all_picks.extend(layer_result.get('top_3', []))
            
            # Count how many times ensemble picks appear in layer picks
            agreement = sum(1 for num in ensemble['main_3'] if all_picks.count(num) >= 2)
            
            # Base confidence + agreement bonus
            base_conf = 55
            agreement_bonus = agreement * 12
            confidence = min(96, base_conf + agreement_bonus)
            
            # Reduce if avoid numbers are in top votes
            if ensemble.get('avoid'):
                confidence = max(40, confidence - 15)
            
            return confidence
            
        except:
            return 50
    
    def _build_logic(self, layers, ensemble):
        """Build human-readable explanation."""
        try:
            parts = []
            
            # Frequency info
            if layers.get('frequency', {}).get('details'):
                top_freq = list(layers['frequency']['details'].items())[:2]
                if top_freq:
                    parts.append(f"Tần suất: {','.join([f[0] for f in top_freq])}")
            
            # Pattern info
            if layers.get('pattern', {}).get('patterns_found'):
                count = len(layers['pattern']['patterns_found'])
                parts.append(f"{count} pattern")
            
            # Markov info
            if layers.get('markov', {}).get('score', 0) > 40:
                parts.append("Markov mạnh")
            
            # Avoid warning
            if ensemble.get('avoid'):
                parts.append(f"⚠️ Tránh: {','.join(ensemble['avoid'][:2])}")
            
            return ' | '.join(parts) if parts else 'Phân tích đa tầng AI'
            
        except:
            return 'Phân tích AI'
    
    def get_ai_status(self):
        """Return AI engine status for display."""
        try:
            recent_win_rate = 0
            if self.win_history:
                recent = self.win_history[-10:]
                recent_win_rate = sum(recent) / len(recent) * 100
            
            return {
                'weights': self.weights,
                'recent_win_rate': round(recent_win_rate, 1),
                'predictions_tracked': len(self.win_history),
                'pattern_memory_size': sum(len(v) for v in self.pattern_memory.values())
            }
        except:
            return {'weights': self.weights, 'recent_win_rate': 0, 'predictions_tracked': 0}
    
    def update_weights(self, won: bool, winning_method: str = None):
        """
        SELF-LEARNING: Adjust algorithm weights based on outcome.
        FIXED: Proper weight normalization, pattern memory update.
        """
        try:
            # Record outcome
            self.win_history.append(1 if won else 0)
            if len(self.win_history) > self.max_history:
                self.win_history.pop(0)
            
            # Calculate adjustment
            adjustment = self.learning_rate if won else -self.learning_rate * 0.5
            
            # Boost the method that contributed (if known)
            if winning_method and winning_method in self.weights:
                self.weights[winning_method] = min(50, self.weights[winning_method] + adjustment * 10)
            
            # Normalize weights to sum to 100
            total = sum(self.weights.values())
            if total > 0:
                for key in self.weights:
                    self.weights[key] = round(self.weights[key] * 100 / total)
            
            # Update pattern memory
            if winning_method:
                self.pattern_memory[winning_method]['wins' if won else 'losses'] += 1
            
        except:
            pass  # Silent fail to avoid crashing app
    
    def _fallback_prediction(self, error_msg):
        """Return safe fallback prediction."""
        return {
            'main_3': ['?', '?', '?'],
            'support_4': ['0', '0', '0', '0'],
            'confidence': 0,
            'logic': error_msg,
            'risk_metrics': {'score': 50, 'level': 'MEDIUM', 'reasons': ['Fallback mode']},
            'layer_scores': {},
            'algorithm': 'Fallback'
        }