# ==============================================================================
# TITAN AI v5.0 - Database Management
# ==============================================================================

import re
from typing import List, Dict, Optional
from datetime import datetime

class DatabaseManager:
    """Lottery data management."""
    
    def __init__(self):
        self.data = []
        self.test_log = []
    
    def clean_data(self, raw_text):
        """Clean and extract 5-digit numbers."""
        if not raw_text or not isinstance(raw_text, str):
            return []
        
        numbers = re.findall(r'\d{5}', raw_text)
        validated = []
        for num in numbers:
            if len(num) == 5 and num.isdigit():
                validated.append(num)
        
        return validated
    
    def add_numbers(self, new_numbers, existing_data=None):
        """Add new numbers with deduplication."""
        if existing_data is None:
            existing_data = self.data
        
        existing_set = set(existing_data)
        unique_new = []
        
        for num in new_numbers:
            if isinstance(num, str) and len(num) == 5 and num.isdigit():
                if num not in existing_set:
                    unique_new.append(num)
                    existing_set.add(num)
        
        updated_data = unique_new + existing_data
        
        if len(updated_data) > 500:
            updated_data = updated_data[:500]
        
        self.data = updated_data
        return updated_data, len(unique_new)
    
    def record_test(self, prediction, actual, won, confidence, house_risk=0):
        """Record test prediction."""
        try:
            if not isinstance(prediction, list) or len(prediction) != 3:
                return False
            
            if not isinstance(actual, str) or len(actual) != 5:
                return False
            
            self.test_log.append({
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction,
                'actual': actual,
                'won': won,
                'confidence': confidence,
                'house_risk': house_risk
            })
            
            if len(self.test_log) > 100:
                self.test_log = self.test_log[-100:]
            
            return True
        except:
            return False
    
    def get_accuracy_stats(self):
        """Calculate accuracy statistics."""
        if not self.test_log:
            return {
                'total': 0,
                'wins': 0,
                'win_rate': 0.0,
                'avg_confidence': 0.0,
                'by_confidence': {}
            }
        
        total = len(self.test_log)
        wins = sum(1 for t in self.test_log if t.get('won', False))
        win_rate = (wins / total * 100) if total > 0 else 0.0
        avg_conf = sum(t.get('confidence', 0) for t in self.test_log) / total
        
        by_confidence = {}
        for bracket in ['50-69', '70-84', '85+']:
            if bracket == '50-69':
                subset = [t for t in self.test_log if 50 <= t.get('confidence', 0) < 70]
            elif bracket == '70-84':
                subset = [t for t in self.test_log if 70 <= t.get('confidence', 0) < 85]
            else:
                subset = [t for t in self.test_log if t.get('confidence', 0) >= 85]
            
            if subset:
                w = sum(1 for t in subset if t.get('won', False))
                by_confidence[bracket] = {
                    'count': len(subset),
                    'win_rate': round(w / len(subset) * 100, 1)
                }
        
        return {
            'total': total,
            'wins': wins,
            'win_rate': round(win_rate, 1),
            'avg_confidence': round(avg_conf, 1),
            'by_confidence': by_confidence
        }
    
    def clear(self):
        """Clear all data."""
        self.data = []
        self.test_log = []
    
    def get_statistics(self):
        """Get database statistics."""
        return {
            'total_numbers': len(self.data),
            'total_tests': len(self.test_log)
        }