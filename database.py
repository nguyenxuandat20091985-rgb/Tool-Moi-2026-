# ==============================================================================
# TITAN AI v5.0 - Database Management
# ==============================================================================

import re
from typing import List, Dict, Tuple
from datetime import datetime

class DatabaseManager:
    """Manage lottery data storage and retrieval."""
    
    def __init__(self):
        self.data = []
        self.test_log = []
    
    def clean_data(self, raw_text: str) -> List[str]:
        """
        Clean and extract 5-digit numbers from raw text.
        
        Args:
            raw_text: Raw input text
            
        Returns:
            List of cleaned 5-digit numbers
        """
        if not raw_text or not raw_text.strip():
            return []
        
        # Extract 5-digit numbers
        numbers = re.findall(r'\d{5}', raw_text)
        return numbers
    
    def add_numbers(self, new_numbers: List[str], existing_data: List[str] = None) -> Tuple[List[str], int]:
        """
        Add new numbers to database with deduplication.
        
        Args:
            new_numbers: List of new numbers to add
            existing_data: Existing database (optional)
            
        Returns:
            Tuple of (updated_data, count_added)
        """
        if existing_data is None:
            existing_data = self.data
        
        existing_set = set(existing_data)
        unique_new = []
        
        for num in new_numbers:
            if num not in existing_set:
                unique_new.append(num)
                existing_set.add(num)
        
        # Add new numbers to front (newest first)
        updated_data = unique_new + existing_data
        
        # Limit size
        if len(updated_data) > 500:
            updated_data = updated_data[:500]
        
        return updated_data, len(unique_new)
    
    def record_test(self, prediction: Dict, actual: str, won: bool, 
                   confidence: int, house_risk: int = 0) -> None:
        """Record a test prediction."""
        self.test_log.append({
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'actual': actual,
            'won': won,
            'confidence': confidence,
            'house_risk': house_risk
        })
        
        # Keep last 100 tests
        if len(self.test_log) > 100:
            self.test_log = self.test_log[-100:]
    
    def get_accuracy_stats(self) -> Dict:
        """Calculate accuracy statistics."""
        if not self.test_log:
            return {
                'total': 0,
                'wins': 0,
                'win_rate': 0,
                'avg_confidence': 0,
                'by_confidence': {}
            }
        
        total = len(self.test_log)
        wins = sum(1 for t in self.test_log if t['won'])
        win_rate = wins / total * 100 if total > 0 else 0
        avg_conf = sum(t['confidence'] for t in self.test_log) / total
        
        # By confidence bracket
        by_confidence = {}
        for bracket in ['50-69', '70-84', '85+']:
            if bracket == '50-69':
                subset = [t for t in self.test_log if 50 <= t['confidence'] < 70]
            elif bracket == '70-84':
                subset = [t for t in self.test_log if 70 <= t['confidence'] < 85]
            else:
                subset = [t for t in self.test_log if t['confidence'] >= 85]
            
            if subset:
                w = sum(1 for t in subset if t['won'])
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
    
    def clear(self) -> None:
        """Clear all data."""
        self.data = []
        self.test_log = []
    
    def export_data(self) -> Dict:
        """Export all data."""
        return {
            'data': self.data,
            'test_log': self.test_log,
            'exported_at': datetime.now().isoformat()
        }
    
    def import_data(self, data_dict: Dict) -> bool:
        """Import data from dictionary."""
        try:
            if 'data' in data_dict:
                self.data = data_dict['data']
            if 'test_log' in data_dict:
                self.test_log = data_dict['test_log']
            return True
        except Exception:
            return False