# ==============================================================================
# TITAN v35.0 - Database Management
# Data cleaning and storage
# ==============================================================================

import re
from datetime import datetime

class DatabaseManager:
    """Manage lottery database."""
    
    def __init__(self):
        self.max_records = 3000
    
    def clean_data(self, raw_text):
        """Clean and extract 5-digit numbers from raw text."""
        if not raw_text or not raw_text.strip():
            return []
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', raw_text.strip())
        lines = normalized.split('\n')
        
        numbers = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove all spaces within line (handle "41 720" → "41720")
            line_no_spaces = re.sub(r'\s', '', line)
            
            # Find all 5-digit sequences
            matches = re.findall(r'\d{5}', line_no_spaces)
            numbers.extend(matches)
        
        return numbers
    
    def add_numbers(self, new_numbers, existing_db):
        """Add new numbers to database with deduplication."""
        if not new_numbers:
            return 0
        
        db_set = set(existing_db)
        added = 0
        
        for num in new_numbers:
            if len(num) == 5 and num not in db_set:
                existing_db.insert(0, num)  # Add to front (newest first)
                db_set.add(num)
                added += 1
        
        # Limit database size
        if len(existing_db) > self.max_records:
            existing_db[:] = existing_db[:self.max_records]
        
        return added
    
    def export_data(self, lottery_db, predictions_log, bankroll):
        """Export all data as dictionary."""
        return {
            'lottery_db': lottery_db,
            'predictions_log': predictions_log,
            'bankroll': bankroll,
            'exported_at': datetime.now().isoformat(),
            'version': '35.0'
        }
    
    def import_data(self, data, lottery_db, predictions_log, bankroll):
        """Import data from dictionary."""
        try:
            if 'lottery_db' in data:
                lottery_db[:] = data['lottery_db'][:self.max_records]
            
            if 'predictions_log' in data:
                predictions_log[:] = data['predictions_log'][-200:]
            
            if 'bankroll' in data:
                bankroll.update(data['bankroll'])
            
            return True
        except Exception as e:
            return False