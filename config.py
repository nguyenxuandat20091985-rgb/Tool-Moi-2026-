# ==============================================================================
# TITAN v35.0 - Configuration
# ==============================================================================

class Config:
    """System configuration."""
    
    # App settings
    APP_NAME = "TITAN v35.0 PRO MAX"
    APP_VERSION = "35.0"
    
    # Database
    MAX_RECORDS = 3000
    MAX_PREDICTIONS = 200
    
    # Prediction
    MIN_HISTORY = 20
    RECENT_WINDOW = 50
    FREQUENCY_WINDOW = 100
    
    # Risk
    RISK_THRESHOLD = 70
    MAX_STREAK = 5
    
    # Bankroll
    DEFAULT_BANKROLL = 1000000
    DEFAULT_BET = 10000
    MAX_BET_PERCENT = 5
    
    # Auto-refresh
    REFRESH_INTERVAL = 60  # seconds
    
    # API (if using Gemini)
    GEMINI_LIMIT = 15  # requests per day