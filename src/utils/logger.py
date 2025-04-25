import os
import sys
from enum import Enum
from datetime import datetime
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


class LogLevel(Enum):
    """Enum for log levels with numeric values for comparison."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class Logger:
    """
    Centralized logging system with support for different verbosity levels.
    """
    def __init__(self, name="AI Hedge Fund", level=LogLevel.INFO, log_to_file=False, log_file=None):
        self.name = name
        self.level = level
        self.log_to_file = log_to_file
        self.log_file = log_file or f"logs/{name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Create logs directory if it doesn't exist and log_to_file is True
        if self.log_to_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, 'w') as f:
                f.write(f"=== {self.name} Log Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    def set_level(self, level):
        """Set the logging level."""
        if isinstance(level, str):
            level = getattr(LogLevel, level.upper(), LogLevel.INFO)
        self.level = level
        
    def debug(self, message, module=None, ticker=None):
        """Log debug message if level is DEBUG or lower."""
        if self.level.value <= LogLevel.DEBUG.value:
            self._log(message, LogLevel.DEBUG, module, ticker)
    
    def info(self, message, module=None, ticker=None):
        """Log info message if level is INFO or lower."""
        if self.level.value <= LogLevel.INFO.value:
            self._log(message, LogLevel.INFO, module, ticker)
    
    def warning(self, message, module=None, ticker=None):
        """Log warning message if level is WARNING or lower."""
        if self.level.value <= LogLevel.WARNING.value:
            self._log(message, LogLevel.WARNING, module, ticker)
    
    def error(self, message, module=None, ticker=None):
        """Log error message if level is ERROR or lower."""
        if self.level.value <= LogLevel.ERROR.value:
            self._log(message, LogLevel.ERROR, module, ticker)
    
    def critical(self, message, module=None, ticker=None):
        """Log critical message if level is CRITICAL or lower."""
        if self.level.value <= LogLevel.CRITICAL.value:
            self._log(message, LogLevel.CRITICAL, module, ticker)
    
    def _log(self, message, level, module=None, ticker=None):
        """Internal method to format and output log messages."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format with colors for console output
        level_color = {
            LogLevel.DEBUG: Fore.CYAN,
            LogLevel.INFO: Fore.GREEN,
            LogLevel.WARNING: Fore.YELLOW,
            LogLevel.ERROR: Fore.RED,
            LogLevel.CRITICAL: Fore.MAGENTA + Style.BRIGHT
        }
        
        level_text = f"{level_color[level]}{level.name}{Style.RESET_ALL}"
        timestamp_text = f"{Fore.WHITE}{timestamp}{Style.RESET_ALL}"
        
        # Add optional module and ticker
        context = ""
        if module:
            context += f"{Fore.BLUE}{module}{Style.RESET_ALL}"
        if ticker:
            context += f" [{Fore.CYAN}{ticker}{Style.RESET_ALL}]"
        
        if context:
            context = f" {context}"
        
        console_message = f"{timestamp_text} [{level_text}]{context}: {message}"
        
        # Plain text version for file logging
        file_message = f"{timestamp} [{level.name}]"
        if module:
            file_message += f" {module}"
        if ticker:
            file_message += f" [{ticker}]"
        file_message += f": {message}"
        
        # Output to console
        print(console_message)
        
        # Output to file if enabled
        if self.log_to_file:
            with open(self.log_file, 'a') as f:
                f.write(file_message + "\n")


# Create a global logger instance
logger = Logger()


def setup_logger(debug_mode=False, log_to_file=False, log_file=None):
    """Configure the global logger based on command line arguments."""
    level = LogLevel.DEBUG if debug_mode else LogLevel.INFO
    logger.level = level
    logger.log_to_file = log_to_file
    if log_file:
        logger.log_file = log_file