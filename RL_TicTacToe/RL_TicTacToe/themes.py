def get_theme(theme_name="dark_alliance"):
    themes = {
        "dark_alliance": {
            "bg_primary": "#2c3e50",
            "bg_secondary": "#34495e",
            "bg_tertiary": "#233140",
            "bg_interactive": "#1c2833",
            "text_primary": "#ecf0f1",
            "text_secondary": "#bdc3c7",
            "accent_primary": "#3498db",
            "accent_primary_hover": "#5dade2",
            "accent_primary_pressed": "#2980b9",
            "accent_positive": "#27ae60",
            "accent_warning": "#f1c40f",
            "accent_danger": "#e74c3c",
            "disabled_bg": "#566573",
            "disabled_fg": "#99a3a4",
            "shadow_color": "#1a2530"
        },
        "light_rebellion": {
            "bg_primary": "#ecf0f1",
            "bg_secondary": "#bdc3c7",
            "bg_tertiary": "#e1e5e7",
            "bg_interactive": "#ffffff",
            "text_primary": "#2c3e50",
            "text_secondary": "#566573",
            "accent_primary": "#2980b9",
            "accent_primary_hover": "#3498db",
            "accent_primary_pressed": "#1f618d",
            "accent_positive": "#229954",
            "accent_warning": "#f39c12",
            "accent_danger": "#c0392b",
            "disabled_bg": "#bdc3c7",
            "disabled_fg": "#7f8c8d",
            "shadow_color": "#99a3a4"
        }
    }
    return themes[theme_name]

def get_stylesheet(theme_name):
    theme = get_theme(theme_name)
    return f"""
        QMainWindow, QWidget {{
            background-color: {theme['bg_primary']};
            color: {theme['text_primary']};
            font-family: Segoe UI;
        }}
        QTabWidget::pane {{ border: none; }}
        QTabBar::tab {{
            background: {theme['bg_secondary']};
            color: {theme['text_secondary']};
            padding: 10px 20px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            font-weight: bold;
        }}
        QTabBar::tab:selected {{ background: {theme['bg_tertiary']}; color: {theme['text_primary']}; }}
        QTabBar::tab:hover {{ background: {theme['bg_tertiary']}; }}
        
        QGroupBox {{
            font-size: 14px;
            font-weight: bold;
            color: {theme['text_secondary']};
            border: 1px solid {theme['bg_secondary']};
            border-radius: 5px;
            margin-top: 1ex;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }}
        
        QPushButton {{
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {theme['accent_primary_hover']}, stop:1 {theme['accent_primary']});
            border: 1px solid {theme['accent_primary_pressed']};
            padding: 8px;
            border-radius: 4px;
            font-size: 13px;
            color: white;
            font-weight: bold;
        }}
        QPushButton:hover {{ background-color: {theme['accent_primary_hover']}; }}
        QPushButton:pressed {{ background-color: {theme['accent_primary_pressed']}; }}
        QPushButton:disabled {{
            background-color: {theme['disabled_bg']};
            color: {theme['disabled_fg']};
            border: 1px solid {theme['disabled_fg']};
        }}
        
        QLabel {{ font-size: 13px; color: {theme['text_secondary']}; }}
        QLabel#header {{ font-size: 13px; font-weight: bold; color: {theme['text_primary']}; }}
        
        QSpinBox {{
            padding: 4px;
            border: 1px solid {theme['bg_secondary']};
            border-radius: 4px;
            background-color: {theme['bg_interactive']};
        }}
        
        QProgressBar {{
            border: 1px solid {theme['bg_secondary']};
            border-radius: 4px;
            text-align: center;
            color: white;
            font-weight: bold;
        }}
        QProgressBar::chunk {{ background-color: {theme['accent_positive']}; border-radius: 3px; }}
        
        QTextEdit {{
            background-color: {theme['bg_interactive']};
            border: 1px solid {theme['bg_secondary']};
            border-radius: 4px;
            color: {theme['text_secondary']};
            font-family: Consolas, monospace;
        }}
        
        QSlider::groove:horizontal {{
            border: 1px solid {theme['bg_secondary']};
            height: 4px;
            background: {theme['bg_interactive']};
            margin: 2px 0;
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: {theme['accent_primary']};
            border: 1px solid {theme['accent_primary_pressed']};
            width: 18px;
            margin: -6px 0;
            border-radius: 9px;
        }}
        
        QStatusBar {{ font-weight: bold; }}
        
        #leftPanel, #boardBezel {{
            background-color: {theme['bg_tertiary']};
            border: 1px solid {theme['bg_secondary']};
            border-radius: 8px;
        }}
        #statusHeader {{
            font-size: 20px;
            font-weight: bold;
            padding: 12px;
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {theme['bg_secondary']}, stop:1 {theme['bg_tertiary']});
            border-top-left-radius: 7px;
            border-top-right-radius: 7px;
            border-bottom: 1px solid {theme['bg_secondary']};
            color: {theme['text_primary']};
        }}
        
        CellWidget {{
            background-color: {theme['bg_secondary']};
            border-radius: 0px;
            font-weight: bold;
            border: none;
            border-top: 1px solid {theme['bg_tertiary']};
            border-left: 1px solid {theme['bg_tertiary']};
        }}
        CellWidget:hover {{ background-color: {theme['bg_primary']}; }}
        CellWidget:disabled {{ background-color: {theme['bg_secondary']}; }}
        CellWidget[mark="X"] {{ color: {theme['accent_danger']}; }}
        CellWidget[mark="O"] {{ color: {theme['accent_warning']}; }}
        
        GameBoardWidget {{ background-color: transparent; }}
    """