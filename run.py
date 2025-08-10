#!/usr/bin/env python3
"""
Simple runner script for the Streamlit app
"""

import subprocess
import sys

def main():
    """Run the Streamlit application"""
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()