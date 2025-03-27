"""
Main entry point for Smart Trade Advisor.
This launcher allows you to choose between CLI and Web interfaces.
"""
import sys
import argparse
from main import run_cli
from app import app

def main():
    parser = argparse.ArgumentParser(description="Smart Trade Advisor")
    parser.add_argument("--web", action="store_true", help="Run the web interface")
    parser.add_argument("--cli", action="store_true", help="Run the command line interface")
    parser.add_argument("--port", type=int, default=5000, help="Port for web interface")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    if args.web:
        print("Starting web interface...")
        app.run(debug=args.debug, port=args.port)
    elif args.cli:
        print("Starting command line interface...")
        run_cli()
    else:
        # Default to web interface
        print("No interface specified, defaulting to web...")
        app.run(debug=args.debug, port=args.port)

if __name__ == "__main__":
    main() 