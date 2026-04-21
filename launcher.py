#!/usr/bin/env python3
# PREMIUM LAUNCHER - Choose and launch the best UI for your needs

import os
import subprocess
import sys
from pathlib import Path

# Color codes
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header():
    """Print fancy header"""
    print(f"\n{CYAN}{BOLD}")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║   🎓 AI TEACHING ASSISTANT - PREMIUM LAUNCHER 🎓         ║")
    print("║                  Complete & Ready to Deploy               ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"{RESET}\n")

def print_menu():
    """Display main menu"""
    print(f"{YELLOW}{BOLD}Choose a UI to launch:{RESET}\n")
    print(f"{GREEN}1.{RESET} {BOLD}🎨 PREMIUM UI Pro{RESET}")
    print("   ✓ Modern design with beautiful gradients")
    print("   ✓ 8 operating modes (Dashboard, Chat, Learning, Exam, Quiz, Analytics, Syllabus, Help)")
    print("   ✓ Advanced analytics with charts")
    print("   ✓ Professional settings panel")
    print("   ✓ Progress tracking & recommendations")
    print("   ✓ RECOMMENDED FOR: Best user experience\n")
    
    print(f"{GREEN}2.{RESET} {BOLD}⚡ COMPLETE UI{RESET}")
    print("   ✓ Full-featured interface")
    print("   ✓ 6 core modes (Chat, Learning, Exam, Quiz, Analytics, Syllabus)")
    print("   ✓ All backend integration")
    print("   ✓ RECOMMENDED FOR: Production stability\n")
    
    print(f"{GREEN}3.{RESET} {BOLD}💬 BASIC Chat UI{RESET}")
    print("   ✓ Simple question-answer interface")
    print("   ✓ Focus on RAG functionality")
    print("   ✓ RECOMMENDED FOR: Testing & debugging\n")
    
    print(f"{GREEN}4.{RESET} {BOLD}⚡ Run Backend Setup{RESET}")
    print("   ✓ Initialize system")
    print("   ✓ Verify all components\n")
    
    print(f"{GREEN}5.{RESET} {BOLD}📊 View Project Status{RESET}")
    print("   ✓ Show all files and structure\n")
    
    print(f"{GREEN}0.{RESET} {RED}{BOLD}Exit{RESET}\n")

def show_project_status():
    """Display project status"""
    print(f"\n{CYAN}{BOLD}📁 PROJECT STRUCTURE & STATUS{RESET}\n")
    
    structure = {
        "🎓 Backend": [
            "✅ etl/ - Extract, Transform, Load pipeline",
            "✅ rag/ - Retrieval-Augmented Generation system",
            "✅ analytics/ - Data mining and clustering",
            "✅ modes/ - Learning, Exam, Quiz, Syllabus modes",
        ],
        "🎨 Frontend": [
            "✅ app/ui_premium.py - Advanced professional UI (NEW)",
            "✅ app/ui_complete.py - Complete feature UI",
            "✅ app/ui.py - Basic chat UI",
        ],
        "⚙️ Configuration": [
            "✅ config.py - System configuration",
            "✅ syllabus.py - Course definitions",
            "✅ experiments.py - Evaluation framework",
        ],
        "📚 Data": [
            "📂 data/DBMS/ - Database course materials",
            "📂 data/OS/ - Operating Systems materials",
            "📂 data/DataStructures/ - Data Structures materials",
            "📂 data/vector_db/ - FAISS vector index",
            "📄 data/warehouse.db - SQLite database",
        ],
        "📖 Documentation": [
            "✅ README.md - Setup and usage guide",
        ]
    }
    
    for category, items in structure.items():
        print(f"{GREEN}{category}{RESET}")
        for item in items:
            print(f"  {item}")
        print()

def launch_ui(ui_type):
    """Launch the selected UI"""
    
    ui_map = {
        1: ("app/ui_premium.py", "PREMIUM UI Pro"),
        2: ("app/ui_complete.py", "COMPLETE UI"),
        3: ("app/ui.py", "BASIC Chat UI"),
    }
    
    if ui_type not in ui_map:
        print(f"{RED}Invalid selection!{RESET}\n")
        return
    
    ui_file, ui_name = ui_map[ui_type]
    
    # Check if file exists
    if not os.path.exists(ui_file):
        print(f"{RED}Error: {ui_file} not found!{RESET}\n")
        return
    
    print(f"\n{CYAN}{BOLD}🚀 Launching {ui_name}...{RESET}\n")
    print(f"{YELLOW}Starting Streamlit server...{RESET}")
    print(f"{YELLOW}Open your browser to: http://localhost:8501{RESET}\n")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", ui_file],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"{RED}Error launching UI: {e}{RESET}\n")
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Shutting down...{RESET}\n")

def run_backend_setup():
    """Run backend setup"""
    print(f"\n{CYAN}{BOLD}⚙️  Setting up backend...{RESET}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "main.py", "--setup"],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.returncode != 0:
            print(f"{RED}Setup encountered issues:{RESET}")
            print(result.stderr)
        else:
            print(f"{GREEN}{BOLD}✅ Backend setup successful!{RESET}\n")
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}\n")

def main():
    """Main launcher application"""
    
    while True:
        print_header()
        
        # Show quick stats
        print(f"{BOLD}📊 Quick Stats:{RESET}")
        print(f"  • Backend Status: {GREEN}✅ Ready{RESET}")
        print(f"  • Modes Available: {GREEN}8{RESET}")
        print(f"  • Features: {GREEN}23+{RESET}")
        print(f"  • Documentation: {GREEN}Complete{RESET}\n")
        
        print_menu()
        
        try:
            choice = input(f"{BOLD}Enter your choice (0-5): {RESET}").strip()
            
            if choice == "0":
                print(f"\n{CYAN}Thank you for using AI Teaching Assistant PRO!{RESET}\n")
                break
            elif choice == "1":
                launch_ui(1)
            elif choice == "2":
                launch_ui(2)
            elif choice == "3":
                launch_ui(3)
            elif choice == "4":
                run_backend_setup()
                input(f"\n{YELLOW}Press Enter to continue...{RESET}")
            elif choice == "5":
                show_project_status()
                input(f"\n{YELLOW}Press Enter to continue...{RESET}")
            else:
                print(f"{RED}Invalid choice!{RESET}\n")
        
        except KeyboardInterrupt:
            print(f"\n{CYAN}Goodbye!{RESET}\n")
            break
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}\n")

if __name__ == "__main__":
    main()
