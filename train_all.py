#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   🎓 AI TEACHING ASSISTANT — FULL TRAINING PIPELINE 🎓         ║
║                                                                  ║
║   One-Click Solution: Extract → Clean → Chunk → Store → RAG     ║
║   Handles: PDF, PPTX, DOCX, TXT                               ║
║   Filters: Junk files, videos, images, compiled binaries        ║
║   Detects: Subject, Unit, Topic, Difficulty automatically       ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
  python train_all.py                          # Process default 'data/' directory
  python train_all.py --data-dir ./my_data     # Process a custom directory
  python train_all.py --interactive            # Interactive mode (asks for input)
  python train_all.py --dry-run                # Preview what would be processed
  python train_all.py --clean                  # Wipe old data before processing
"""

import os
import sys
import json
import time
import shutil
import argparse
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from etl.extract import DataExtractor
from etl.transform import DataTransformer
from etl.load import DataWarehouse
from analytics.evaluator import Evaluator


# ─────────────────────────────────────────────────────────────────────────────
# Console Colors
# ─────────────────────────────────────────────────────────────────────────────

GREEN  = '\033[92m'
BLUE   = '\033[94m'
YELLOW = '\033[93m'
RED    = '\033[91m'
CYAN   = '\033[96m'
MAGENTA = '\033[95m'
RESET  = '\033[0m'
BOLD   = '\033[1m'
DIM    = '\033[2m'


# ─────────────────────────────────────────────────────────────────────────────
# Banner and UI
# ─────────────────────────────────────────────────────────────────────────────

def print_banner():
    print(f"\n{CYAN}{BOLD}")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                                                                ║")
    print("║   🎓  AI TEACHING ASSISTANT — TRAINING PIPELINE  🎓           ║")
    print("║                                                                ║")
    print("║   Extract  →  Clean  →  Chunk  →  Embed  →  Store  →  RAG    ║")
    print("║                                                                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"{RESET}\n")


def print_step(step_num, total, title, icon="⚙️"):
    print(f"\n{BLUE}{BOLD}{'─'*60}{RESET}")
    print(f"{BLUE}{BOLD}  {icon}  STEP {step_num}/{total}: {title}{RESET}")
    print(f"{BLUE}{BOLD}{'─'*60}{RESET}\n")


def print_success(msg):
    print(f"  {GREEN}✅ {msg}{RESET}")


def print_warning(msg):
    print(f"  {YELLOW}⚠️  {msg}{RESET}")


def print_error(msg):
    print(f"  {RED}❌ {msg}{RESET}")


def print_info(msg):
    print(f"  {CYAN}ℹ️  {msg}{RESET}")


def print_metric(label, value, color=GREEN):
    print(f"  {DIM}{label:30s}{RESET} {color}{BOLD}{value}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Dependency Check
# ─────────────────────────────────────────────────────────────────────────────

def check_dependencies():
    """Check if required packages are installed"""
    missing = []

    checks = {
        'pypdf': 'pypdf',
        'pptx': 'python-pptx',
        'docx': 'python-docx',
        'pandas': 'pandas',
        'nltk': 'nltk',
    }

    for module, pip_name in checks.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)

    return missing


def install_missing(packages):
    """Install missing packages"""
    import subprocess
    for pkg in packages:
        print(f"  📦 Installing {pkg}...")
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', pkg, '-q'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"  {GREEN}✅ {pkg} installed{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Directory Scanner (Dry Run)
# ─────────────────────────────────────────────────────────────────────────────

def dry_run_scan(data_dir):
    """Preview what files would be processed without actually processing them"""
    from etl.extract import classify_file, is_junk_file

    print(f"\n{YELLOW}{BOLD}🔍 DRY RUN — Scanning: {os.path.abspath(data_dir)}{RESET}\n")

    process_count = 0
    skip_count = 0
    junk_count = 0
    by_type = {}

    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != 'vector_db']

        rel_root = os.path.relpath(root, data_dir)
        if rel_root == '.':
            rel_root = '(root)'

        has_content = False
        for f in sorted(files):
            file_path = os.path.join(root, f)
            classification, ext = classify_file(file_path)

            if not has_content:
                print(f"\n  📁 {rel_root}/")
                has_content = True

            if classification == 'process':
                print(f"      {GREEN}✅ {f} [{ext}]{RESET}")
                process_count += 1
                by_type[ext] = by_type.get(ext, 0) + 1
            elif classification == 'junk':
                print(f"      {RED}🗑️  {f} [JUNK]{RESET}")
                junk_count += 1
            else:
                print(f"      {DIM}⏭️  {f} [{ext} — skip]{RESET}")
                skip_count += 1

    print(f"\n{'='*50}")
    print(f"  {GREEN}Would process : {process_count} files{RESET}")
    print(f"  {YELLOW}Would skip    : {skip_count} files{RESET}")
    print(f"  {RED}Would remove  : {junk_count} junk files{RESET}")

    if by_type:
        print(f"\n  File types to process:")
        for ext, count in sorted(by_type.items()):
            print(f"    {ext:8s} → {count} file(s)")

    print(f"{'='*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Clean Old Data
# ─────────────────────────────────────────────────────────────────────────────

def clean_old_data(db_path="data/warehouse.db", vector_path="data/vector_db"):
    """Wipe existing processed data (warehouse + vector DB) for a fresh start"""
    print_warning("Cleaning old processed data...")

    if os.path.exists(db_path):
        os.remove(db_path)
        print_info(f"Removed {db_path}")

    if os.path.exists(vector_path):
        shutil.rmtree(vector_path)
        os.makedirs(vector_path, exist_ok=True)
        print_info(f"Cleared {vector_path}")

    print_success("Old data cleaned — starting fresh!")


# ─────────────────────────────────────────────────────────────────────────────
# Delete Physical Junk Files
# ─────────────────────────────────────────────────────────────────────────────

def delete_junk_files(data_dir):
    """Physically remove temp/junk files from the data directory"""
    from etl.extract import is_junk_file

    deleted = []
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != 'vector_db']

        for f in files:
            if is_junk_file(f):
                file_path = os.path.join(root, f)
                try:
                    os.remove(file_path)
                    deleted.append(f)
                    print(f"    🗑️  Deleted: {f}")
                except Exception as e:
                    print(f"    ⚠️  Could not delete {f}: {e}")

    return deleted


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_training_pipeline(data_dir, clean=False, delete_junk=True, chunk_size=500, chunk_overlap=50, run_eval=True):
    """
    Full ETL Training Pipeline:
      1) Check dependencies
      2) Clean junk files
      3) Extract all valid documents (multi-format)
      4) Transform (clean → chunk → enrich metadata)
      5) Load into data warehouse (SQL + Vector DB)
      6) Print comprehensive report
    """

    start_time = time.time()

    # ─────────────────────────────────────────────────────────────────
    # STEP 1: Environment Check
    # ─────────────────────────────────────────────────────────────────
    print_step(1, 6, "CHECKING ENVIRONMENT", "🔧")

    missing = check_dependencies()
    if missing:
        print_warning(f"Missing packages: {', '.join(missing)}")
        print_info("Installing automatically...")
        install_missing(missing)
        print_success("All dependencies ready!")
    else:
        print_success("All dependencies are installed")

    # Verify data directory
    abs_dir = os.path.abspath(data_dir)
    if not os.path.exists(abs_dir):
        print_error(f"Data directory not found: {abs_dir}")
        print_info("Please provide a valid directory path")
        return False

    print_success(f"Data directory: {abs_dir}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 2: Clean Old Data (optional) + Remove Junk Files
    # ─────────────────────────────────────────────────────────────────
    print_step(2, 6, "CLEANING & PREPARATION", "🧹")

    if clean:
        clean_old_data()

    if delete_junk:
        print_info("Scanning for junk/temp files...")
        deleted = delete_junk_files(data_dir)
        if deleted:
            print_success(f"Removed {len(deleted)} junk file(s)")
        else:
            print_success("No junk files found — directory is clean")

    # ─────────────────────────────────────────────────────────────────
    # STEP 3: Extract (Multi-Format)
    # ─────────────────────────────────────────────────────────────────
    print_step(3, 6, "EXTRACTING DATA (Multi-Format)", "📥")

    extractor = DataExtractor(data_dir=data_dir)
    documents = extractor.extract_from_directory(data_dir, recursive=True)

    # Print extraction stats
    extractor.print_stats()

    if not documents:
        print_error("No documents were extracted!")
        print_info("Check that your data directory contains valid files (.pdf, .pptx, .docx, .txt, .md)")
        return False

    print_success(f"Extracted {len(documents)} document(s) from {extractor.stats['files_processed']} file(s)")

    # ─────────────────────────────────────────────────────────────────
    # STEP 4: Transform (Clean + Chunk + Metadata)
    # ─────────────────────────────────────────────────────────────────
    print_step(4, 6, "TRANSFORMING DATA", "🔄")

    transformer = DataTransformer()
    print_info(f"Chunk size: {chunk_size} words | Overlap: {chunk_overlap} words")

    chunks = transformer.transform_pipeline(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    if not chunks:
        print_error("No chunks created after transformation!")
        return False

    print_success(f"Created {len(chunks)} chunks from {len(documents)} documents")

    # Show chunk distribution
    from collections import Counter
    subject_dist = Counter(c.metadata.get('subject', 'unknown') for c in chunks)
    topic_dist = Counter(c.metadata.get('topic', 'unknown') for c in chunks)
    type_dist = Counter(c.metadata.get('source_type', 'unknown') for c in chunks)

    print(f"\n  {CYAN}📊 Chunk Distribution:{RESET}")
    print(f"\n  {BOLD}By Subject:{RESET}")
    for subj, count in subject_dist.most_common():
        bar = '█' * min(count, 30)
        print(f"    {subj:20s} {count:4d} {DIM}{bar}{RESET}")

    print(f"\n  {BOLD}By Topic:{RESET}")
    for topic, count in topic_dist.most_common(10):
        bar = '█' * min(count, 30)
        print(f"    {topic:20s} {count:4d} {DIM}{bar}{RESET}")

    print(f"\n  {BOLD}By Source Type:{RESET}")
    for stype, count in type_dist.most_common():
        bar = '█' * min(count, 30)
        print(f"    {stype:20s} {count:4d} {DIM}{bar}{RESET}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 5: Load into Data Warehouse
    # ─────────────────────────────────────────────────────────────────
    print_step(5, 6, "LOADING INTO DATA WAREHOUSE", "💾")

    warehouse = DataWarehouse()
    warehouse.load_all(chunks)

    print_success("Data loaded into SQL database (warehouse.db)")
    print_success("Data loaded into Vector database (vector_db/)")

    # ─────────────────────────────────────────────────────────────────
    # STEP 6: Verification & Final Report
    # ─────────────────────────────────────────────────────────────────
    print_step(6, 7, "VERIFICATION & REPORT", "📊")

    elapsed = time.time() - start_time

    # Verify warehouse
    try:
        chunk_count = warehouse.query_sql("SELECT COUNT(*) FROM chunks")[0][0]
        subjects_in_db = warehouse.query_sql("SELECT DISTINCT subject FROM chunks")
        print_success(f"Warehouse verified: {chunk_count} chunks stored")
        print_success(f"Subjects in DB: {', '.join(s[0] for s in subjects_in_db)}")
    except Exception as e:
        print_warning(f"Could not verify warehouse: {e}")

    # Verify vector DB
    vector_chunks = os.path.join("data", "vector_db", "chunks.json")
    if os.path.exists(vector_chunks):
        with open(vector_chunks, 'r') as f:
            vec_data = json.load(f)
        print_success(f"Vector DB verified: {len(vec_data)} chunks indexed")
    else:
        print_warning("Vector DB file not found")

    # Final Summary
    print(f"\n{CYAN}{BOLD}{'═'*60}{RESET}")
    print(f"{CYAN}{BOLD}  🏁 TRAINING COMPLETE{RESET}")
    print(f"{CYAN}{BOLD}{'═'*60}{RESET}\n")

    print_metric("Total time", f"{elapsed:.1f} seconds")
    print_metric("Files processed", str(extractor.stats['files_processed']))
    print_metric("Documents extracted", str(len(documents)))
    print_metric("Chunks created", str(len(chunks)))
    print_metric("Junk files filtered", str(extractor.stats['junk_removed']))
    print_metric("Files skipped", str(extractor.stats['files_skipped']))

    print(f"\n  {BOLD}File types processed:{RESET}")
    for ftype, count in extractor.stats['by_type'].items():
        print_metric(f"    .{ftype}", f"{count} file(s)")

    print(f"\n{GREEN}{BOLD}  🎉 Your AI Teaching Assistant is now trained and ready!{RESET}")
    print(f"{DIM}  Run 'python launcher.py' to start the web interface{RESET}\n")

    # Save training log
    log = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': abs_dir,
        'elapsed_seconds': round(elapsed, 2),
        'files_processed': extractor.stats['files_processed'],
        'files_skipped': extractor.stats['files_skipped'],
        'junk_removed': extractor.stats['junk_removed'],
        'documents_extracted': len(documents),
        'chunks_created': len(chunks),
        'by_type': extractor.stats['by_type'],
        'by_subject': extractor.stats['by_subject'],
    }

    log_path = os.path.join("data", "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print_info(f"Training log saved to {log_path}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 7: Post-Training Evaluation
    # ─────────────────────────────────────────────────────────────────
    if run_eval:
        print_step(7, 7, "POST-TRAINING EVALUATION", "🔬")
        print_info("Running evaluation suite (Precision, Recall, F1, MAP, NDCG)…")
        try:
            evaluator = Evaluator(warehouse=warehouse)
            eval_report = evaluator.generate_evaluation_report(
                k=5, save_path="data/eval_report.json"
            )
            retrieval = eval_report.get('retrieval', {})
            print_metric("Macro Precision",  f"{retrieval.get('macro_precision', 0):.4f}")
            print_metric("Macro Recall",     f"{retrieval.get('macro_recall', 0):.4f}")
            print_metric("Macro F1",         f"{retrieval.get('macro_f1', 0):.4f}")
            print_metric("MAP",              f"{retrieval.get('MAP', 0):.4f}")
            print_metric("Mean NDCG",        f"{retrieval.get('mean_ndcg', 0):.4f}")
            print_success("Full evaluation report saved → data/eval_report.json")
        except Exception as e:
            print_warning(f"Evaluation skipped: {e}")
        print()

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Interactive Mode
# ─────────────────────────────────────────────────────────────────────────────

def interactive_mode():
    """Interactive training setup — asks user for inputs"""

    print_banner()

    print(f"{YELLOW}{BOLD}🎯 INTERACTIVE TRAINING MODE{RESET}\n")

    # Ask for data directory
    default_dir = "data"
    data_dir = input(f"  📂 Data directory [{default_dir}]: ").strip()
    if not data_dir:
        data_dir = default_dir

    if not os.path.exists(data_dir):
        print_error(f"Directory not found: {data_dir}")
        return

    # Ask about cleaning
    clean = input(f"  🧹 Clean old data before training? (y/N): ").strip().lower() == 'y'

    # Ask about chunk size
    chunk_input = input(f"  📏 Chunk size [500]: ").strip()
    chunk_size = int(chunk_input) if chunk_input.isdigit() else 500

    overlap_input = input(f"  🔄 Chunk overlap [50]: ").strip()
    chunk_overlap = int(overlap_input) if overlap_input.isdigit() else 50

    # Ask about dry run first
    preview = input(f"\n  👁️  Preview files first? (y/N): ").strip().lower() == 'y'

    if preview:
        dry_run_scan(data_dir)
        proceed = input(f"\n  🚀 Proceed with training? (Y/n): ").strip().lower()
        if proceed == 'n':
            print_info("Training cancelled.")
            return

    print()
    run_training_pipeline(
        data_dir=data_dir,
        clean=clean,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🎓 AI Teaching Assistant — Full Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_all.py                              Process default 'data/' directory
  python train_all.py --data-dir ./my_materials    Process custom directory
  python train_all.py --interactive                Interactive setup mode
  python train_all.py --dry-run                    Preview without processing
  python train_all.py --clean                      Fresh start (wipe old data)
  python train_all.py --chunk-size 300             Custom chunk size
        """
    )

    parser.add_argument('--data-dir', type=str, default='data',
                       help='Path to the data directory (default: data/)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview what would be processed without actually doing it')
    parser.add_argument('--clean', action='store_true',
                       help='Wipe old processed data before training')
    parser.add_argument('--no-delete-junk', action='store_true',
                       help='Do NOT delete junk files from disk')
    parser.add_argument('--chunk-size', type=int, default=500,
                       help='Number of words per chunk (default: 500)')
    parser.add_argument('--chunk-overlap', type=int, default=50,
                       help='Number of overlapping words between chunks (default: 50)')
    parser.add_argument('--no-eval', action='store_true',
                       help='Skip post-training evaluation (faster)')  

    args = parser.parse_args()

    print_banner()

    if args.interactive:
        interactive_mode()
    elif args.dry_run:
        dry_run_scan(args.data_dir)
    else:
        run_training_pipeline(
            data_dir=args.data_dir,
            clean=args.clean,
            delete_junk=not args.no_delete_junk,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            run_eval=not args.no_eval,
        )


if __name__ == "__main__":
    main()
