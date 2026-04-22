#!/usr/bin/env python3
"""Migrate existing signal history to persistent location.

If you have data in the default eval_results/ directory that you want to
preserve, run this script to migrate it to an external directory.

Usage:
    # First, configure the external directory
    export EVAL_RESULTS_DIR=/path/to/external/eval_results
    
    # Then run migration
    python scripts/migrate_history.py
    
    # Or run setup first, which sets the env var
    python scripts/setup_persistence.py
    python scripts/migrate_history.py
"""

import shutil
import sys
import json
from pathlib import Path
from datetime import datetime


def get_source_dir() -> Path:
    """Get source directory (current eval_results)."""
    return Path("eval_results")


def get_target_dir() -> Path:
    """Get target directory from environment or exit."""
    env_dir = Path.home() / "TradingAgentsData" / "eval_results"
    
    # Check if EVAL_RESULTS_DIR is set
    import os
    if "EVAL_RESULTS_DIR" in os.environ:
        return Path(os.environ["EVAL_RESULTS_DIR"])
    
    # Ask user
    print(f"Target directory not set. Default: {env_dir}")
    response = input(f"Use default? (y/n/custom path): ").strip().lower()
    
    if response == 'y' or response == '':
        return env_dir
    elif response == 'n':
        print("Please set EVAL_RESULTS_DIR environment variable and re-run.")
        print("Example: export EVAL_RESULTS_DIR=/path/to/eval_results")
        sys.exit(1)
    else:
        return Path(response)


def count_files_and_size(directory: Path) -> tuple:
    """Count files and total size in directory."""
    if not directory.exists():
        return 0, 0
    
    count = 0
    total_size = 0
    
    for path in directory.rglob("*"):
        if path.is_file():
            count += 1
            total_size += path.stat().st_size
    
    return count, total_size


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def migrate_ticker(source_ticker_dir: Path, target_ticker_dir: Path) -> dict:
    """Migrate a single ticker's data."""
    stats = {
        "log_files": 0,
        "pulse_files": 0,
        "shadow_files": 0,
        "other_files": 0,
        "errors": [],
    }
    
    # Create target directory
    target_ticker_dir.mkdir(parents=True, exist_ok=True)
    
    # Migrate TradingAgentsStrategy_logs
    source_logs = source_ticker_dir / "TradingAgentsStrategy_logs"
    target_logs = target_ticker_dir / "TradingAgentsStrategy_logs"
    
    if source_logs.exists():
        target_logs.mkdir(exist_ok=True)
        
        for log_file in source_logs.glob("*.json"):
            try:
                target_file = target_logs / log_file.name
                
                # Check if target already exists and is newer
                if target_file.exists():
                    source_mtime = log_file.stat().st_mtime
                    target_mtime = target_file.stat().st_mtime
                    
                    if target_mtime >= source_mtime:
                        continue  # Skip, target is newer or same
                
                shutil.copy2(log_file, target_file)
                stats["log_files"] += 1
                
            except Exception as e:
                stats["errors"].append(f"{log_file.name}: {e}")
    
    # Migrate pulse directory
    source_pulse = source_ticker_dir / "pulse"
    target_pulse = target_ticker_dir / "pulse"
    
    if source_pulse.exists():
        target_pulse.mkdir(exist_ok=True)
        
        for pulse_file in source_pulse.glob("*.jsonl"):
            try:
                target_file = target_pulse / pulse_file.name
                shutil.copy2(pulse_file, target_file)
                stats["pulse_files"] += 1
            except Exception as e:
                stats["errors"].append(f"pulse/{pulse_file.name}: {e}")
    
    # Migrate shadow directory
    source_shadow = source_ticker_dir / "shadow"
    target_shadow = target_ticker_dir / "shadow"
    
    if source_shadow.exists():
        target_shadow.mkdir(exist_ok=True)
        
        for shadow_file in source_shadow.glob("*.jsonl"):
            try:
                target_file = target_shadow / shadow_file.name
                shutil.copy2(shadow_file, target_file)
                stats["shadow_files"] += 1
            except Exception as e:
                stats["errors"].append(f"shadow/{shadow_file.name}: {e}")
    
    # Copy other files (champion.json, metrics.json, etc.)
    for other_file in source_ticker_dir.glob("*.json"):
        try:
            target_file = target_ticker_dir / other_file.name
            shutil.copy2(other_file, target_file)
            stats["other_files"] += 1
        except Exception as e:
            stats["errors"].append(f"{other_file.name}: {e}")
    
    return stats


def main():
    print("=" * 60)
    print("📦 Signal History Migration Tool")
    print("=" * 60)
    
    source = get_source_dir()
    target = get_target_dir()
    
    print(f"\n📁 Source: {source.absolute()}")
    print(f"📁 Target: {target.absolute()}")
    
    # Check if source exists
    if not source.exists():
        print("\n❌ No existing eval_results directory found.")
        print("   Nothing to migrate.")
        return
    
    # Count source data
    source_count, source_size = count_files_and_size(source)
    print(f"\n📊 Found {source_count} files ({format_size(source_size)}) to migrate")
    
    if source_count == 0:
        print("\n   No files to migrate. Exiting.")
        return
    
    # Check if target already has data
    if target.exists():
        target_count, target_size = count_files_and_size(target)
        if target_count > 0:
            print(f"\n⚠️  Target already has {target_count} files ({format_size(target_size)})")
            response = input("   Merge (keep both) / Overwrite / Cancel? (m/o/c): ").strip().lower()
            
            if response == 'c':
                print("\n❌ Cancelled.")
                return
            elif response == 'o':
                print("\n   ⚠️  Will overwrite existing files...")
        else:
            print("\n✅ Target directory is empty, ready for migration")
    
    # Confirm
    print("\n" + "-" * 60)
    confirm = input("Start migration? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("\n❌ Cancelled.")
        return
    
    # Perform migration
    print("\n🚀 Starting migration...")
    print("-" * 60)
    
    total_stats = {
        "tickers": 0,
        "log_files": 0,
        "pulse_files": 0,
        "shadow_files": 0,
        "other_files": 0,
        "errors": [],
    }
    
    # Migrate each ticker
    for ticker_dir in source.iterdir():
        if not ticker_dir.is_dir():
            continue
        
        ticker = ticker_dir.name
        target_ticker_dir = target / ticker
        
        print(f"  📈 Migrating {ticker}...", end=" ")
        
        stats = migrate_ticker(ticker_dir, target_ticker_dir)
        
        total_migrated = (
            stats["log_files"] + 
            stats["pulse_files"] + 
            stats["shadow_files"] + 
            stats["other_files"]
        )
        
        print(f"✓ {total_migrated} files")
        
        total_stats["tickers"] += 1
        total_stats["log_files"] += stats["log_files"]
        total_stats["pulse_files"] += stats["pulse_files"]
        total_stats["shadow_files"] += stats["shadow_files"]
        total_stats["other_files"] += stats["other_files"]
        total_stats["errors"].extend(stats["errors"])
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Migration Complete!")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"  Tickers migrated: {total_stats['tickers']}")
    print(f"  Log files: {total_stats['log_files']}")
    print(f"  Pulse files: {total_stats['pulse_files']}")
    print(f"  Shadow files: {total_stats['shadow_files']}")
    print(f"  Other files: {total_stats['other_files']}")
    
    total_migrated = (
        total_stats["log_files"] + 
        total_stats["pulse_files"] + 
        total_stats["shadow_files"] + 
        total_stats["other_files"]
    )
    print(f"  Total: {total_migrated} files")
    
    if total_stats["errors"]:
        print(f"\n⚠️  {len(total_stats['errors'])} errors:")
        for error in total_stats["errors"][:10]:  # Show first 10
            print(f"    - {error}")
        if len(total_stats["errors"]) > 10:
            print(f"    ... and {len(total_stats['errors']) - 10} more")
    
    # Backup source
    print("\n💾 Backing up source directory...")
    backup_name = f"eval_results_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_path = Path(backup_name)
    
    try:
        shutil.move(str(source), str(backup_path))
        print(f"✅ Source backed up to: {backup_path}")
    except Exception as e:
        print(f"⚠️  Could not backup source: {e}")
        print("   You may want to manually rename eval_results/")
    
    # Next steps
    print("\n" + "=" * 60)
    print("📝 Next Steps:")
    print("=" * 60)
    print("\n1. Update your server startup to use the new directory:")
    print(f"   export EVAL_RESULTS_DIR={target}")
    print("   python server.py")
    print("\n2. Or add to your .env file:")
    print(f"   EVAL_RESULTS_DIR={target}")
    print("\n3. If deploying to Render, set this in Environment variables")
    print("\n4. Your data is now in a persistent location outside the git repo!")
    
    # Create symlink for convenience (optional)
    print("\n" + "-" * 60)
    response = input("Create symlink from eval_results → new location? (y/n): ").strip().lower()
    if response == 'y':
        try:
            Path("eval_results").symlink_to(target, target_is_directory=True)
            print("✅ Created symlink: eval_results → {target}")
        except Exception as e:
            print(f"⚠️  Could not create symlink: {e}")


if __name__ == "__main__":
    main()
