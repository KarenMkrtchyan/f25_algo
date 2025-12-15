#!/usr/bin/env python3
"""
Analyze requirements.txt for Python 3.11 compatibility and dependency conflicts.
"""

import sys
import subprocess
import re
from typing import List, Tuple, Dict

def check_python_version_compatibility(package: str, version: str) -> Tuple[bool, str]:
    """Check if a package version is compatible with Python 3.11."""
    # Known issues from our experience
    issues = {
        'ipython==9.6.0': (True, 'Requires Python 3.11+ ✓'),
        'networkx==3.5': (True, 'Requires Python 3.11+ ✓'),
        'ipykernel==7.1.0': (True, 'Should work with Python 3.11'),
        'contourpy==1.3.3': (False, 'Should be <1.3.3 (use 1.3.2)'),
    }
    
    key = f"{package}=={version}"
    if key in issues:
        return issues[key]
    
    return (True, 'Likely compatible')

def check_known_conflicts() -> Dict[str, List[str]]:
    """Return known dependency conflicts."""
    return {
        'torch==2.9.0': [
            'torchvision 0.20.1 requires torch==2.5.1 (conflict)',
            'Solution: Use torchvision compatible with torch 2.9.0 or downgrade torch to 2.5.1'
        ],
        'contourpy==1.3.3': [
            'Version 1.3.3 may not exist or may have issues',
            'Solution: Use contourpy<1.3.3 (e.g., 1.3.2)'
        ],
        'numpy==1.26.4': [
            'May conflict with some packages expecting older numpy',
            'Generally compatible but watch for warnings'
        ],
    }

def parse_requirements(filename: str) -> List[Tuple[str, str]]:
    """Parse requirements.txt file."""
    packages = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Skip editable installs (we'll note them separately)
            if line.startswith('-e') or line.startswith('git+'):
                packages.append(('git', line))
                continue
            
            # Parse package==version
            if '==' in line:
                parts = line.split('==')
                package = parts[0].strip()
                version = parts[1].strip()
                packages.append((package, version))
            else:
                packages.append((line, 'any'))
    
    return packages

def analyze_requirements(requirements_file: str):
    """Analyze requirements.txt for issues."""
    print("🔍 Analyzing requirements.txt for Python 3.11 compatibility\n")
    print("=" * 70)
    
    packages = parse_requirements(requirements_file)
    known_conflicts = check_known_conflicts()
    
    issues = []
    warnings = []
    git_packages = []
    compatible = []
    
    print(f"\n📦 Found {len(packages)} packages to analyze\n")
    
    for package, version in packages:
        if package == 'git':
            git_packages.append(version)
            continue
        
        is_compat, message = check_python_version_compatibility(package, version)
        key = f"{package}=={version}"
        
        if key in known_conflicts:
            issues.append((package, version, known_conflicts[key]))
        elif not is_compat:
            issues.append((package, version, [message]))
        elif 'Likely' in message:
            warnings.append((package, version, message))
        else:
            compatible.append((package, version))
    
    # Print results
    print("✅ COMPATIBLE PACKAGES (Python 3.11):")
    print("-" * 70)
    for package, version in compatible[:20]:  # Show first 20
        print(f"  ✓ {package}=={version}")
    if len(compatible) > 20:
        print(f"  ... and {len(compatible) - 20} more compatible packages")
    
    if git_packages:
        print(f"\n📥 GIT/EDITABLE PACKAGES ({len(git_packages)}):")
        print("-" * 70)
        for pkg in git_packages:
            print(f"  • {pkg}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        print("-" * 70)
        for package, version, message in warnings:
            print(f"  ⚠ {package}=={version}: {message}")
    
    if issues:
        print(f"\n❌ ISSUES FOUND ({len(issues)}):")
        print("-" * 70)
        for package, version, problems in issues:
            print(f"\n  ❌ {package}=={version}")
            for problem in problems:
                print(f"     • {problem}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY:")
    print(f"  ✅ Compatible: {len(compatible)}")
    print(f"  ⚠️  Warnings: {len(warnings)}")
    print(f"  ❌ Issues: {len(issues)}")
    print(f"  📥 Git packages: {len(git_packages)}")
    print(f"  📦 Total: {len(packages)}")
    
    # Recommendations
    if issues:
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 70)
        
        if any('torch' in pkg for pkg, _, _ in issues):
            print("  1. Fix torch/torchvision conflict:")
            print("     • Option A: Use torch 2.9.0 with compatible torchvision")
            print("       pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu121")
            print("     • Option B: Use torch 2.5.1 with torchvision 0.20.1")
            print("       pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121")
        
        if any('contourpy' in pkg for pkg, _, _ in issues):
            print("  2. Fix contourpy version:")
            print("     Change: contourpy==1.3.3")
            print("     To:     contourpy<1.3.3  # or contourpy==1.3.2")
        
        print("\n  3. The setup scripts now handle torch/torchvision automatically")
        print("     They will install compatible versions based on your requirements.txt")

if __name__ == "__main__":
    requirements_file = "requirements.txt"
    if len(sys.argv) > 1:
        requirements_file = sys.argv[1]
    
    try:
        analyze_requirements(requirements_file)
    except FileNotFoundError:
        print(f"❌ Error: {requirements_file} not found!")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error analyzing requirements: {e}")
        sys.exit(1)

