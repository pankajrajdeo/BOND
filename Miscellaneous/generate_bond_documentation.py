#!/usr/bin/env python3
"""
Script to generate comprehensive documentation of the BOND codebase.
Generates a tree structure and contents of all files in the bond directory.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

def get_file_tree(directory, max_depth=10, current_depth=0):
    """Generate a tree structure of files and directories."""
    if current_depth > max_depth:
        return "  " * current_depth + "... (max depth reached)\n"
    
    tree = ""
    try:
        items = sorted(os.listdir(directory))
        for i, item in enumerate(items):
            item_path = os.path.join(directory, item)
            is_last = i == len(items) - 1
            
            # Determine the tree character
            if is_last:
                tree_char = "└── "
                next_prefix = "    "
            else:
                tree_char = "├── "
                next_prefix = "│   "
            
            tree += "  " * current_depth + tree_char + item + "\n"
            
            if os.path.isdir(item_path):
                # Recursively get subdirectory tree
                subtree = get_file_tree(item_path, max_depth, current_depth + 1)
                if subtree:
                    # Add the prefix to each line of the subtree
                    subtree_lines = subtree.split('\n')
                    prefixed_subtree = []
                    for line in subtree_lines:
                        if line.strip():
                            prefixed_subtree.append("  " * current_depth + next_prefix + line)
                        else:
                            prefixed_subtree.append("")
                    tree += '\n'.join(prefixed_subtree) + '\n'
    
    except PermissionError:
        tree += "  " * current_depth + "    [Permission denied]\n"
    except Exception as e:
        tree += "  " * current_depth + f"    [Error: {e}]\n"
    
    return tree

def get_file_contents(file_path):
    """Get complete file contents with line numbers."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        content = f"File: {file_path}\n"
        content += f"Size: {len(lines)} lines\n"
        content += "=" * 80 + "\n"
        
        # Show all lines
        for i, line in enumerate(lines, 1):
            content += f"{i:4d}: {line.rstrip()}\n"
        
        content += "\n" + "=" * 80 + "\n\n"
        return content
    
    except Exception as e:
        return f"Error reading {file_path}: {e}\n\n"

def is_text_file(file_path):
    """Check if a file is likely a text file."""
    text_extensions = {
        '.py', '.txt', '.md', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
        '.conf', '.sh', '.bash', '.zsh', '.fish', '.env', '.gitignore',
        '.html', '.css', '.js', '.ts', '.jsx', '.tsx', '.vue', '.svelte',
        '.xml', '.csv', '.sql', '.r', '.R', '.m', '.mat', '.ipynb',
        '.rst', '.tex', '.bib', '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp',
        '.java', '.kt', '.scala', '.go', '.rs', '.rb', '.php', '.pl',
        '.lua', '.swift', '.dart', '.fs', '.f90', '.f95', '.f03', '.f08'
    }
    
    # Check extension
    if Path(file_path).suffix.lower() in text_extensions:
        return True
    
    # Check if file has no extension but might be text
    if not Path(file_path).suffix:
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                # Check if it's mostly printable ASCII
                return all(32 <= b <= 126 or b in (9, 10, 13) for b in chunk)
        except:
            return False
    
    return False

def generate_documentation(bond_dir, output_file):
    """Generate comprehensive documentation of the BOND codebase."""
    
    print(f"Generating documentation for: {bond_dir}")
    print(f"Output file: {output_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("BOND Codebase Documentation\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source Directory: {bond_dir}\n")
        f.write(f"Output File: {output_file}\n")
        f.write("=" * 80 + "\n\n")
        
        # Tree structure
        f.write("DIRECTORY TREE STRUCTURE\n")
        f.write("-" * 80 + "\n")
        tree = get_file_tree(bond_dir)
        f.write(tree)
        f.write("\n" + "=" * 80 + "\n\n")
        
        # File contents
        f.write("FILE CONTENTS\n")
        f.write("-" * 80 + "\n\n")
        
        # Walk through all files
        total_files = 0
        processed_files = 0
        
        for root, dirs, files in os.walk(bond_dir):
            # Skip common directories that shouldn't be included
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
            
            for file in sorted(files):
                file_path = os.path.join(root, file)
                total_files += 1
                
                # Skip binary files and common non-text files
                if not is_text_file(file_path):
                    continue
                
                print(f"Processing: {file_path}")
                content = get_file_contents(file_path)
                f.write(content)
                processed_files += 1
        
        # Summary
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total files found: {total_files}\n")
        f.write(f"Text files processed: {processed_files}\n")
        f.write(f"Documentation generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\nDocumentation generated successfully!")
    print(f"Total files found: {total_files}")
    print(f"Text files processed: {processed_files}")
    print(f"Output saved to: {output_file}")

def main():
    """Main function."""
    bond_dir = "/Users/rajlq7/Downloads/Terms/BOND/bond"
    output_file = "/Users/rajlq7/Downloads/Terms/BOND/Miscellaneous/bond_documentation.txt"
    
    if not os.path.exists(bond_dir):
        print(f"Error: Directory {bond_dir} does not exist!")
        sys.exit(1)
    
    generate_documentation(bond_dir, output_file)

if __name__ == "__main__":
    main()
