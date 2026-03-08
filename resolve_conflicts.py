import os
import re

def resolve_conflict(block):
    # block is the full string including markers
    match = re.search(r'<<<<<<< (.*?)\n(.*?)\n=======\n(.*?)\n>>>>>>> (.*)', block, re.DOTALL)
    if not match:
        return block
    
    marker_head, content_head, content_other, marker_other = match.groups()
    
    # Logic: Prioritize luxical_tw over luxical
    has_luxical_head = 'luxical' in content_head and 'luxical_tw' not in content_head
    has_luxical_tw_other = 'luxical_tw' in content_other
    
    if has_luxical_head and has_luxical_tw_other:
        return content_other
    
    # Default to HEAD
    return content_head

def process_file(file_path):
    print(f"Processing {file_path}...")
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all conflict blocks
    # We use a non-greedy match for the content between markers
    # But conflicts can be multiline, so we need DOTALL
    # And we need to handle nested-ish looking markers (though git doesn't usually do that)
    
    def replace_func(match):
        return resolve_conflict(match.group(0))
    
    pattern = re.compile(r'<<<<<<< .*?\n.*?\n=======\n.*?\n>>>>>>> .*?\n', re.DOTALL)
    
    # Wait, the above pattern might be too simple if markers have extra info like filenames
    # Git conflict markers:
    # <<<<<<< HEAD
    # =======
    # >>>>>>> branch-name
    # Or with renames:
    # <<<<<<< HEAD:file_a
    # =======
    # >>>>>>> branch:file_b
    
    # Let's use a more robust regex for the markers
    pattern = re.compile(r'<<<<<<< [^\n]*\n(.*?)\n=======\n(.*?)\n>>>>>>> [^\n]*(\n|$)', re.DOTALL)
    
    new_content = pattern.sub(lambda m: resolve_conflict_from_groups(m), content)
    
    with open(file_path, 'w') as f:
        f.write(new_content)

def resolve_conflict_from_groups(match):
    content_head = match.group(1)
    content_other = match.group(2)
    
    # Special case for conductor/product.md where HEAD has more info
    # Let's check if luxical_tw is in both.
    
    has_luxical_tw_head = 'luxical_tw' in content_head.lower()
    has_luxical_tw_other = 'luxical_tw' in content_other.lower()
    has_luxical_head = 'luxical' in content_head.lower() and not has_luxical_tw_head
    
    if has_luxical_head and has_luxical_tw_other:
        return content_other + "\n"
    
    # If both have luxical_tw, or neither have luxical/luxical_tw, or HEAD has luxical_tw
    return content_head + "\n"

if __name__ == "__main__":
    files = [
        "README.md",
        "conductor/product.md",
        "conductor/tech-stack.md",
        "examples/custom_training_end_to_end.py",
        "examples/demo_luxical_capabilities.py",
        "src/luxical_tw/chinese_teacher_embedder.py",
        "src/luxical_tw/chinese_tokenization.py",
        "src/luxical_tw/embedder.py",
        "src/luxical_tw/fast_teacher_embedder.py",
        "src/luxical_tw/scripts/validate_dataset.py",
        "src/luxical_tw/trainer.py",
        "tests/benchmark_async_elite.py",
        "tests/test_chinese_tokenization.py",
        "tools/benchmark_luxical_native.py",
        "tools/profile_embedding.py",
        "tools/verify_framework_sota.py",
        "tools/verify_semantic_similarity.py",
        "tools/verify_semantic_similarity_ranked.py"
    ]
    for f in files:
        if os.path.exists(f):
            process_file(f)
        else:
            print(f"File not found: {f}")
