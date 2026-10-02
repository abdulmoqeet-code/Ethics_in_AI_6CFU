import os
import re
from datasets import load_dataset
import os
import json
import re
from datasets import load_dataset
import os
import json
import random
from collections import defaultdict
import os
from PIL import Image

import shutil

from collections import defaultdict
############################### code check
def get_and_clean_resumes(
    dataset_name, 
    token=None, 
    min_length=2000, 
    cache_path="data/filtered_resumes.json"
):

    # Check if JSON file already exists; if so, load directly
    if os.path.exists(cache_path):
        print(f"--- [Cache Found] Loading filtered resumes from: '{cache_path}' ---")
        with open(cache_path, "r", encoding="utf-8") as f:
            resumes = json.load(f)
        print(f"Loaded {len(resumes)} cached resumes from JSON.\n")
        return resumes


    # Load dataset from Hugging Face if JSON does not exist
    print(f"--- [Cache Not Found] Loading dataset from Hugging Face: {dataset_name} ---")
    hf_ds = load_dataset(dataset_name, split="train", token=token)
    print(f"Total raw dataset size: {len(hf_ds)} rows.")


    # Get data with raw length > 2,000 characters
    length_filtered = []
    for idx, row in enumerate(hf_ds):
        raw_text = row.get("Resume", "").strip()
        if len(raw_text) > min_length:
            length_filtered.append({"resume_id": idx, "raw_text": raw_text})
            
    print(f"\n[Step 2] Resumes with raw length > {min_length} chars: {len(length_filtered)}")


    # Step 3 & 4: Check for 'Professional Summary' first, fallback to 'Summary'
    pattern_prof = re.compile(r'professional\s+summary', re.IGNORECASE)
    pattern_sum = re.compile(r'summary', re.IGNORECASE)

    has_header_records = []
    missing_records = []

    for record in length_filtered:
        raw_text = record["raw_text"]
        
        # 1. Try matching 'Professional Summary' first
        match = pattern_prof.search(raw_text)
        
        # 2. Fallback to standalone 'Summary' if 'Professional Summary' is not found
        if not match:
            match = pattern_sum.search(raw_text)

        if match:
            has_header_records.append((record, match))
        else:
            missing_records.append(record)

    print(f"[Step 3 & 4] Rows missing both 'Professional Summary' & 'Summary': {len(missing_records)} (Removed)")
    print(f"             Rows kept with header ('Professional Summary' / 'Summary'): {len(has_header_records)}")

    # Print up to 30 rows that do NOT have either 'Professional Summary' or 'Summary'
    if missing_records:
        print("\n" + "="*80)
        print(f"PRINTING UP TO 30 ROWS MISSING HEADERS (TOTAL MISSING: {len(missing_records)}):")
        print("="*80)
        for i, rec in enumerate(missing_records[:30]):
            preview = rec["raw_text"][:150].replace("\n", " ")
            print(f"[{i+1}] Resume ID: {rec['resume_id']} | Raw Length: {len(rec['raw_text'])}")
            print(f"    Preview: {preview}...\n")
        print("="*80 + "\n")


    # Step 5: Remove text before the matched header (KEEPING the header word intact)
    final_resumes = []
    for record, match in has_header_records:
        # match.start() keeps 'Professional Summary' or 'Summary' and strips preceding text
        cleaned_text = record["raw_text"][match.start():].strip()

        final_resumes.append({
            "resume_id": record["resume_id"],
            "cv_text": cleaned_text,
            "length": len(cleaned_text)
        })

    print(f"[Step 5] Final resumes count after cleaning: {len(final_resumes)}")

    # ------------------------------------------------------------------
    # Step 6: Save final cleaned data to JSON file
    # ------------------------------------------------------------------
    cache_dir = os.path.dirname(cache_path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(final_resumes, f, indent=4, ensure_ascii=False)

    print(f"\n[Saved] Cleaned dataset saved to '{cache_path}' for future runs.")
    return final_resumes





# =====================================================================
# FUNCTION 3: Combine Resumes & Face Images
# =====================================================================


import os
import json
import random
from collections import defaultdict

from collections import defaultdict
import json
import os
import random


def prepare_audit_dataset(
    resumes_cache_path,
    faces_dir,
    group_idx=1,  # Group ID (1, 2, 3, 4, 5)
    num_cvs=None,  # If None, processes all CVs in the file (e.g., 36)
    pics_per_demographic=5,
    share_faces_across_cvs=True,  # If True, all CVs in group get same faces
    eval_same_faces=True,  # Pass 2: Determinism check (Set A repeated)
    eval_diff_faces=True,  # Pass 3: Robustness check (Set B)
    save_dataset_path=None,
    base_seed=42,  # Base seed for deterministic reproducibility
):
    """Builds an audit dataset pairing CVs with face images across 5 racial groups and 2 genders.

    Uses group_idx to guarantee exact, repeatable face selection for each CV
    group.
    """
    # 1. Load CVs from file
    with open(resumes_cache_path, "r", encoding="utf-8") as f:
        all_resumes = json.load(f)

    if num_cvs is None or num_cvs >= len(all_resumes):
        sampled_cvs = all_resumes
    else:
        # Deterministically sample subset of CVs if num_cvs < total
        cv_rng = random.Random(f"{base_seed}_cv_sample_group_{group_idx}")
        sampled_cvs = cv_rng.sample(all_resumes, k=num_cvs)

    # 2. Index available face images (Deterministic sorting)
    if not os.path.exists(faces_dir):
        raise FileNotFoundError(f"Face dataset directory not found: '{faces_dir}'")

    race_map = {"W": "White", "B": "Black", "A": "Asian", "L": "Latino", "I": "Indian"}
    gender_map = {"F": "Female", "M": "Male"}

    valid_prefixes = tuple(f"{r}{g}" for r in race_map.keys() for g in gender_map.keys())
    pool = defaultdict(list)

    # Use sorted() to ensure identical file order across different OS/runs
    for fname in sorted(os.listdir(faces_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        prefix = fname.split("_")[0].upper()
        if prefix in valid_prefixes:
            pool[prefix].append(os.path.join(faces_dir, fname))

    # Diagnostic check
    missing_groups = [p for p in valid_prefixes if len(pool[p]) == 0]
    if missing_groups:
        print(
            f"[WARNING] Demographic prefixes with 0 images in '{faces_dir}': {missing_groups}"
        )

    # 3. Helper function for deterministic face sampling
    def sample_ordered_face_set(seed_key):
        # Local RNG instance ensures isolation from global random state
        rng = random.Random(f"{base_seed}_group_{group_idx}_{seed_key}")
        face_set = []

        for race_code, race_label in race_map.items():
            for gender_code, gender_label in gender_map.items():
                prefix = f"{race_code}{gender_code}"
                available = sorted(
                    pool[prefix]
                )  # Sort pool for strict determinism

                chosen = rng.sample(
                    available, k=min(pics_per_demographic, len(available))
                )
                for img_path in chosen:
                    face_set.append(
                        {
                            "filename": os.path.basename(img_path),
                            "image_path": img_path,
                            "demographic_code": prefix,
                            "race": race_label,
                            "gender": gender_label,
                        }
                    )
        return face_set

    # 4. Pre-sample shared face sets if share_faces_across_cvs is True
    shared_set_a = (
        sample_ordered_face_set("shared_set_a")
        if share_faces_across_cvs
        else None
    )
    shared_set_b = (
        sample_ordered_face_set("shared_set_b")
        if (share_faces_across_cvs and eval_diff_faces)
        else None
    )

    # 5. Construct dataset structure
    prepared_data = {
        "config": {
            "group_idx": group_idx,
            "num_cvs": len(sampled_cvs),
            "pics_per_demographic": pics_per_demographic,
            "total_images_per_set": len(race_map)
            * len(gender_map)
            * pics_per_demographic,
            "share_faces_across_cvs": share_faces_across_cvs,
            "eval_same_faces": eval_same_faces,
            "eval_diff_faces": eval_diff_faces,
            "demographic_categories": list(valid_prefixes),
        },
        "items": [],
    }

    # 6. Build evaluations for each CV
    for cv_idx, cv in enumerate(sampled_cvs, start=1):
        cv_id = cv["resume_id"]

        # If unique per CV, generate face sets deterministically tied to this CV's ID
        faces_set_a = (
            shared_set_a
            if share_faces_across_cvs
            else sample_ordered_face_set(f"cv_{cv_id}_set_a")
        )

        # Pass 1: Set A (Base Evaluation)
        prepared_data["items"].append(
            {
                "cv_index": cv_idx,
                "cv_id": cv_id,
                "cv_text": cv["cv_text"],
                "eval_mode": "set_a_pass_1",
                "selected_faces": faces_set_a,
            }
        )

        # Pass 2: Set A Duplicate (Determinism Check)
        if eval_same_faces:
            prepared_data["items"].append(
                {
                    "cv_index": cv_idx,
                    "cv_id": cv_id,
                    "cv_text": cv["cv_text"],
                    "eval_mode": "set_a_pass_2_same_faces",
                    "selected_faces": faces_set_a,
                }
            )

        # Pass 3: Set B (Robustness Check with Set B faces)
        if eval_diff_faces:
            faces_set_b = (
                shared_set_b
                if share_faces_across_cvs
                else sample_ordered_face_set(f"cv_{cv_id}_set_b")
            )
            prepared_data["items"].append(
                {
                    "cv_index": cv_idx,
                    "cv_id": cv_id,
                    "cv_text": cv["cv_text"],
                    "eval_mode": "set_b_different_faces",
                    "selected_faces": faces_set_b,
                }
            )

    # 7. Save to disk
    if save_dataset_path:
        os.makedirs(os.path.dirname(save_dataset_path) or ".", exist_ok=True)
        with open(save_dataset_path, "w", encoding="utf-8") as f:
            json.dump(prepared_data, f, indent=2, ensure_ascii=False)
        print(
            f"Dataset for Group {group_idx} prepared successfully! Saved to '{save_dataset_path}'."
        )

    return prepared_data

################################### check done
# extract files from CFD 


def extract_and_move_cfd_images(cfd_dir="cfd", target_dir="cfd_processed", copy_files=False):
    """
    Extracts .jpg images from CFD subfolders, renames them with their demographic 
    prefix (e.g., BF_1.jpg, WM_2.jpg), and moves/copies them into a specified target directory.
    
    Parameters:
    - cfd_dir: Source folder containing CFD subfolders.
    - target_dir: Destination folder where all processed images will be placed.
    - copy_files: If False, moves files. If True, copies them.
    """
    if not os.path.exists(cfd_dir):
        raise FileNotFoundError(f"Source directory '{cfd_dir}' does not exist.")

    # Create destination directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    valid_prefixes = ("WF", "WM", "AF", "AM", "BF", "BM", "LF", "LM")
    demographic_counters = defaultdict(int)
    total_processed = 0

    for item in sorted(os.listdir(cfd_dir)):
        subfolder_path = os.path.join(cfd_dir, item)
        
        if os.path.isdir(subfolder_path) and item.startswith(valid_prefixes):
            prefix = item[:2].upper()
            
            for root, _, files in os.walk(subfolder_path):
                for fname in sorted(files):
                    if fname.lower().endswith(('.jpg', '.jpeg')):
                        demographic_counters[prefix] += 1
                        ext = os.path.splitext(fname)[1].lower()
                        
                        new_filename = f"{prefix}_{demographic_counters[prefix]}{ext}"
                        src_file = os.path.join(root, fname)
                        dest_file = os.path.join(target_dir, new_filename)
                        
                        if copy_files:
                            shutil.copy2(src_file, dest_file)
                        else:
                            shutil.move(src_file, dest_file)
                            
                        total_processed += 1

    action = "Copied" if copy_files else "Moved"
    print(f"Done! {action} {total_processed} JPG images into '{target_dir}/'.\n")
    print("Demographic Summary:")
    for prefix, count in sorted(demographic_counters.items()):
        print(f"  - {prefix}: {count} images")
#######################################
# get files from CFD_IND 


def process_cfd_india_hyphenated(
    source_dir="cfd_india_folder",
    target_dir="my_dataset/cfd_images",
    limit=20,
    copy_files=False
):
    """
    Selects up to 20 male (IM) and 20 female (IF) images with hyphenated names,
    renames them to IM_1.jpg, IF_1.jpg, etc., and moves/copies them to target_dir.
    Ignores all subdirectories.
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")

    os.makedirs(target_dir, exist_ok=True)

    # Filter strictly for top-level files
    all_files = [
        f for f in sorted(os.listdir(source_dir))
        if os.path.isfile(os.path.join(source_dir, f))
        and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    im_files = []
    if_files = []

    # Filter by hyphenated tag
    for fname in all_files:
        f_upper = fname.upper()
        if "-IM" in f_upper:
            im_files.append(fname)
        elif "-IF" in f_upper:
            if_files.append(fname)

    targets = {
        "IM": im_files[:limit],
        "IF": if_files[:limit]
    }

    processed_counts = {}

    for new_prefix, selected_files in targets.items():
        for idx, fname in enumerate(selected_files, start=1):
            ext = os.path.splitext(fname)[1].lower()
            new_filename = f"{new_prefix}_{idx}{ext}"
            
            src_path = os.path.join(source_dir, fname)
            dest_path = os.path.join(target_dir, new_filename)

            if copy_files:
                shutil.copy2(src_path, dest_path)
            else:
                shutil.move(src_path, dest_path)

        processed_counts[new_prefix] = len(selected_files)

    action = "Copied" if copy_files else "Moved"
    print(f"Done! {action} images into '{target_dir}/':")
    for pref, count in processed_counts.items():
        print(f"  - {pref}: {count} images")

########################
# image resize


def batch_resize_dataset(image_dir, target_size=(512, 512), quality=95):
    """
    Resizes all images in a folder to a standardized resolution using 
    LANCZOS resampling to prevent aliasing artifacts.
    """
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            fpath = os.path.join(image_dir, fname)
            with Image.open(fpath) as img:
                img = img.convert("RGB")
                # Downsample while maintaining high sharpness
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                img_resized.save(fpath, quality=quality)


    