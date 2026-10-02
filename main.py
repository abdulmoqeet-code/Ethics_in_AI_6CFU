import yaml
from src.dataset_loader import (
    
    get_and_clean_resumes,
   
    prepare_audit_dataset,
    extract_and_move_cfd_images,
    process_cfd_india_hyphenated,
    batch_resize_dataset
)
from src.dataset_pre import categorize_and_save_cvs
from src.dataset_pre import create_tiered_distance_groups

from src.model_loader import load_vlm

from src.eval import evaluate_audit_dataset

from src.matrix import run_multi_model_audit
import json
import os
from src.post_pro import combine_cluster_json_files

# 1. Load config
with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
  
# check to run only data pre-evalution or post Matrices code
data_pre_eval=cfg.get('data_pre')
post_pro=cfg.get('post_pro')

if (data_pre_eval):
  model, processor = load_vlm(cfg)
  # extract the CFD india images

  
  process_cfd_india_hyphenated(
        source_dir="/content/drive/MyDrive/CFD-INDIA",  # Source folder with CFD_IM... and CFD_IF... files
        target_dir="/content/drive/MyDrive/Ethics-in-AI/data/cfd_face",     # Destination folder
        limit=20,                                # 20 from each category
        copy_files=False                         # Set True to copy, False to move
    )
  # extract other CFD images (other than indian)
  extract_and_move_cfd_images(
        cfd_dir="/content/drive/MyDrive/CFD", 
        target_dir="/content/drive/MyDrive/Ethics-in-AI/data/cfd_face", 
        copy_files=False  # Set to False to move files directly
    )
  # resize the images
  batch_resize_dataset(image_dir="/content/drive/MyDrive/Ethics-in-AI/data/cfd_face", target_size=(224, 224), quality=95)

  # get the resumes from the hugging face, and select only those with atleast 2k char
  
  dataset_resume=cfg.get('hf_dataset_name_resume')
  hf_token=cfg.get("hf_token")
  
  resumes=get_and_clean_resumes(
    dataset_name=dataset_resume, 
    token=hf_token, 
    min_length=2000, 
    cache_path="data/filtered_resumes.json")

  # scale the CVs, based on the pre evalution
  updated_cv_list=categorize_and_save_cvs(resumes, model, processor, output_json_path="data/CVs_with_tier")
  
  #load the json file, if not getting from previous function
  json_file_path = "/content/drive/MyDrive/Ethics-in-AI/data/CVs_with_tier.json"

  with open(json_file_path, "r", encoding="utf-8") as f:
    updated_cv_list = json.load(f)

  # Create 5 distinct groups from the 36 CV pool (12 Confident, 12 Average, 12 Weak)
  distance_groups = create_tiered_distance_groups(
    cv_list=updated_cv_list,
    n_groups=5,
    clusters_per_tier=12,
    embedding_model_name="all-MiniLM-L6-v2",
    output_dir="data/final_ranked_CVs"
  )
#post matrix code 
elif(post_pro):
  #the result of all of three models
  input_files_25 = [
        "/content/drive/MyDrive/Ethics-in-AI/results/audit_results_Qwen_Qwen2.5-VL-3B-Instruct_group_1share_faces_across_cvsFalse.json",
        "/content/drive/MyDrive/Ethics-in-AI/results/audit_results_Qwen_Qwen2.5-VL-3B-Instruct_group_2share_faces_across_cvsFalse.json",
        "/content/drive/MyDrive/Ethics-in-AI/results/audit_results_Qwen_Qwen2.5-VL-3B-Instruct_group_3share_faces_across_cvsFalse.json",
    ]
  input_files_35=[
    "/content/drive/MyDrive/Ethics-in-AI/results/audit_results_Qwen_Qwen3.5-2B_group_1share_faces_across_cvsFalse.json",
    "/content/drive/MyDrive/Ethics-in-AI/results/audit_results_Qwen_Qwen3.5-2B_group_2share_faces_across_cvsFalse.json",
    "/content/drive/MyDrive/Ethics-in-AI/results/audit_results_Qwen_Qwen3.5-2B_group_3share_faces_across_cvsFalse.json",
  ]
  input_files_35_m=[
    "/content/drive/MyDrive/Ethics-in-AI/results/audit_results_microsoft_Phi-3.5-vision-instruct_group_1share_faces_across_cvsFalse.json",
    "/content/drive/MyDrive/Ethics-in-AI/results/audit_results_microsoft_Phi-3.5-vision-instruct_group_2share_faces_across_cvsFalse.json",
    "/content/drive/MyDrive/Ethics-in-AI/results/audit_results_microsoft_Phi-3.5-vision-instruct_group_3share_faces_across_cvsFalse.json",

  ]
  output_json_path_25="/content/drive/MyDrive/Ethics-in-AI/results/qwen25_model_all_clusters_combined.json"
  output_json_path_35="/content/drive/MyDrive/Ethics-in-AI/results/qwen35_model_all_clusters_combined.json"
  output_json_path_35m="/content/drive/MyDrive/Ethics-in-AI/results/micro35_model_all_clusters_combined.json"
  combine_cluster_json_files(
        file_paths=input_files_25,
        output_json_path=output_json_path_25,
    )
  combine_cluster_json_files(
        file_paths=input_files_35,
        output_json_path=output_json_path_35,
    )
  combine_cluster_json_files(
        file_paths=input_files_35_m,
        output_json_path=output_json_path_35m,
    )
  
  model_paths = {
        "Qwen2.5-VL": output_json_path_25,
        "Qwen3.5-VL": output_json_path_35,
        "Phi-3.5-Vision":output_json_path_35m,
        
    }

    # Execute Audit across all 3 models
  run_multi_model_audit(model_paths)

# mian model evalution code
else:
  #load model and the processor
  model, processor = load_vlm(cfg)

  # select the cv file based on index
  group_idx = cfg["group_idx"]
  cv_dir = cfg.get("cv_dir")
  faces_dir = cfg.get("faces_dir")
  output_dir = cfg.get("final_eval_dataset_output_dir")
  share_faces_across_cvs=cfg.get("share_faces_across_cvs")

  # 2. Match exact naming convention from create_tiered_distance_groups
  cv_filename = f"group_{group_idx}_tier_selection.json"
  input_cv_file = os.path.join(cv_dir, cv_filename)

  # 3. Match output dataset naming convention
  output_filename = f"prepared_eval_dataset_group_{group_idx}.json"
  output_dataset_file = os.path.join(output_dir, output_filename)

    # Ensure input CV file exists
  if not os.path.exists(input_cv_file):
    raise FileNotFoundError(
            f"Target CV group file not found for group {group_idx} at: '{input_cv_file}'"
        )
  # combine cv and face dataset
  dataset = prepare_audit_dataset(
        resumes_cache_path=input_cv_file,
        faces_dir=faces_dir,
        group_idx=group_idx,  # Guarantees unique & fixed faces per group
        num_cvs=None,  # Uses all 36 CVs (12 Confident, 12 Average, 12 Weak)
        pics_per_demographic=1,
        share_faces_across_cvs=share_faces_across_cvs, #each set of CVs will get same set of faces, False(to gte different)
        eval_same_faces=True,  # Pass 2
        eval_diff_faces=True,  # Pass 3
        save_dataset_path=output_dataset_file,
    )
  # output path
 


  # Clean model name to ensure valid file path (e.g., "Qwen/Qwen2-VL-7B" -> "Qwen_Qwen2-VL-7B")
  raw_model_name = cfg.get("model_id")
  clean_model_name = raw_model_name.replace("/", "_").replace(" ", "_")

  # Directories from config
  results_dir = cfg.get("results_dir")

  # Ensure results directory exists
  os.makedirs(results_dir, exist_ok=True)

  # Build dynamic paths based on group index and model name
  results_output_path = os.path.join(
    results_dir, f"audit_results_{clean_model_name}_group_{group_idx}share_faces_across_cvs{share_faces_across_cvs}.json"
  )

  #eval code
  results = evaluate_audit_dataset(
    dataset_input=dataset, # Pass prepared_dataset dict or path string
    model=model,
    processor=processor,
    model_type=cfg.get("model_type"),
    output_results_path=results_output_path,
    device=cfg["device"],
    batch_size=cfg["batch_size"]
  )

    



