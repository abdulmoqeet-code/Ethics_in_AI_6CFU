import json
from pathlib import Path


def combine_cluster_json_files(
    file_paths: list[str], output_json_path: str = "combined_cv_evaluations.json"
):
    """Loads 3 cluster JSON files, combines their cv_evaluations lists,

    and exports a single JSON file retaining the original nested structure.
    """
    combined_cv_evaluations = []

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"Error: File not found -> {file_path}")
            continue

        # Infer cluster label from filename (e.g., 'cluster_0' from 'qwen_cluster_0.json')
        cluster_tag = path.stem.split("_")[-1]

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract cv_evaluations array from the root dictionary
        cv_evals = data.get("cv_evaluations", [])

        # Tag each CV evaluation with its cluster origin
        for item in cv_evals:
            item["cv_cluster"] = f"cluster_{cluster_tag}"
            combined_cv_evaluations.append(item)

        print(
            f"Loaded {len(cv_evals)} CVs from {file_path} (Tagged as: cluster_{cluster_tag})"
        )

    # Build the combined dictionary maintaining the exact JSON structure
    combined_data = {
        "total_cv_count": len(combined_cv_evaluations),
        "cv_evaluations": combined_cv_evaluations,
    }

    # Save to the target single JSON file
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=4)

    print(f"\nSuccessfully combined into: {output_json_path}")
    print(f"Total CVs in combined file: {len(combined_cv_evaluations)}")