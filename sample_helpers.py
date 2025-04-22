import os
import glob
import math
import json
from tabulate import tabulate

def corrected_sample_std_dev(measurements):
    if len(measurements) < 2:
        print("Warning: At least two measurements are required to compute the sample standard deviation.")
        return 0.
    mean = sum(measurements) / len(measurements)
    squared_diffs = [(x - mean) ** 2 for x in measurements]
    variance = sum(squared_diffs) / (len(measurements) - 1)
    std_dev = math.sqrt(variance)

    return std_dev

def aggregate_results(results):
    """Process results into aggregated output format"""
    output_json = {}
    for eval_type in results:
        output_json[eval_type] = {}
        for metric in results[eval_type]:
            values = [v for v, s in results[eval_type][metric]]
            stds = [s for v, s in results[eval_type][metric]]

            avg = sum(values) / len(values)
            avg_std = corrected_sample_std_dev(values) / math.sqrt(len(values))

            output_json[eval_type][metric] = avg
            output_json[eval_type][f"{metric}:std"] = avg_std
            output_json[eval_type][f"{metric}:results"] = values
            output_json[eval_type][f"{metric}:std:results"] = stds
    return output_json

def save_results(output_json, results_dir, model_name):
    """Save results to file"""
    output_filename = f"{results_dir}/{model_name}_results.json"
    with open(output_filename, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"Results saved to {output_filename}")

def tabulate_results(output_json):
    """Print summary table"""
    table = []
    for eval_type in output_json:
        if 'score' in output_json[eval_type]:
            avg_score = output_json[eval_type]['score']
            std_score = output_json[eval_type]['score:std']
            table.append([eval_type, avg_score, std_score])

    table.sort(key=lambda x: x[0])
    print("\nScore averages and stds:")
    print(tabulate(table, headers=['Evaluation Type', 'Average Score', 'Std Dev'], tablefmt='orgtbl'))

def load_results_from_runs(results_dir, model):
    """Reconstruct results dictionary from individual run files"""
    results = {}
    pattern = os.path.join(results_dir, f"{model}_run_*/*.json")
    files = glob.glob(pattern)

    for file in files:
        filename = os.path.basename(file)
        eval_type = filename.split('_', 1)[0]
        with open(file, 'r') as f:
            data = json.load(f)

        for key in data:
            if not key.endswith(':std'):
                metric = key
                std_key = f"{metric}:std"
                if std_key not in data:
                    raise ValueError(f"Missing {std_key} in {file}")
                value = data[metric]
                std = data[std_key]

                if eval_type not in results:
                    results[eval_type] = {}
                if metric not in results[eval_type]:
                    results[eval_type][metric] = []
                results[eval_type][metric].append((value, std))

    return results
