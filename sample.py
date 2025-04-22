import argparse
import subprocess
import glob
import json
import math
import os
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

parser = argparse.ArgumentParser(description='Run LLM evaluation multiple times and aggregate results.')
parser.add_argument('--api-url', required=True, help='OpenAI base URL')
parser.add_argument('--api-key', required=True, help='OpenAI API key')
parser.add_argument('--model', required=True, help='Model name')
parser.add_argument('--num-samples', type=int, required=True, help='Number of examples to sample')
parser.add_argument('--num-runs', type=int, required=True, help='Number of runs')
args = parser.parse_args()

results = {}
if not os.path.exists("results"):
    os.mkdir("results")

for run in range(args.num_runs):
    print(f"Run {run + 1}/{args.num_runs}")
    env = os.environ.copy()
    env['OPENAI_BASE_URL'] = args.api_url
    env['OPENAI_API_KEY'] = args.api_key
    subprocess.run(
        ['python3', '-m', 'simple-evals.simple_evals', '--model', args.model, '--examples', str(args.num_samples)],
        env=env,
        check=True
    )
    run_results_path = f"results/{args.model}_run_{run}"
    if not os.path.exists(run_results_path):
        os.mkdir(run_results_path)
    pattern = f"results/*_{args.model}.json"
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
        os.rename(file, f"{run_results_path}/{os.path.basename(file)}")

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

output_filename = f"results/{args.model}_results.json"
with open(output_filename, 'w') as f:
    json.dump(output_json, f, indent=2)
print(f"Results saved to {output_filename}")

table = []
for eval_type in output_json:
    if 'score' in output_json[eval_type]:
        avg_score = output_json[eval_type]['score']
        std_score = output_json[eval_type]['score:std']
        table.append([eval_type, avg_score, std_score])

table.sort(key=lambda x: x[0])
print("\nScore averages and stds:")
print(tabulate(table, headers=['Evaluation Type', 'Average Score', 'Std Dev'], tablefmt='orgtbl'))
