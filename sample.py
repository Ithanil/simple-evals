import argparse
import subprocess
import json
import os
import glob

from sample_helpers import aggregate_results, save_results, tabulate_results

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

output_json = aggregate_results(results)
save_results(output_json, "results", args.model)
tabulate_results(output_json)
