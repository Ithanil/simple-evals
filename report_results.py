import argparse

from sample_helpers import load_results_from_runs, aggregate_results, save_results, tabulate_results

def main():
    parser = argparse.ArgumentParser(description='Aggregate results from multiple runs')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing run results')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name for filtering results')
    args = parser.parse_args()

    results = load_results_from_runs(args.results_dir, args.model)
    results_json = aggregate_results(results)
    save_results(results_json, args.results_dir, args.model)
    tabulate_results(results_json)

if __name__ == "__main__":
    main()
