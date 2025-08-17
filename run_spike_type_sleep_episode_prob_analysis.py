"""
This is the entry point for the spike type vs. sleep episode relationship
analysis. This script calculates the probabilities of spike types and sleep
episodes (prior and joint probability).
 - 2025. 08. 16. - initial version
"""

from workflows.episode_spiketype_relationship import SpikeEpisodeProbabilityAnalysis
from workflows.run_util.config import load_config
from workflows.run_util.io import list_input_files

CONFIG_NAME = 'run_spike_type_sleep_episode_prob_analysis'

if __name__ == "__main__":

    config = load_config(CONFIG_NAME)
    analysis = SpikeEpisodeProbabilityAnalysis(config)

    statistics = []
    input_files = list_input_files(config['data'], config['input'])
    for input_file in input_files:
        print(f"Processing {input_file.name}...")
        statistics.append(analysis.run(input_file))
        print("done.")

    if config['grouping']['do']:
        analysis.run_grouped(statistics)