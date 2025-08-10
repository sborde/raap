

from utils.file_utils import load_customized_config
from triggered_firing.io_utils import list_input_files
from triggered_firing.runners import IndividualSpikeTypePTTHAnalysis

CONFIG_NAME = 'firing_activity_around_trigger'

if __name__ == "__main__":

    run_config = load_customized_config(CONFIG_NAME)
    analysis = IndividualSpikeTypePTTHAnalysis(run_config)

    input_files = list_input_files(run_config['data'], run_config['input'])
    for input_file in input_files:
        try:
            print(f"Processing {input_file.name}...", end=" ")
            analysis(input_file)
            print("done.")
        except Exception as e:
            print(
                f"Some error occurred during processing {input_file.name}: {e}"
            )
