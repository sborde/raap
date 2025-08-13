
import pandas as pd

from triggered_firing.io_utils import list_input_files, load_customized_config
from phase_event_coupling.runners import IndividualSpikeTypePECAnalysis
from triggered_firing.group_summarizers import run_grouped_analysis


CONFIG_NAME = 'phase_event_coupling'

if __name__ == "__main__":

    run_config = load_customized_config(CONFIG_NAME)
    analysis = IndividualSpikeTypePECAnalysis(run_config)

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

    # if run_config['grouping']['do']:
    #     run_config['output']['suffix'] = analysis.suffix
    #     run_grouped_analysis(run_config)
