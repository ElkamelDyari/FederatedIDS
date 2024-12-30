import os
import pandas as pd

def data_gathering():
    """
        Gathers and concatenates data from multiple CSV files into a single DataFrame.

        This function reads CSV files from a specified directory, concatenates them into a single pandas DataFrame, and returns the combined DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the concatenated data from all specified CSV files.

        Raises:
            FileNotFoundError: If any of the specified CSV files are not found in the directory.
            pd.errors.EmptyDataError: If any of the specified CSV files are empty.
            pd.errors.ParserError: If there is a parsing error while reading any of the CSV files.

        Example:
            full_data = data_gathering()
            print(full_data.head())
        """

    DIR_PATH = os.path.join('..', 'data', 'SrcData')

    FILE_NAMES = ["Monday-WorkingHours.pcap_ISCX.csv",
                  "Tuesday-WorkingHours.pcap_ISCX.csv",
                  "Wednesday-workingHours.pcap_ISCX.csv",
                  "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                  "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
                  "Friday-WorkingHours-Morning.pcap_ISCX.csv",
                  "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                  "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]

    df = [pd.read_csv(os.path.join(DIR_PATH, f), skipinitialspace=True) for f in FILE_NAMES]
    df = pd.concat(df, ignore_index=True)

    return df