import os
import pandas as pd

def data_gathering():
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