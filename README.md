Dataset link : https://www.unb.ca/cic/datasets/ids-2017.html

website link : https://frontend-app-service-fcd3hqdygrfxdee8.francecentral-01.azurewebsites.net/

![Description](https://github.com/ElkamelDyari/FederatedIDS/blob/master/Untitled%20video%20-%20Made%20with%20Clipchamp.gif)

dataset 

## Data Overview

| Column Name                  | Description                                                                                       |
|------------------------------|---------------------------------------------------------------------------------------------------|
| `Flow ID`                    | Unique identifier for the flow.                                                                   |
| `Source IP`                  | IP address of the source.                                                                         |
| `Source Port`                | Port number of the source.                                                                        |
| `Destination IP`             | IP address of the destination.                                                                    |
| `Destination Port`           | Port number of the destination.                                                                   |
| `Protocol`                   | Network protocol used (e.g., TCP, UDP, ICMP).                                                     |
| `Timestamp`                  | Time at which the packet was captured.                                                            |
| `Flow Duration`              | Duration of the network flow in microseconds.                                                     |
| `Total Fwd Packets`          | Total number of packets in the forward direction (from source to destination).                    |
| `Total Backward Packets`     | Total number of packets in the backward direction (from destination to source).                   |
| `Total Length of Fwd Packets`| Total length of all forward packets in bytes.                                                     |
| `Total Length of Bwd Packets`| Total length of all backward packets in bytes.                                                    |
| `Fwd Packet Length Max`      | Maximum length of a packet in the forward direction.                                              |
| `Fwd Packet Length Min`      | Minimum length of a packet in the forward direction.                                              |
| `Fwd Packet Length Mean`     | Mean length of packets in the forward direction.                                                  |
| `Fwd Packet Length Std`      | Standard deviation of packet length in the forward direction.                                     |
| `Bwd Packet Length Max`      | Maximum length of a packet in the backward direction.                                             |
| `Bwd Packet Length Min`      | Minimum length of a packet in the backward direction.                                             |
| `Bwd Packet Length Mean`     | Mean length of packets in the backward direction.                                                 |
| `Bwd Packet Length Std`      | Standard deviation of packet length in the backward direction.                                    |
| `Flow Bytes/s`               | Number of bytes per second for the flow.                                                          |
| `Flow Packets/s`             | Number of packets per second for the flow.                                                        |
| `Flow IAT Mean`              | Mean inter-arrival time between packets in the flow.                                              |
| `Flow IAT Std`               | Standard deviation of inter-arrival time between packets in the flow.                             |
| `Flow IAT Max`               | Maximum inter-arrival time between packets in the flow.                                           |
| `Flow IAT Min`               | Minimum inter-arrival time between packets in the flow.                                           |
| `Fwd IAT Total`              | Total inter-arrival time for packets in the forward direction.                                    |
| `Fwd IAT Mean`               | Mean inter-arrival time for packets in the forward direction.                                     |
| `Fwd IAT Std`                | Standard deviation of inter-arrival time for packets in the forward direction.                    |
| `Fwd IAT Max`                | Maximum inter-arrival time for packets in the forward direction.                                  |
| `Fwd IAT Min`                | Minimum inter-arrival time for packets in the forward direction.                                  |
| `Bwd IAT Total`              | Total inter-arrival time for packets in the backward direction.                                   |
| `Bwd IAT Mean`               | Mean inter-arrival time for packets in the backward direction.                                    |
| `Bwd IAT Std`                | Standard deviation of inter-arrival time for packets in the backward direction.                   |
| `Bwd IAT Max`                | Maximum inter-arrival time for packets in the backward direction.                                 |
| `Bwd IAT Min`                | Minimum inter-arrival time for packets in the backward direction.                                 |
| `Fwd PSH Flags`              | Number of times the PSH (Push) flag was set in the forward direction.                             |
| `Bwd PSH Flags`              | Number of times the PSH flag was set in the backward direction.                                   |
| `Fwd URG Flags`              | Number of times the URG (Urgent) flag was set in the forward direction.                           |
| `Bwd URG Flags`              | Number of times the URG flag was set in the backward direction.                                   |
| `Fwd Header Length`          | Total length of forward headers in bytes.                                                         |
| `Bwd Header Length`          | Total length of backward headers in bytes.                                                        |
| `Fwd Packets/s`              | Number of forward packets per second.                                                             |
| `Bwd Packets/s`              | Number of backward packets per second.                                                            |
| `Min Packet Length`          | Minimum length of all packets in the flow.                                                        |
| `Max Packet Length`          | Maximum length of all packets in the flow.                                                        |
| `Packet Length Mean`         | Mean length of all packets in the flow.                                                           |
| `Packet Length Std`          | Standard deviation of packet length in the flow.                                                  |
| `Packet Length Variance`     | Variance of packet length in the flow.                                                            |
| `FIN Flag Count`             | Number of times the FIN (Finish) flag was set.                                                    |
| `SYN Flag Count`             | Number of times the SYN (Synchronize) flag was set.                                               |
| `RST Flag Count`             | Number of times the RST (Reset) flag was set.                                                     |
| `PSH Flag Count`             | Number of times the PSH flag was set.                                                             |
| `ACK Flag Count`             | Number of times the ACK (Acknowledgment) flag was set.                                            |
| `URG Flag Count`             | Number of times the URG flag was set.                                                             |
| `CWE Flag Count`             | Count of CWE (Congestion Window Reduced) flags.                                                   |
| `ECE Flag Count`             | Count of ECE (Explicit Congestion Notification Echo) flags.                                       |
| `Down/Up Ratio`              | Ratio of download to upload packets.                                                              |
| `Average Packet Size`        | Average packet size across both forward and backward directions.                                  |
| `Avg Fwd Segment Size`       | Average segment size in the forward direction.                                                    |
| `Avg Bwd Segment Size`       | Average segment size in the backward direction.                                                   |
| `Fwd Header Length.1`        | Duplicate of `Fwd Header Length`, may be used for different analysis.                             |
| `Fwd Avg Bytes/Bulk`         | Average number of bytes in each forward bulk.                                                     |
| `Fwd Avg Packets/Bulk`       | Average number of packets in each forward bulk.                                                   |
| `Fwd Avg Bulk Rate`          | Average rate of data transmission in forward bulk.                                                |
| `Bwd Avg Bytes/Bulk`         | Average number of bytes in each backward bulk.                                                    |
| `Bwd Avg Packets/Bulk`       | Average number of packets in each backward bulk.                                                  |
| `Bwd Avg Bulk Rate`          | Average rate of data transmission in backward bulk.                                               |
| `Subflow Fwd Packets`        | Number of packets in the forward subflow.                                                         |
| `Subflow Fwd Bytes`          | Number of bytes in the forward subflow.                                                           |
| `Subflow Bwd Packets`        | Number of packets in the backward subflow.                                                        |
| `Subflow Bwd Bytes`          | Number of bytes in the backward subflow.                                                          |
| `Init_Win_bytes_forward`     | Initial window size in bytes in the forward direction.                                            |
| `Init_Win_bytes_backward`    | Initial window size in bytes in the backward direction.                                           |
| `act_data_pkt_fwd`           | Number of forward direction data packets.                                                         |
| `min_seg_size_forward`       | Minimum segment size in the forward direction.                                                    |
| `Active Mean`                | Mean time the connection was active before going idle.                                            |
| `Active Std`                 | Standard deviation of the time the connection was active before going idle.                       |
| `Active Max`                 | Maximum time the connection was active before going idle.                                         |
| `Active Min`                 | Minimum time the connection was active before going idle.                                         |
| `Idle Mean`                  | Mean time the connection was idle before becoming active.                                         |
| `Idle Std`                   | Standard deviation of the time the connection was idle before becoming active.                    |
| `Idle Max`                   | Maximum time the connection was idle before becoming active.                                      |
| `Idle Min`                   | Minimum time the connection was idle before becoming active.                                      |
| `Label`                      | Label assigned to the traffic flow, indicating whether it is normal or an attack (e.g., DoS).     |
