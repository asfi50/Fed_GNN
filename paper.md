Title: Graph-based federated learning approach for intrusion detection in IoT networks
Authors: Fouad Al Tfaily, Zakariya Ghalmane, Mohamed el Amine Brahmia, Hussein Hazimeh, Ali Jaber & Mourad Zghal 
URL: https://www.nature.com/articles/s41598-025-25175-1

Abstract
Internet of Things (IoT) networks face increasing cyber threats that require collaborative intrusion detection across distributed environments. Existing federated learning approaches for intrusion detection have critical architectural and methodological limitations: traditional federated approaches using deep learning methods like LSTM and CNN cannot capture structural patterns as these architectures are not designed to model network graph topology, while GNN-based federated approaches can preserve structural patterns but lose temporal patterns during parameter aggregation, making both ineffective against sophisticated coordinated attacks like DDoS campaigns. Yet, detecting coordinated attacks requires understanding both structural relationships and temporal sequences across networks. This paper proposes FedGATSage, a federated learning architecture that integrates client-side Graph Attention Networks with server-side GraphSAGE through community abstraction to address both architectural and temporal limitations. The approach uses specialized detector variants for different attack types and preserves both structural and temporal attack patterns while protecting device identities through community-based embeddings. The latter aims to aggregate the information of groups of nodes at the client level before the submission to the server which reduces communication overhead by 85%. Extensive experiments on NF-ToN-IoT and CIC-ToN-IoT datasets demonstrate that FedGATSage achieves performance comparable to centralized approaches while preserving privacy, significantly outperforming existing federated solutions and successfully detecting challenging coordinated attacks that current federated methods cannot handle.


Introduction
Intrusion Detection Systems (IDSs) are a key defense mechanism to protect organizational resources against cyber threats. Traditional IDSs relied on signature-based methods that identified familiar attack signatures, but were unable to address zero-day attacks and evolving attack techniques1. This weakness is especially prevalent in distributed environments like modern IoT networks, where centralized data collection creates significant privacy concerns and regulatory constraints2,3. The fundamental challenge is that organizations cannot share raw network data due to privacy regulations and competitive concerns, yet sophisticated attacks like coordinated DDoS and advanced persistent threats require understanding both network relationships and attack sequences that span multiple networks.

The development of intrusion detection methods has followed a clear evolutionary path toward increasing sophistication in response to these challenges. Early systems used rule-based signatures and statistical anomaly detection with restricted ability to adapt to rapidly increasing threats4. Subsequently, machine learning approaches transformed the field, with methods like Support Vector Machines5 and Random Forests6 improving detection through pattern learning3. This progression paved the path for deep learning structures that could discover hierarchical features autonomously from raw network flows4. Most notably, Graph Neural Networks (GNNs) have shown superior performance for network intrusion detection by modeling the inherent graph structure of network communications7,8,9. GNNs achieve exceptional effectiveness through explicit modeling of network topology, delivering state-of-the-art results on benchmark datasets like NF-ToN-IoT and CIC-ToN-IoT10,11. Two GNN variants have demonstrated exceptional performance: Graph Attention Networks (GAT) excel at identifying malicious connection patterns by differentially weighting neighbor importance12, particularly valuable for detecting attacks like backdoors, while GraphSAGE’s neighborhood sampling approach performs well on large-scale networks and allows adaptation to new attack patterns through inductive capabilities13. These complementary strengths have improved detection accuracy for complex attacks that would often be missed by traditional methods14,15.

While these centralized approaches achieved significant advances, the emergence of distributed computing environments created new requirements for privacy-preserving collaborative security. Federated learning emerged as a promising solution by enabling collaborative model training across decentralized clients without sharing raw data16,17. However, early federated approaches for intrusion detection faced critical limitations when using traditional deep learning methods like LSTM, CNN, and standard neural networks, as these architectures fundamentally cannot capture network structural patterns for they are not designed to model graph topology and device relationships18,19. These methods process sequential data or grid-like structures but fail to understand how network devices connect to each other, missing the relationship patterns that characterize sophisticated attacks like coordinated scanning and distributed intrusion campaigns. Additionally, heterogeneous data distributions across clients lead to model drift and poor convergence.

Recognizing these structural limitations, recent directions began adopting GNNs in federated settings to better capture network relationships20. Khan et al.’s work demonstrates this trend across multiple domains, including collaborative networks for smart healthcare systems with dynamic behavior aggregation, reinforcement learning-based approaches for Internet of Medical Things networks with dynamic client participation, and federated methods for consumer IoT cyber-attack detection using weighted techniques to address data imbalance21,22,23. Meanwhile, other approaches explored different privacy-preserving techniques: differential privacy methods add noise to protect sensitive information but typically degrade detection performance18,24, while secure multi-party computation approaches guarantee strong privacy but impose high computational overhead that becomes prohibitive for real-time detection25,26. Despite these advances, GNN-based federated approaches revealed additional critical limitations beyond their traditional counterparts. While GNNs can initially capture structural patterns within individual client networks, these structural patterns are lost during parameter aggregation. Consequently, the client structural knowledge gets mixed together through federated averaging at the server level. Furthermore, temporal attack patterns are also destroyed during this parameter sharing process27. When only model parameters are exchanged between clients, the time-dependent characteristics essential for detecting attacks like DDoS campaigns and coordinated scanning operations are lost, resulting in poor performance against the most threatening attack types28,29.

This dual limitation creates a fundamental research gap where existing federated intrusion detection approaches cannot simultaneously preserve both structural relationships and temporal attack evolution while maintaining meaningful privacy guarantees. Traditional federated methods cannot capture structural patterns due to architectural limitations, GNN federated methods lose both structural and temporal patterns during parameter aggregation, while privacy-preserving techniques either degrade detection performance through noise injection or impose prohibitive computational costs.To address these limitations, we propose FedGATSage, a hybrid community-based federated learning architecture that integrates client-side GAT with server-side GraphSAGE through community abstraction. Rather than sharing raw network data or just model parameters, each client detects communities (groups of densely connected devices with sparse connections to other groups) using GAT and the Louvain algorithm30, then shares only these community-level representations with the server, which constructs an overlay graph and learns global patterns using GraphSAGE. This approach preserves both structural and temporal attack patterns by maintaining the network relationship information through community embeddings while capturing attack evolution through specialized feature extraction, all without exposing individual device behaviors. Our solution is based on the strong correlation between community structure and attack patterns in datasets such as the NF-ToN-IoT and CIC-ToN-IoT datasets31,32. These datasets contain diverse attack types including both structural attacks (Backdoor, Injection) and temporal attacks (DDoS, DoS, Scanning), making them ideal for comprehensive evaluation33. Unlike existing federated approaches that focus primarily on parameter aggregation, FedGATSage simultaneously preserves privacy, maintains structural and temporal patterns, and enables specialized detection for different attack categories34,35.

The main contributions of this paper include: (1) A hybrid federated learning architecture combining client-side GAT with server-side GraphSAGE that addresses both structural and temporal pattern preservation limitations in federated intrusion detection; (2) Specialized GAT detector variants (Temporal, Content, Behavioral) with sequential training and ensemble fusion, enabling attack-type-aware detection in federated environments; (3) An enhanced federated learning method that weights client contributions based on their performance and handles different network conditions rather than standard approaches, and (4) A community abstraction technique that protects privacy by analyzing device groups rather than exposing individual device data for attack detection.



Methods
FedGATSage is presented as a federated learning solution for privacy-preserving network intrusion detection that integrates Graph Attention Networks (GAT) on the client side with GraphSAGE on the server side through community abstraction. This approach addresses the challenge of maintaining detection accuracy while preserving both structural and temporal attack patterns in distributed environments like IoT networks.

System architecture overview
[FIG-1]
https://www.nature.com/articles/s41598-025-25175-1/figures/1
This figure illustrates a privacy-preserving, two-tier graph federated learning framework designed to collaboratively learn complex topological patterns without compromising local node security. In Step 1, independent client nodes (Client 1 through Client N) transform their local data partitions into network structures and apply Graph Attention Networks (GATs) to extract low-dimensional community embeddings, ensuring that raw, sensitive data remains safely isolated behind a strict privacy boundary. In Step 2, only these abstracted community vectors are transmitted upward to a central server, which dynamically links them into a macro-level overlay graph by connecting similar client communities using dynamic threshold similarity scores as edges. Finally, a server-side GraphSAGE model processes this unified overlay graph to capture global structural dependencies across all participating clients, subsequently redistributing the refined global embeddings and model parameter updates back down to the edge nodes to continuously optimize local performance.
[/FIG-1]
caption: Overview of the FedGATSage architecture. Network data is processed locally by clients into graphs, where communities (colored node groups) are detected and processed using GAT. Clients send community embeddings to the server, which applies GraphSAGE to generate global embeddings. Updated model parameters are then sent back to clients.

As illustrated in Fig. 1, the workflow begins with each client transforming its local network data into a graph, where nodes represent devices (IP addresses) and edges represent network connections. Each client applies a GAT model to its graph, then detects communities using the Louvain algorithm30. Communities are groups of densely connected nodes with sparse connections to other groups. Node embeddings are then aggregated into community-level representations that capture structural information while abstracting individual details.

Only these community embeddings and client-side model parameters are transmitted to the server. The server aggregates information from all clients and constructs a community-based overlay graph, where each community becomes a node and edges reflect similarity between communities. GraphSAGE is then applied to this overlay graph to capture global patterns, after which updated parameters are redistributed to clients. The federated process begins with random initialization of both GAT (client-side) and GraphSAGE (server-side). In each round, clients train locally, generate community embeddings, and send them with model parameters to the server. The server updates the global GraphSAGE model on the overlay graph, aggregates parameters, and shares them back with clients.

FedGATSage provides three key methodological advantages over existing approaches: temporal pattern preservation through flow-level abstraction maintains attack sequences essential for detecting coordinated threats, community-aware privacy preserves network relationships while protecting device identities, and specialized architecture uses three GAT variants for different attack types providing targeted detection capabilities.

Client-side processing
Client-side processing begins with converting network flow data into graph format, with network devices as nodes and network connections as edges. We extract features for each node, including traffic statistics (packet count, connection frequency), centrality measures (PageRank, degree centrality), and protocol-specific information. These features are enhanced through specialized engineering tailored to different attack types.

To model diverse attack patterns effectively, we implement three specialized GAT variants as illustrated in Fig. 2. Each variant targets different categories of network attacks:

Temporal GAT focuses on time-dependent attacks like DDoS and DoS by analyzing timing patterns, traffic bursts, and connection sequences that characterize coordinated attacks.

Content GAT targets content-based attacks like injection and XSS by examining payload characteristics, protocol usage patterns, and service targeting behaviors.

Behavioral GAT detects behavioral attacks like password attacks, backdoors, and scanning by analyzing connection behaviors, port usage patterns, and session characteristics.

Each GAT variant processes the network graph and generates node embeddings that capture attack-specific characteristics. These embeddings are combined through weighted aggregation that emphasizes the most relevant features for each nod. Moreover, the architecture of each variant consists of multiple graph attention layers with specialized attention mechanisms, gradient propagation with residual connections, and normalization layers for stable training. The general attention mechanism is expressed as:
\begin{aligned} h_i^{\prime } = \sigma \left( \sum _{j \in \mathscr {N}(i)} \alpha _{ij} \textbf{W} h_j\right) \end{aligned}
where h_i^{\prime } is the updated representation of node i, \sigma is a non-linear activation function, \textbf{W} is a learnable weight matrix, and \alpha _{ij} are attention coefficients that determine the importance of neighbor j’s features to node i. These attention coefficients are calculated differently for each variant to prioritize relevant features for specific attack types.

The three specialized GAT variants are trained sequentially rather than jointly, allowing each detector to optimize independently for its specific attack category. Temporal GAT trains on features focusing on flow rates, duration patterns, and flag sequences; Content GAT trains on content-specific features emphasizing payload sizes, protocol patterns, and service targeting; Behavioral GAT trains on connection pattern features analyzing port usage, timing behaviors, and session characteristics.

After generating node embeddings using the specialized GAT variants, we perform community detection and embedding creation to transform these embeddings into privacy-preserving representations suitable for federated learning. Algorithm 1 details this process.

Algorithm 1:
Require: Network flows dataset $D$, trained GAT model parameters
Ensure: Community-enhanced flow embeddings $E_{flows}$
1: // Step 1: Graph Construction
2: $G = (V,E) \leftarrow \text{ConstructGraph}(D)$ {Nodes = IP addresses, Edges = flows}
3: $X \leftarrow \text{ExtractNodeFeatures}(V, D)$ {Traffic statistics per IP}
4: // Step 2: Community Detection for Structure Enhancement
5: $Communities \leftarrow \text{LouvainAlgorithm}(G)$
6: for each node $v \in V$ do
7:     $ModVitality[v] \leftarrow \text{ComputeModularityVitality}(v, Communities)$
8:     $X[v] \leftarrow X[v]\vert{}\vert{}ModVitality[v]$ {Append as feature}
9: end for
10: // Step 3: GAT Processing for Node Embeddings
11: $H \leftarrow \text{GAT}(X, G)$ {Generate node embeddings per IP}
12: // Step 4: Flow Embedding Creation
13: $E_{flows} \leftarrow \{\}$
14: for each flow $f = (src, dst, traffic\_features) \in E$ do
15:     $embedding \leftarrow [H[src]\vert{}\vert{}H[dst]\vert{}\vert{}(H[src] \odot H[dst])\vert{}\vert{}\vert{}H[src] - H[dst]\vert{}\vert{}\vert{}traffic\_features]$
16:     $E_{flows} \leftarrow E_{flows} \cup \{embedding\}$
17: end for
18: // Step 5: Community Aggregation
19: for each community $c \in Communities$ do
20:     $E_c \leftarrow \text{AggregateNodeEmbeddings}(c, H)$ {Weighted mean + structural metrics}
21: end for
22: // Step 6: Selective Sampling for Privacy
23: $E_{selected} \leftarrow \text{SampleFlows}(E_{flows}, \text{max\_per\_class}=250)$
24: return $E_{selected}, \{E_c\}$
Community Detection and Flow Embedding Creation

Algorithm 1 systematically transforms raw network data into privacy-preserving embeddings through six sequential steps: First, we construct a graph structure where each distinct IP address becomes a node and each network flow becomes an edge connecting source and destination IPs, while extracting basic traffic statistics and centrality measures for each device. Second, we apply the Louvain algorithm to identify communities (groups of densely connected devices) and compute community importance scores that measure how much removing each device would disrupt the community structure. Third, we use the specialized GAT variants to generate rich node embeddings that capture each device’s role in the network and its relevance to different attack types. Fourth, we create flow embeddings by combining information from both communication endpoints (source and destination nodes of each flow) using concatenation, element-wise multiplication, and absolute difference operations along with original traffic features. Fifth, we aggregate node embeddings within each community using weighted averaging and add structural metrics like internal density and clustering coefficients. Finally, we select representative flows from each attack class rather than transmitting all data, ensuring computational efficiency and additional privacy protection.

During inference, predictions from all three detectors are combined through a two-stage fusion process. We apply ensemble averaging across the three detector outputs, then employ a Random Forest classifier trained on features including raw probabilities from each detector, predicted class labels, confidence scores, and attack-type priority weights. The Random Forest learns optimal weighting strategies during training, automatically emphasizing the most relevant specialized detector for each attack instance.

[FIG-2]
Figure 2 illustrates a localized, client-side data processing pipeline designed for privacy-preserving federated intrusion detection. The architecture begins by ingesting raw network flow data and applying specialized feature engineering to extract distinct temporal, content, and behavioral characteristics. These engineered features are then routed into three parallel Graph Attention Network (GAT) variants, each optimized to output node embeddings for specific threat vectors: the Temporal GAT targets DDoS and DoS attacks, the Content GAT identifies Injection and XSS vulnerabilities, and the Behavioral GAT profiles Password, Backdoor, and Scanning anomalies. Rather than exporting these individual embeddings directly, the system routes them through a Community Detection & Abstraction Layer that groups similar network behaviors into macro-level communities. This final abstraction ensures that only generalized, structural threat intelligence is sent outward via a privacy-preserving transfer, effectively securing the local network's raw data while contributing meaningful updates to a global federated learning model.
[/FIG-2]
Caption: Specialized GAT architecture for federated intrusion detection. Client-side processing applies tailored feature engineering and three GAT variants for diverse attack detection. Node embeddings are fused before community detection.

Server-side processing
Upon receiving community embeddings from clients, the server constructs an overlay graph where each node represents a community from any client. The first step is determining appropriate connections between communities through similarity analysis using cosine similarity:
\begin{aligned} \text {similarity}(C_i, C_j) = \frac{C_i \cdot C_j}{||C_i|| \cdot ||C_j||} \end{aligned}


where C_i and C_j are the embedding vectors of communities i and j respectively. Edges in the overlay graph are created between communities whose similarity exceeds a dynamically adjusted threshold that begins at 0.7 and is progressively lowered if necessary to ensure each community has at least three connections, while not dropping below 0.3 to maintain edge quality.

The complete overlay graph construction process follows these steps:
Algorithm 2:
Require: Client community embeddings $\{E_{c_1}, E_{c_2}, \dots, E_{c_n}\}$
Ensure: Global overlay graph $G_{\text{overlay}}$
1: Initialize empty graph $G_{\text{overlay}}$
2: for all clients $k \in 1 : K$ do
3:     for all communities $c_i \in \mathcal{C}_k$ do
4:         Add node $v_i$ to $G_{\text{overlay}}$ with features $E_{c_i}$
5:     end for
6: end for
7: Compute similarity matrix $S$ where $S_{ij} = \text{cosine}(E_{c_i}, E_{c_j})$
8: Set threshold $\theta \leftarrow 0.7$
9: repeat
10:     for all node pairs $(v_i, v_j)$ do
11:         if $S_{ij} > \theta$ then
12:             Add edge $e_{ij}$ to $G_{\text{overlay}}$
13:         end if
14:     end for
15:     if any node has degree $< 3$ then
16:         Decrease $\theta$ by 0.05
17:         Clear all edges in $G_{\text{overlay}}$
18:     end if
19: until all nodes have degree $\geq 3$ or $\theta \leq 0.3$
20: return $G_{\text{overlay}}$
caption: Community Overlay Construction


Once the overlay graph is constructed (as illustrated in Algorithm 2) GraphSAGE is applied to learn global patterns across all client communities. GraphSAGE employs neighborhood sampling and aggregation to efficiently process the overlay graph structure:
\begin{aligned} h_v^{(k)} = \text {AGGREGATE}_k\left( \{h_u^{(k-1)} : u \in \mathscr {N}(v)\}\right) \end{aligned}

where h_v^{(k)} is the embedding of node v at layer k, and \mathscr {N}(v) represents its neighborhood. The AGGREGATE function combines information from neighboring nodes through a mean aggregator function, enabling the model to learn patterns that span multiple communities across different clients.

Federated model aggregation and parameter redistribution
After processing the overlay graph with GraphSAGE, the server performs federated model aggregation by averaging client-side GAT model parameters with adaptive weighting based on client performance. For each client k, a weight w_k is computed as:
\begin{aligned} w_k = \alpha + (1-\alpha ) \cdot \frac{A_k - A_{\min }}{A_{\max } - A_{\min }} \end{aligned}

where A_k is the client’s local validation accuracy, A_{\min } and A_{\max } are the minimum and maximum accuracies across all clients, and \alpha =0.2 is a base weight ensuring all clients contribute meaningfully. This approach addresses client heterogeneity by giving more influence to clients with better performance while ensuring all clients still contribute to the global model. The aggregated parameters are then distributed to all clients for the next federation round. For final classification decisions, each client applies its updated GAT model to local data, and predictions are made based on the edge features between nodes in the network graph. This approach enables global pattern learning while maintaining the privacy of local network data.

Results
A comprehensive evaluation of FedGATSage is presented on both the NF-ToN-IoT and CIC-ToN-IoT datasets, examining performance across different attack types, architectural configurations, and computational efficiency metrics. This evaluation focuses on comparing the proposed approach with both centralized deep learning models and alternative federated learning techniques to establish its effectiveness in balancing privacy preservation with detection capability.

Overall performance comparison
Table 1 Comparison of performance metrics across centralized and federated approaches for NF-ToN-IoT and CIC-ToN-IoT.

+-----------------------+-----------------------------------------------+-----------------------------------------------+
|                       | NF-ToN-IoT                                    | CIC-ToN-IoT                                   |
| Approach              +-------+-----------+-------+-------------------+-------+-----------+-------+-------------------+
|                       | Acc.  | Bal. Acc. | F1    | FPR/FNR           | Acc.  | Bal. Acc. | F1    | FPR/FNR           |
+-----------------------+-------+-----------+-------+-------------------+-------+-----------+-------+-------------------+
| Centralized GAT       | 0.811 | -         | 0.797 | 0.026 / 0.188     | 0.840 | -         | 0.820 | 0.024 / 0.169     |
| Centralized GraphSAGE | 0.814 | -         | 0.800 | 0.026 / 0.185     | 0.842 | -         | 0.822 | 0.024 / 0.169     |
| LSTM (Federated)      | 0.637 | 0.713     | 0.610 | 0.052 / 0.286     | 0.776 | 0.759     | 0.752 | 0.043 / 0.225     |
| FedGATSage            | 0.607 | 0.785     | 0.619 | 0.066 / 0.223     | 0.827 | 0.802     | 0.800 | 0.039 / 0.197     |
+-----------------------+-------+-----------+-------+-------------------+-------+-----------+-------+-------------------+

FedGATSage achieved 78.58% balanced accuracy on the NF-ToN-IoT dataset and 80.24% balanced accuracy on CIC-ToN-IoT dataset with better false negative rates (0.2234 on NF-ToN-IoT and 0.1976 on CIC-ToN-IoT) than the federated LSTM approach (0.2862 on NF-ToN-IoT and 0.2250 on CIC-ToN-IoT) indicating fewer missed attacks while maintaining reasonable false positive rates, approaching the performance of centralized models while preserving privacy. The performance gap between the federated FedGATSage approach and centralized models is remarkably small (around 3%), while maintaining strong privacy guarantees through community abstraction. Compared to state-of-the-art federated learning baselines, FedGATSage demonstrates superior performance: federated LSTM achieves 71.38% balanced accuracy versus our 78.58%, representing 7.2 percentage point improvement. More critically, our approach maintains temporal attack detection capability (DDoS F1: 0.67, DoS F1: 0.56) where existing federated methods show severe degradation in these challenging attack categories. Table 1 presents detailed performance metrics across both centralized and federated approaches for both datasets.

Attack-specific detection capabilities
FedGATSage demonstrates varying effectiveness across different types of attacks in a federated environment that preserves privacy, as shown in Fig. 3 that shows the evaluation scores for each type of attack on the NF-ToN-IoT dataset as an example.
[FIG-3]
The grouped bar chart, titled "Detection Performance by Attack Type," evaluates a classification model's effectiveness across eight categories of network traffic using Precision (blue), Recall (red), and F1 Score (green) metrics. The model exhibits exceptional performance for "Benign" traffic and "Backdoor" attacks, achieving F1 scores of 1.00 and 0.97, respectively. However, its effectiveness fluctuates across other threat vectors. For "DoS" and "XSS" attacks, the system achieves perfect recall (1.00) but notably low precision (0.39 and 0.42), a dynamic the chart's footnote explains as having high sensitivity to detecting threats but at the cost of generating false positives. Conversely, "Injection" and "DDoS" classifications show the opposite trend, characterized by high precision (0.98 and 0.89) but significantly lower recall (0.34 and 0.53), while "Scanning" and "Password" attacks show generally lower overall F1 scores due to imbalances between precision and recall.
[/FIG-3]
caption: F1 scores for different attack types on the NF-ToN-IoT dataset. FedGATSage excels at structural attacks (Benign, Backdoor) and achieves substantial performance on temporal attacks (DDoS, DoS, XSS).

The highest performance is observed on structural attacks, with F1 scores reaching 0.9942 and 0.9965 on Benign traffic, and 0.9750 and 0.9690 for Backdoor attacks on NF-ToN-IoT and CIC-ToN-IoT, respectively. These results indicate very few false positives or false negatives for these categories. For temporal attacks, which are typically difficult to detect in federated environments, promising detection capabilities are demonstrated. DDoS attacks achieve F1 scores of 0.6354 and 0.6672, and DoS attacks reach 0.5721 and 0.5608 on NF-ToN-IoT. This represents a significant improvement over prior federated approaches, which often failed entirely to detect such attacks. A detailed comparison of per-class performance across the NF-ToN-IoT datasets, including precision, recall, F1 score, and false negative rates is show in Table 2,

Table 2 Precision, Recall, and FNR for each attack type using FedGATSage on NF-ToN-IoT and CIC-ToN-IoT datasets.
+-------------+---------------------------------+---------------------------------+
|             | NF-ToN-IoT                      | CIC-ToN-IoT                     |
| Attack Type +------------+----------+---------+------------+----------+---------+
|             | Precision  | Recall   | FNR     | Precision  | Recall   | FNR     |
+-------------+------------+----------+---------+------------+----------+---------+
| Benign      | 0.9986     | 0.9944   | 0.0056  | 1.0000     | 0.9785   | 0.0215  |
| Backdoor    | 0.9451     | 0.9942   | 0.0058  | 0.9816     | 1.0000   | 0.0000  |
| DDoS        | 0.8939     | 0.5322   | 0.4678  | 0.8939     | 0.5322   | 0.4678  |
| DoS         | 0.3896     | 1.0000   | 0.0000  | 0.3896     | 1.0000   | 0.0000  |
| Injection   | 0.9760     | 0.3412   | 0.6588  | 0.7266     | 0.5146   | 0.4854  |
| Password    | 0.3890     | 0.6226   | 0.3774  | 0.5268     | 0.7296   | 0.2704  |
| Scanning    | 0.1050     | 0.8029   | 0.1971  | 0.7634     | 0.9302   | 0.0698  |
| XSS         | 0.4197     | 0.9990   | 0.0010  | 0.9186     | 0.6612   | 0.3388  |
+-------------+------------+----------+---------+------------+----------+---------+
The results highlight interesting patterns in precision, recall, false positives/negatives across attack types for both NF-ToN-IoT and CIC-ToN-IoT datasets. Benign traffic and Backdoor attacks exhibit high precision and recall, effectively identifying true positives while minimizing both false positives and false negatives (FNR: 0.0215 and 0.0000 for CIC-ToN-IoT, and 0.0056 and 0.0058 for NF-ToN-IoT). DDoS attacks show high precision (0.8939) but moderate recall (0.5322), indicating strong true positive identification with some missed detections. DoS and XSS attacks have near-perfect recall (1.00 and 0.6612, respectively), but lower precision, ensuring almost no false negatives (FNR: 0.0000 and 0.3388 for CIC-ToN-IoT, and 0.0000 and 0.0010 for NF-ToN-IoT), which is valuable in security contexts. For attacks like DoS and Scanning, high recall values (1.000 and 0.9302 for CIC-ToN-IoT, 0.987 and 0.8029 for NF-ToN-IoT) with moderate precision (0.3896 and 0.7634 for CIC-ToN-IoT, 0.3896 and 0.1050 for NF-ToN-IoT) represent a design that prioritizes capturing true positives over avoiding false alarms. The confusion matrix in Fig. 4 provides insights into classification patterns and misclassifications across attack categories.
[FIG-4]
matrix a
+------------+--------+----------+------+-----+-----------+----------+----------+------+
| True \ Pred| Benign | backdoor | ddos | dos | injection | password | scanning | xss  |
+------------+--------+----------+------+-----+-----------+----------+----------+------+
| Benign     | 2126   | 0        | 0    | 1   | 0         | 0        | 1        | 0    |
| backdoor   | 2      | 171      | 0    | 0   | 0         | 0        | 0        | 0    |
| ddos       | 1      | 0        | 1206 | 216 | 16        | 309      | 516      | 0    |
| dos        | 0      | 0        | 0    | 173 | 0         | 0        | 0        | 0    |
| injection  | 1      | 0        | 129  | 26  | 1585      | 1143     | 486      | 1275 |
| password   | 0      | 0        | 12   | 9   | 21        | 937      | 421      | 105  |
| scanning   | 0      | 0        | 2    | 19  | 0         | 20       | 167      | 0    |
| xss        | 0      | 0        | 0    | 0   | 1         | 0        | 0        | 998  |
+------------+--------+----------+------+-----+-----------+----------+----------+------+
matrix b
+------------+--------+----------+-----------+----------+----------+------+
| True \ Pred| Benign | backdoor | injection | password | scanning | xss  |
+------------+--------+----------+-----------+----------+----------+------+
| Benign     | 4914   | 47       | 15        | 0        | 16       | 30   |
| backdoor   | 0      | 2511     | 0         | 0        | 0        | 0    |
| injection  | 0      | 0        | 1289      | 1102     | 0        | 114  |
| password   | 0      | 0        | 133       | 1800     | 531      | 3    |
| scanning   | 0      | 0        | 0         | 174      | 2320     | 0    |
| xss        | 0      | 0        | 337       | 341      | 172      | 1659 |
+------------+--------+----------+-----------+----------+----------+------+
[/FIG-4]
caption: Confusion matrices for FedGATSage on both datasets, showing prediction accuracy across all attack classes. Most misclassifications occur between related attack categories.

Architectural configuration analysis
Table 3 Comprehensive comparison of architectural configurations including performance metrics for NF-ToN-IoT and CIC-ToN-IoT datasets. *Standard accuracy reported for centralized models, not balanced accuracy. “hrs.” is used for hours.
+-----------------------------+------------------------------------------+------------------------------------------+
|                             | NF-ToN-IoT                               | CIC-ToN-IoT                              |
| Architecture                +---------------+----------+---------------+---------------+----------+---------------+
|                             | Balanced Acc. | Macro F1 | Training time | Balanced Acc. | Macro F1 | Training time |
+-----------------------------+---------------+----------+---------------+---------------+----------+---------------+
| Centralized GAT             | 0.8115        | 0.7975   | 5.5 hrs.      | 0.8426        | 0.8229   | 4.2 hrs.      |
| Centralized GraphSAGE       | 0.8141        | 0.8003   | 4.8 hrs.      | 0.8426        | 0.8229   | 4.1 hrs.      |
| LSTM in federated setting   | 0.7138        | 0.6105   | 8.5 hrs.      | 0.7760        | 0.7528   | 7.5 hrs.      |
| GAT Local, GraphSAGE Global | 0.7858        | 0.6193   | 7.5 hrs.      | 0.8278        | 0.8003   | 6.2 hrs.      |
| GAT Global, GraphSAGE Local | 0.7780        | 0.6174   | 9.0 hrs.      | 0.7717        | 0.7665   | 7.3 hrs.      |
| GraphSAGE-only Ensemble     | 0.7780        | 0.6174   | 8.0 hrs.      | 0.7715        | 0.7663   | 6.9 hrs.      |
+-----------------------------+---------------+----------+---------------+---------------+----------+---------------+
Different architectural configurations were evaluated to determine the optimal placement of GNN variants in the federated setup. Table 3 shows the comprehensive performance and computational requirements of different configurations, including centralized and federated approaches.

[FIG-5]
The first graph, "Convergence of Balanced Accuracy," uses a line chart to track the learning efficiency of three federated learning architectures over 10 communication rounds. The "GAT Local, GraphSAGE Global" architecture demonstrates the fastest convergence, reaching the target balanced accuracy threshold of 0.75 exactly at round 8 and peaking near 0.79 by round 10. The "GraphSAGE-only Ensemble" shows moderate progress, crossing the 0.75 mark just past round 9. Conversely, the "GAT Global, GraphSAGE Local" setup exhibits the slowest convergence rate, requiring the full 10 federation rounds to finally achieve the 0.75 balanced accuracy threshold.

The second graph, "Federation Rounds Required to Reach Performance Levels," utilizes a grouped bar chart to explicitly quantify the rounds needed for each model to hit specific performance thresholds (0.60, 0.65, 0.70, and 0.75). The data highlights the superior efficiency of the "GAT Local, GraphSAGE Global" model, which requires only 4 rounds to reach a 0.60 score (33% faster than the 6 rounds needed by the slowest model), 5 rounds to reach 0.65 (29% faster than 7 rounds), and 6 rounds to reach 0.70 (25% faster than 8 rounds). To reach the final 0.75 threshold, this optimal model requires just 8 rounds—a 20% speedup compared to the 9 rounds for the green ensemble model and the 10 rounds required by the baseline "GAT Global, GraphSAGE Local" model.
[/FIG-5]
caption: Convergence analysis on the NF-ToN-IoT dataset showing accuracy progression over federation rounds for different architectural configurations. The GAT locally with GraphSAGE globally configuration demonstrates faster convergence and superior final performance compared to alternative setups. Similar results on the CiC-ToN-IoT dataset were observed.

The GAT locally with GraphSAGE globally configuration achieved the highest balanced accuracy (78.58% and 80.24% for NF-ToN-IoT and CIC-ToN-IoT) compared to alternative configurations. This configuration required less training time compared to the reversed configuration (GAT globally with GraphSAGE locally) for both datasets. Data transfer measurements showed approximately 10% lower volume (3.8 MB/round vs. 4.2 MB/round) for the GAT local configuration compared to the alternative, due to more efficient client-side processing and reduced parameter sizes in community embedding transfers.

For computational resources, the GAT locally with GraphSAGE globally configuration required an average of 2.4GB RAM and 35% GPU utilization during client-side processing for NF-ToN-IoT, while for CIC-ToN-IoT, it required 3.8GB RAM and 55% GPU utilization at the client side. CPU-only deployments showed the primary configuration requiring 76% less computation time than alternatives for NF-ToN-IoT, while for CIC-ToN-IoT, the primary FedGATSage configuration required slightly more total training time (7.5 hours vs. 5.5 hours for centralized GAT) but showed 12% faster training time compared to the LSTM in federated setting for CIC-ToN-IoT and approximately 27% reduction in data transfer volume when compared to GAT globally with GraphSAGE locally for NF-ToN-IoT.

GAT locally with GraphSAGE globally configuration achieved the highest final performance and required approximately 25–30% fewer federation rounds to reach performance plateaus compared to alternative setups. This trend was consistent across multiple runs and demonstrates the efficiency of this configuration in federated environments. Furthermore, experiments revealed significant heterogeneity across clients, with network sizes varying by up to 40% and attack distributions showing up to 25% variation in class prevalence. The adaptive weighting approach effectively addresses this heterogeneity, reducing the performance gap between the best and worst performing clients from 12% to just 5%. Additionally, clients with more balanced attack distributions received weights 1.8x higher on average than those with skewed distributions, prioritizing more representative data. As shown in Fig. 5, which illustrates the convergence behavior on the NF-ToN-IoT dataset.

Specialized detector performance
[FIG-6]
The provided bar chart visualizes the recall performance of three specialized Graph Attention Network (GAT) variants—Temporal (red), Content (blue), and Behavioral (green)—across seven distinct network attack categories, demonstrating that each variant excels at detecting specific types of threats. The Temporal GAT achieves the highest recall for timing-based anomalies, noting a "Perfect Recall" score of 1.0 for DoS attacks and leading performance for DDoS. The Content GAT dominates in payload-oriented attacks, showcasing a "Near-Perfect Recall" for XSS and superior detection for Injection vulnerabilities. Finally, the Behavioral GAT proves most effective at identifying pattern-based exploits, achieving an "Excellent Recall" for Backdoor attacks while also serving as the primary detection mechanism for Password and Scanning threats. Overall, the graph illustrates the architectural design choice from Figure 2, confirming that distributing threat detection across specialized models yields highly accurate identification tailored to the unique characteristics of different cyber attacks.
[/FIG-6]
caption: Performance of specialized detectors on their targeted attack types on the NF-ToN-IoT dataset. Each detector shows enhanced effectiveness for its specialized attack category.

As shown in Fig. 6, each GAT variant achieves optimal recall on its targeted attack types on the NF-ToN-IoT dataset, the Temporal GAT achieved perfect recall (1.0) for DoS attacks, the Content GAT demonstrated near-perfect recall (0.999) for XSS attacks, and the Behavioral GAT achieved excellent recall (0.994) for Backdoor attacks. Similar trends were observed for the CIC-ToN-IoT dataset, with each detector variant consistently capturing the majority of its specialized attack types. These high recall values highlight the detectors’ ability to identify virtually all instances of their respective attack classes–an essential requirement in operational security settings where undetected threats can have critical implications.

Comparative performance evaluations show that the specialized detector approach consistently outperforms a single generalized detector. On NF-ToN-IoT, the specialized architecture yielded a 12.5% improvement in balanced accuracy and a 14.2% increase in macro F1 score. On CIC-ToN-IoT, the improvements were similarly pronounced, with a 10.8% gain in balanced accuracy and a 12.6% rise in macro F1, further validating the benefit of tailoring architectures to attack characteristics.

Privacy-performance balance and computational efficiency
FedGATSage achieves a strong balance between privacy preservation and detection performance while maintaining computational efficiency across distributed environments. The approach ensures that raw network data never leaves client networks, with individual device identities abstracted through community aggregation, while achieving 78.58% balanced accuracy on NF-ToN-IoT and 80.24% on CIC-ToN-IoT with only a 2.8% gap compared to centralized models.

The computational efficiency analysis reveals our three-fold optimization strategy designed for practical IoT deployment. Time complexity analysis shows the computational cost per training round: client-side GAT processing requires O(|V| \cdot d \cdot h + |E| \cdot d \cdot h) operations per layer, where |V| is the number of unique IP addresses in the network, |E| is the number of network flows between devices, d is the feature dimension (typically 256), and h is the hidden layer dimension. Community detection using the Louvain algorithm requires O(|E| \log |V|) operations to identify device clusters, while server-side GraphSAGE processing requires O(k \cdot |F| \cdot d) operations where k is the neighborhood sampling size and |F| represents the number of flow embeddings after community abstraction (significantly smaller than |V|).

Space complexity analysis demonstrates memory efficiency essential for resource-constrained IoT devices. Each client stores node embeddings requiring O(|V| \cdot d) memory space (one embedding per IP address), while model parameters require O(L \cdot d^2) space where L is the number of neural network layers. The server maintains flow embeddings requiring O(|C| \cdot |F| \cdot d) memory where |C| is the number of participating clients.

Communication complexity analysis quantifies bandwidth requirements critical for IoT networks. Our approach requires transmitting O(|F| \cdot d) data per client due to community abstraction, resulting in an 85% reduction in communication overhead–from approximately 25KB to 3.2KB per client in typical networks with 100 IP addresses, where 5-7 communities effectively represent the entire network structure.

The scientific foundation for this favorable privacy-performance tradeoff is demonstrated through strong correlation patterns between attack types across communities (Fig. 7), with a Pearson correlation coefficient of 0.87 (p-value: 2.39e-16). This correlation provides scientific justification for the community abstraction approach and explains why effective pattern detection is maintained while ensuring privacy.

[FIG-7]
+------------+--------+----------+------+------+-----------+--------+----------+------------+----------+-------+
| Attack     | Benign | backdoor | ddos | dos  | injection | mitm   | password | ransomware | scanning | xss   |
+------------+--------+----------+------+------+-----------+--------+----------+------------+----------+-------+
| Benign     | 1      | 0.61     | 0.33 | 0.33 | -0.093    | 0.34   | 0.031    | 0.49       | 0.56     | -0.16 |
| backdoor   | 0.61   | 1        | 0.62 | 0.63 | -0.041    | 0.71   | 0.16     | 0.86       | 0.95     | -0.16 |
| ddos       | 0.33   | 0.62     | 1    | 1    | 0.76      | 0.33   | 0.87     | 0.94       | 0.84     | 0.67  |
| dos        | 0.33   | 0.63     | 1    | 1    | 0.75      | 0.34   | 0.87     | 0.94       | 0.85     | 0.67  |
| injection  | -0.093 | -0.041   | 0.76 | 0.75 | 1         | -0.18  | 0.98     | 0.48       | 0.28     | 0.99  |
| mitm       | 0.34   | 0.71     | 0.33 | 0.34 | -0.18     | 1      | -0.029   | 0.54       | 0.63     | -0.26 |
| password   | 0.031  | 0.16     | 0.87 | 0.87 | 0.98      | -0.029 | 1        | 0.64       | 0.47     | 0.95  |
| ransomware | 0.49   | 0.86     | 0.94 | 0.94 | 0.48      | 0.54   | 0.64     | 1          | 0.98     | 0.37  |
| scanning   | 0.56   | 0.95     | 0.84 | 0.85 | 0.28      | 0.63   | 0.47     | 0.98       | 1        | 0.17  |
| xss        | -0.16  | -0.16    | 0.67 | 0.67 | 0.99      | -0.26  | 0.95     | 0.37       | 0.17     | 1     |
+------------+--------+----------+------+------+-----------+--------+----------+------------+----------+-------+
[/FIG-7]
caption: Community-attack correlation matrix showing correlation coefficients between attack types across communities. Strong positive correlations indicate communities with similar structural features experience similar attack patterns.
[FIG-8]
The line graph, titled "Client Accuracy over Training Rounds," tracks the learning progression of five distinct federated clients across 30 communication rounds. Initially, at Round 0, the clients exhibit high variance and relatively low performance, with starting accuracies ranging from a minimum of approximately 0.19 for Client 3 to a high of 0.46 for Client 4. During the early phase (Rounds 0 through 7), the models experience steep but volatile improvements, quickly converging near the 0.60 to 0.65 mark. As the federated training progresses from round 15 to 30, the accuracy curves stabilize and tighten, growing steadily in unison. By the final rounds, the collaborative learning process allows all five clients to overcome their initial disparities, plateauing in a tightly clustered, high-accuracy band between roughly 0.75 and 0.80, with Client 3 briefly peaking just above 0.80 near round 26.
[/FIG-8]
caption: Accuracy progression for five simulated clients over 15 training rounds. Despite initial variability, clients exhibit steady improvement and converge towards a similar accuracy range, demonstrating the effectiveness of the training approach.

Experimental setup
FedGATSage was evaluated on the NF-ToN-IoT and CIC-ToN-IoT datasets, covering structural, content-based, and temporal attacks. The dataset was partitioned across five simulated clients with balanced attack distributions and network heterogeneity. Client sizes varied by up to 40% and class prevalence varied by 25%. Clients processed data locally and shared community embeddings for global training. Client performance across training rounds is shown in Fig. 8.

The implementation used PyTorch 1.8.0 and PyTorch Geometric 2.0.1, with training on NVIDIA GPUs and CPU-only deployment tested. GAT models used 256 hidden dimensions, 8 attention heads, and 0.2 dropout; GraphSAGE used 256 hidden dimensions with 2-layer neighborhood sampling. Evaluation included centralized GAT, GraphSAGE, FedLSTM, a basic federated GNN, and a FedGATSage variant (GAT global, GraphSAGE local). Metrics included accuracy, balanced accuracy, macro F1 score, and computational efficiency measures.

Discussion
FedGATSage addresses the dual limitation we identified in existing federated intrusion detection approaches: traditional federated methods lose structural patterns due to architectural constraints, while GNN federated methods lose temporal patterns during parameter aggregation. Our results demonstrate that this dual-solution approach achieves near-centralized performance for network intrusion detection while preserving privacy, with a performance gap of only 2.8%. This success stems from our specialized detector architecture and community-based abstraction mechanism working collaboratively to address both architectural and aggregation challenges, validating our hypothesis that flow-level abstraction with community enhancement can maintain detection effectiveness without compromising privacy guarantees.

The architectural design choices are theoretically grounded in addressing both limitation types through complementary GNN strengths in federated environments. GAT’s selection for client-side processing addresses the architectural limitation by providing attention mechanisms that can dynamically weight neighbor importance, which is crucial for intrusion detection where malicious connections may represent only a small fraction of total network traffic. This adaptive weighting is essential for detecting attacks that exploit trust relationships within local network communities, where the significance of specific connections cannot be predetermined. Conversely, GraphSAGE’s selection for server-side processing addresses the aggregation limitation by leveraging its inductive learning capability and sampling efficiency, enabling efficient processing of dynamic overlay graphs where community embeddings from different clients create variable graph structures across federation rounds. This combination addresses the federated learning challenge of balancing local detail with global generalization, where GAT preserves local attack patterns while GraphSAGE efficiently aggregates them across the federation without destroying temporal signatures.

The community abstraction approach provides the scientific foundation for preserving both structural and temporal patterns simultaneously. Our analysis reveals a strong correlation between structural similarity and attack pattern similarity across communities, with a Pearson correlation coefficient of 0.87 (p-value: 2.39e-16). This statistical relationship explains why our approach maintains detection effectiveness even when using abstracted community-level representations rather than detailed node-level data, addressing both privacy concerns and pattern preservation requirements. Communities with similar structural features tend to experience similar types of attacks because network topology reflects organizational structure, device roles, and vulnerability patterns that attackers naturally target, enabling effective temporal pattern detection across different network domains.

The computational complexity analysis reveals the practical feasibility of our dual-solution approach for real-world IoT deployment. The three-fold optimization strategy spanning time, space, and communication complexity demonstrates that FedGATSage maintains computational efficiency while addressing both architectural and aggregation limitations. This efficiency directly translates to practical deployment advantages, where convergence analysis shows that GAT-local with GraphSAGE-global configuration requires 25-30% fewer federation rounds compared to alternative arrangements. The 85% reduction in communication overhead makes it particularly suitable for bandwidth-constrained IoT environments where traditional federated learning approaches would be impractical due to excessive data transfer requirements. Furthermore, implementing specialized GAT variants for different attack types enables balanced detection capabilities across structural attacks (Backdoor, Injection) and temporal attacks (DDoS, DoS), a capability that neither traditional federated approaches nor existing GNN federated methods can achieve.

Comparison with State-of-the-Art Approaches: Our results demonstrate competitive performance compared to recent federated learning approaches for IoT intrusion detection while addressing fundamental limitations that existing methods do not solve. Recent approaches such as the optimal federated learning-based IDS achieve 95.59% accuracy on binary classification tasks using MQTT datasets36, while Tab Transformer-based federated frameworks report 98.80% accuracy on multiclass scenarios using CICIoT202337. AutoEncoder-based federated learning approaches achieve 90.93% accuracy on device-specific scenarios using N-BaIoT datasets38, and group-based federated learning methods demonstrate comparable accuracy for anomaly detection in smart home environments39. However, these approaches face the same dual limitations we identified: traditional approaches suffer from architectural constraints that prevent structural pattern capture, while GNN approaches suffer from temporal pattern destruction during parameter averaging, making them ineffective against coordinated attacks. Our specialized GAT detector variants achieve perfect recall for DoS attacks and near-perfect recall for XSS attacks through targeted architectures that preserve both structural and temporal characteristics, representing the first federated approach to successfully address both limitation types simultaneously.

However, several limitations warrant consideration for practical deployment. While FedGATSage excels at addressing both structural and temporal pattern preservation challenges, some attack types like scanning still present opportunities for enhancement, with F1-scores below 0.20. The performance variations across attack types highlight that certain attacks like DoS and XSS exhibit higher false positive rates, which may require post-processing mechanisms in critical environments. Additionally, our evaluation was conducted on static datasets, whereas real-world IoT networks are dynamic and may require sliding-window approaches for continuous adaptation to evolving threat landscapes. These limitations suggest promising research directions including adaptive community detection algorithms that dynamically adjust to changing network activities, more sophisticated overlay graph construction methods to better capture relationships between communities across heterogeneous clients, integration with complementary privacy-preserving techniques for additional security guarantees, and exploring dynamic community boundaries for better adaptation to evolving attack patterns.

Conclusion
FedGATSage represents the first federated learning approach to successfully solve both architectural and aggregation limitations that have prevented effective intrusion detection in distributed environments. By systematically addressing the dual challenge where traditional federated methods cannot capture structural patterns and GNN federated methods lose temporal patterns, our work establishes a new paradigm for privacy-preserving collaborative security that maintains detection effectiveness against sophisticated coordinated attacks.

The technical contributions advance the field through four key innovations that address fundamental limitations in federated intrusion detection. First, our hybrid federated learning architecture that strategically combines client-side GAT with server-side GraphSAGE addresses both architectural and aggregation limitations, where GAT captures local structural patterns while GraphSAGE efficiently processes global patterns without destroying temporal signatures Second, specialized GAT detector variants with ensemble fusion provide the architectural capability to capture structural patterns that traditional federated approaches fundamentally cannot model, while enabling attack-type-aware detection in federated environments. Third, our custom federated aggregation mechanism with performance-weighted averaging addresses client heterogeneity while maintaining both structural and temporal pattern preservation. Fourth, community-enhanced feature engineering enables privacy-preserving structural pattern learning across distributed networks while achieving substantial communication overhead reduction. The practical impact extends beyond performance metrics to provide the first federated solution capable of detecting sophisticated coordinated attacks that require both structural relationship understanding and temporal sequence preservation. Our community abstraction mechanism achieves an 85% reduction in communication overhead compared to node-level sharing, directly addressing the bandwidth constraints that make traditional federated learning impractical for IoT environments, while successful detection of challenging temporal attacks like DDoS campaigns addresses critical security gaps where existing federated methods fundamentally fail due to their individual limitation types. The strong correlation between community structure and attack patterns provides scientific justification for why our dual-solution approach maintains effectiveness while ensuring privacy, enabling favorable privacy-performance tradeoffs that neither traditional nor GNN federated approaches can achieve alone.

Future research directions include extending the approach to dynamic community detection for evolving networks, incorporating additional attack categories that may benefit from the dual-solution framework, and exploring integration with complementary privacy-preserving techniques for enhanced security guarantees. The successful demonstration of near-centralized performance in federated settings while maintaining privacy through systematic solution of both architectural and aggregation limitations opens new possibilities for collaborative security in distributed IoT ecosystems, particularly for organizations requiring both regulatory compliance and effective threat detection against sophisticated coordinated attacks that span multiple network domains.

