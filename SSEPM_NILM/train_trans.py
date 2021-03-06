from nilmtk import DataSet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
import networkx as nx


# %% Function section


# Power transformations
def find_log(x):
    if x > 0:
        log = np.log(x)
    elif x < 0:
        log = np.log(x * -1) * -1
    elif x == 0:
        log = 0
    return log


# %% Load dataset


DATASET = 'IMDELD'

if DATASET == 'REDD':
    redd = DataSet('REDD/redd.h5')
    machine = 'fridge'
    building_1_elec = redd.buildings[1].elec
    machines_dict = {'oven1': 3,
                     'oven2': 4,
                     'microwave': 11,
                     'kitchen outlets1': 15,
                     'kitchen outlets2': 16,
                     'bathroom gfi': 12,
                     'washer dryer1': 10,
                     'washer dryer2': 20,
                     'fridge': 5,
                     'dish washer': 6}
    df_machine = next(building_1_elec[machines_dict[machine]].load())

    data_trans_app = {f'{machine}': np.log(df_machine['power']['active'].mask(df_machine['power']['active'] <= 0)).fillna(0)}

    df_power_trans_app = pd.DataFrame(data_trans_app)

elif DATASET == 'IMDELD':
    imdeld = DataSet('IMDELD/IMDELD.hdf5')
    machine = 'double_pole_II'
    building_elec = imdeld.buildings[1].elec
    machines_dict = {'milling_I': 10,
                     'milling_II': 11,
                     'pelletizer_I': 3,
                     'pelletizer_II': 4,
                     'exhaust_fan_I': 7,
                     'exhaust_fan_II': 8,
                     'double_pole_I': 5,
                     'double_pole_II': 6}
    df_machine = next(building_elec[machines_dict[machine]].load())

    data_trans_app = {f'{machine}_P': df_machine['power']['active'].apply(find_log),
                      f'{machine}_Q': df_machine['power']['reactive'].apply(find_log)}

    df_power_trans_app = pd.DataFrame(data_trans_app)


# %% Run algorithm


def run_algorithm(appliance, test_index, window_length, shift_count, power_samples, df_power_trans_app,
                  features, rising_features, falling_features, dataset, num_last_steady_sts=4,
                  min_samples_steady_state=10, max_samples_transient_state=3, max_window_size=1350):
    """

    :param appliance: String with the appliance name that will be processed for training
    :param test_index: Integer that indicates the start sample of the whole time series
    :param window_length: Integer that indicates the length of samples of the whole time series
    :param shift_count: Number of samples that must be shifted the window to start at the first sample of the second steady state
    :param power_samples: dictionary to store the power samples in the window
    :param df_power_trans_app: dataframe containing the transformed (log transformation) power of each of the appliances
    :param features: dictionary with the common features for all the events
    :param rising_features: dictionary with the features of the rising events
    :param falling_features: dictionary with the features of the falling events
    :param dataset: string with the name of the dataset
    :param num_last_steady_sts: Number of steady states to save the euclidean distances mean
    :param min_samples_steady_state: Integer that indicates the minimum number of data points that should be included in a group to be consider a cluster
    :param max_samples_transient_state: Integer to indicate the maximum number of transient samples to give less weight in the euclidean distance calculation
    :param max_window_size: Least number of samples to build up to 2 clusters
    :return: shift_count, power_samples, features, rising_features, falling_features, appliance
    """

    events_ts = []  # To record the timestamp of when the events happened
    ON_OFF_event = []  # To record if was an On or OFF event
    euc_dist_means = []  # To record the previous euclidean distances from the past 2 consecutive steady states
    euc_dist_stds = []  # To record the previous euclidean distances' standard deviations from the past 2 consecutive steady states
    noise_detector = (min_samples_steady_state / 2) - 1  # Value that is linked with the minimum amount of steady state samples (therefore the frequency) to see if non consecutive samples are noise
    rising_events = []  # To store the rising events unix timestamp
    falling_events = []  # To store the falling events unix timestamp
    window_size = (min_samples_steady_state * 2) + 1  # Least number of samples to build up to 2 clusters
    weights = [0.4, 0.5, 0.6]  # Give less weight to the 3 most separated samples (largest euclidean distances)
    shift = False  # Flag to shift the power samples to the first sample of the second steady state

    for i in range(len(df_power_trans_app[test_index:test_index + window_length])):
        if shift:
            shift_count += min_samples_steady_state  # It must shifted back the min number of samples of the steady state (DBSCAN parameter)
            shift = False  # Flag back to False to search again for other steady states
        if dataset == 'REDD':
            power_samples['P_t'].append(df_power_trans_app[appliance][test_index:test_index + window_length].iloc[i - shift_count])
        elif dataset == 'IMDELD':
            power_samples['P_t'].append(df_power_trans_app[appliance + '_P'][test_index:test_index + window_length].iloc[i - shift_count])
            power_samples['Q_t'].append(df_power_trans_app[appliance + '_Q'][test_index:test_index + window_length].iloc[i - shift_count])
        if len(power_samples['P_t']) >= window_size:
            if dataset == 'REDD':
                df_power_samples = pd.DataFrame(power_samples, index=df_power_trans_app[appliance].iloc[test_index:test_index + window_length].index[(i - shift_count) - window_size + 1:(i - shift_count) + 1])
            elif dataset == 'IMDELD':
                df_power_samples = pd.DataFrame(power_samples, index=df_power_trans_app[appliance + '_P'].iloc[test_index:test_index + window_length].index[(i - shift_count) - window_size + 1:(i - shift_count) + 1]).dropna()
            ts = df_power_samples.index
            ts = ts.astype('int64')
            # Calculate the euclidean distance between consecutive rows, i.e. 0 and 1, 1 and 2, 2 and 3...
            euc_dist = np.linalg.norm(df_power_samples.diff(axis=0).dropna(), axis=1)
            # Weighted euclidean mean to give less weight to the big transition. Doing so we can still detect more samples from the transient state
            transient_idx = euc_dist.argsort()[-max_samples_transient_state:][::-1]
            # euc_dist[euc_dist.argmax()] = euc_dist[euc_dist.argmax()] * weight
            euc_dist[transient_idx] = euc_dist[transient_idx] * weights
            # Calculating the mean of the euclidean distances to define how apart should be the samples of each cluster (dynamic eps)
            euc_dist_mean = np.mean(euc_dist)
            euc_dist_means.append(euc_dist_mean)
            # Calculating the standard deviation of the euclidean distances to define a margin to add to the dynamic eps
            euc_dist_std = np.std(euc_dist)
            euc_dist_stds.append(euc_dist_std)
            if len(euc_dist_means) > num_last_steady_sts:  # Only consider a certain number of last steady states (This is done to mitigate the noise coming from a long steady state)
                euc_dist_means.pop(0)  # Discard the last (first in the list) steady state mean
                euc_dist_stds.pop(0)  # Discard the last (first in the list) steady state standard deviation
                euc_dists_mean = np.mean(euc_dist_means)  # Mean of the last euclidean distances' means from the last steady states
                euc_dists_std = np.mean(euc_dist_stds)  # Mean of the last euclidean distances' standard deviations from the last steady states
            else:
                euc_dists_mean = np.mean(euc_dist_means)
                euc_dists_std = np.mean(euc_dist_stds)
            # Compute the clustering with DBSCAN
            if euc_dists_mean == 0:  # To avoid having a radio of 0
                euc_dists_mean = 0.00001
            clusters = DBSCAN(eps=euc_dists_mean + (euc_dists_std * 2), min_samples=min_samples_steady_state).fit(df_power_samples)
            values, counts = np.unique(clusters.labels_, return_counts=True)
            if not {0, 1}.issubset(np.unique(clusters.labels_)):  # To check if there are at least 2 steady states and one transient (-1) (equivalent set([0, 1]).issubset(np.unique(clusters.labels_)))
                window_size += 1  # If not, increase window size to increase the likelihood of finding two steady states
                if window_size % max_window_size == 0:  # If the window size increases more than max_window_size samples, start with a new window to avoid the overload for the DBSCAN calculation
                    print(window_size)
                    if dataset == 'REDD':
                        power_samples = {'P_t': []}  # Empty the power samples to detect new transient states
                    elif dataset == 'IMDELD':
                        power_samples = {'P_t': [], 'Q_t': []}  # Empty the power samples to detect new transient states
                    window_size = (min_samples_steady_state * 2) + 1  # Return to the initial window size
            else:
                print(window_size)
                shift = True  # It means that the samples must be shifted back to the first sample of the second steady state
                if dataset == 'REDD':
                    power_samples = {'P_t': []}  # Empty the power samples to detect new transient states
                elif dataset == 'IMDELD':
                    power_samples = {'P_t': [], 'Q_t': []}  # Empty the power samples to detect new transient states
                window_size = (min_samples_steady_state * 2) + 1  # Return to the initial window size
                first_steady_st_inx = [idx for idx, element in enumerate(clusters.labels_) if element == 0]
                second_steady_st_inx = [idx for idx, element in enumerate(clusters.labels_) if element == 1]
                # Check if the second cluster has consecutive indexes, if not it is likely that it is noise or a transient state
                # The difference between consecutive indexes should always be 1, so unique checks if there is a difference outside 1
                first_cond = len(np.unique(np.diff(second_steady_st_inx))) > noise_detector
                # OR if any of the first steady state index is bigger than the first index of the second steady state it can mean that is a long transient between events (the second steady state is in a transient state)
                second_cond = any(first > second_steady_st_inx[0] for first in first_steady_st_inx)
                if first_cond or second_cond:
                    continue
                print(len(clusters.labels_))
                events_ts.append(ts[first_steady_st_inx[-1]])  # Save the last sample of the first steady state: that is where the transition to another state occurred
                # Transition record
                features['transition'].append(np.abs(df_power_samples.iloc[second_steady_st_inx].P_t.mean() - df_power_samples.iloc[first_steady_st_inx].P_t.mean()))
                if dataset == 'IMDELD':
                    features['transition_Q'].append(np.abs(df_power_samples.iloc[second_steady_st_inx].Q_t.mean() - df_power_samples.iloc[first_steady_st_inx].Q_t.mean()))
                # Timestamp record
                features['timestamp'].append(ts[first_steady_st_inx[-1]])
                # Define if it was an ON or OFF event
                ON_or_OFF = np.abs(df_power_samples.iloc[second_steady_st_inx[0]].P_t) - np.abs(df_power_samples.iloc[first_steady_st_inx[-1]].P_t)
                if ON_or_OFF > 0:  # Means an ON event because the appliance is consuming more
                    ON_OFF_event.append(1)  # 1 symbolizes an ON event
                    rising_events.append(ts[first_steady_st_inx[-1]])  # Store the rising event unix timestamp
                    features['high_state_max'].append(df_power_samples.iloc[second_steady_st_inx].P_t.max())
                    features['low_state_max'].append(df_power_samples.iloc[first_steady_st_inx].P_t.max())
                    features['high_state_min'].append(df_power_samples.iloc[second_steady_st_inx].P_t.min())
                    features['low_state_min'].append(df_power_samples.iloc[first_steady_st_inx].P_t.min())
                    if dataset == 'IMDELD':
                        features['high_state_max_Q'].append(df_power_samples.iloc[second_steady_st_inx].Q_t.max())
                        features['low_state_max_Q'].append(df_power_samples.iloc[first_steady_st_inx].Q_t.max())
                        features['high_state_min_Q'].append(df_power_samples.iloc[second_steady_st_inx].Q_t.min())
                        features['low_state_min_Q'].append(df_power_samples.iloc[first_steady_st_inx].Q_t.min())
                elif ON_or_OFF <= 0:  # Means an OFF event because the appliance is consuming less
                    ON_OFF_event.append(0)  # 0 symbolizes an OFF event
                    falling_events.append(ts[first_steady_st_inx[-1]])  # Store the falling event unix timestamp
                    features['low_state_max'].append(df_power_samples.iloc[second_steady_st_inx].P_t.max())
                    features['high_state_max'].append(df_power_samples.iloc[first_steady_st_inx].P_t.max())
                    features['low_state_min'].append(df_power_samples.iloc[second_steady_st_inx].P_t.min())
                    features['high_state_min'].append(df_power_samples.iloc[first_steady_st_inx].P_t.min())
                    if dataset == 'IMDELD':
                        features['low_state_max_Q'].append(df_power_samples.iloc[second_steady_st_inx].Q_t.max())
                        features['high_state_max_Q'].append(df_power_samples.iloc[first_steady_st_inx].Q_t.max())
                        features['low_state_min_Q'].append(df_power_samples.iloc[second_steady_st_inx].Q_t.min())
                        features['high_state_min_Q'].append(df_power_samples.iloc[first_steady_st_inx].Q_t.min())

                # Feature extraction (assuming in the time window there is just one transient state)
                if {-1}.issubset(values):  # meaning that noise was detected from the transient states in between steady states
                    trans_idx = [idx for idx, element in enumerate(clusters.labels_) if element == -1]  # Check which are the indexes of the outliers (i.e., -1) (transient samples)
                    if ON_or_OFF > 0:  # Define if it is a rising or falling spike to determine its dimensions
                        # Transient duration
                        trans_duration = pd.Timedelta.total_seconds(df_power_samples.index[second_steady_st_inx[0]] - df_power_samples.index[first_steady_st_inx[-1]])  # Timedelta to total seconds with pandas
                        rising_features['ris_trans_duration'].append(trans_duration)
                        # Active and reactive power change of the transient state
                        trans_active_change = np.exp(df_power_samples.iloc[trans_idx].P_t.max()) - np.exp(df_power_samples.iloc[first_steady_st_inx].P_t.mean())
                        rising_features['ris_trans_power_change_P'].append(trans_active_change)
                        # Transient spike for reactive and active powers
                        max_P = np.exp(df_power_samples.iloc[trans_idx].P_t.max())
                        min_P = np.exp(df_power_samples.iloc[second_steady_st_inx].P_t.mean())
                        rising_features['ris_trans_spike_P'].append(max_P - min_P)
                        if dataset == 'IMDELD':
                            # Active and reactive power change of the transient state
                            trans_reactive_change = np.exp(df_power_samples.iloc[trans_idx].Q_t.max()) - np.exp(df_power_samples.iloc[first_steady_st_inx].Q_t.mean())
                            rising_features['ris_trans_power_change_Q'].append(trans_reactive_change)
                            # Transient spike for reactive and active powers
                            max_Q = np.exp(df_power_samples.iloc[trans_idx].Q_t.max())
                            min_Q = np.exp(df_power_samples.iloc[second_steady_st_inx].Q_t.mean())
                            rising_features['ris_trans_spike_Q'].append(max_Q - min_Q)
                    else:
                        # Transient duration
                        trans_duration = pd.Timedelta.total_seconds(df_power_samples.index[second_steady_st_inx[0]] - df_power_samples.index[first_steady_st_inx[-1]])  # Timedelta to total seconds with pandas
                        falling_features['fall_trans_duration'].append(trans_duration)
                        # Active and reactive power change of the transient state
                        trans_active_change = np.exp(df_power_samples.iloc[first_steady_st_inx].P_t.mean()) - np.exp(df_power_samples.iloc[trans_idx].P_t.min())
                        # Transient spike for reactive and active powers
                        max_P = np.exp(df_power_samples.iloc[second_steady_st_inx].P_t.mean())
                        min_P = np.exp(df_power_samples.iloc[trans_idx].P_t.min())
                        # Active and reactive power change of the transient state
                        falling_features['fall_trans_power_change_P'].append(trans_active_change)
                        # Transient spike for reactive and active powers
                        falling_features['fall_trans_spike_P'].append(max_P - min_P)
                        if dataset == 'IMDELD':
                            # Active and reactive power change of the transient state
                            trans_reactive_change = np.exp(df_power_samples.iloc[first_steady_st_inx].Q_t.mean()) - np.exp(df_power_samples.iloc[trans_idx].Q_t.min())
                            # Transient spike for reactive and active powers
                            max_Q = np.exp(df_power_samples.iloc[second_steady_st_inx].Q_t.mean())
                            min_Q = np.exp(df_power_samples.iloc[trans_idx].Q_t.min())
                            # Active and reactive power change of the transient state
                            falling_features['fall_trans_power_change_Q'].append(trans_reactive_change)
                            # Transient spike for reactive and active powers
                            falling_features['fall_trans_spike_Q'].append(max_Q - min_Q)
    return shift_count, power_samples, features, rising_features, falling_features, appliance


if DATASET == 'REDD':
    window_length = len(df_power_trans_app)
    power_samples = {'P_t': []}
    features = {'timestamp': [], 'transition': [], 'low_state_max': [],
                'high_state_max': [], 'low_state_min': [],
                'high_state_min': []}
    rising_features = {'ris_trans_power_change_P': [], 'ris_trans_spike_P': [], 'ris_trans_duration': []}
    falling_features = {'fall_trans_power_change_P': [], 'fall_trans_spike_P': [], 'fall_trans_duration': []}
elif DATASET == 'IMDELD':
    machinery = {'milling_I': 383846, 'milling_II': 383521, 'pelletizer_I': 1974586, 'pelletizer_II': 1968096,
                 'exhaust_fan_I': 1984441, 'exhaust_fan_II': 1968800, 'double_pole_I': 1984563, 'double_pole_II': 1976698}
    # 383846 is 2018-02-28 15:59:53 and is used for milling machine I (36.7% of total data samples)
    # 383521 is 2018-02-26 22:09:50 and is used for milling machine II (36.7% of total data samples)
    # 1974586 is 2017-12-14 13:52:16 and is used for pelletizer I (36.7% of total data samples)
    # 1968096 is 2017-12-21 08:49:37 and is used for pelletizer II (36.7% of total data samples)
    # 1984441 is 2017-12-14 16:40:23 and is used for exhaust fan I (36.7% of total data samples)
    # 1968800 is 2017-12-14 12:21:44 and is used for exhaust fan II (36.7% of total data samples)
    # 1984563 is 2017-12-14 16:42:00 and is used for double pole I (36.7% of total data samples)
    # 1976698 is 2017-12-14 14:29:43 and is used for double pole II (36.7% of total data samples)

    window_length = machinery[machine]
    power_samples = {'P_t': [], 'Q_t': []}
    features = {'timestamp': [], 'transition': [], 'transition_Q': [],
                'low_state_max': [], 'high_state_max': [],
                'low_state_min': [], 'high_state_min': [],
                'low_state_max_Q': [], 'high_state_max_Q': [],
                'low_state_min_Q': [], 'high_state_min_Q': []}
    rising_features = {'ris_trans_power_change_P': [], 'ris_trans_spike_P': [],
                       'ris_trans_power_change_Q': [], 'ris_trans_spike_Q': [],
                       'ris_trans_duration': []}
    falling_features = {'fall_trans_power_change_P': [], 'fall_trans_spike_P': [],
                        'fall_trans_power_change_Q': [], 'fall_trans_spike_Q': [],
                        'fall_trans_duration': []}

min_samples_steady_state = 10
test_index = 0
shift_count = min_samples_steady_state + 1  # Number of samples that must be shifted the window to start at the first sample of the second steady state
df_app_transitions = pd.DataFrame()
df_rising_features = pd.DataFrame()
df_falling_features = pd.DataFrame()

while shift_count > min_samples_steady_state:
    shift_count = 0
    shift_count, power_samples, features, rising_features, falling_features, appliance = run_algorithm(machine, test_index, window_length, shift_count,
                                                                                                       power_samples, df_power_trans_app, features,
                                                                                                       rising_features, falling_features, DATASET,
                                                                                                       min_samples_steady_state=min_samples_steady_state)
    print('shift count:', shift_count)
    test_index = test_index + window_length - (shift_count + len(power_samples['P_t'])) - 1
    window_length = shift_count + len(power_samples['P_t'])
    features_df = pd.DataFrame(features)
    rising_features_df = pd.DataFrame(rising_features)
    falling_features_df = pd.DataFrame(falling_features)
    df_app_transitions = pd.concat([df_app_transitions, features_df])
    df_rising_features = pd.concat([df_rising_features, rising_features_df])
    df_falling_features = pd.concat([df_falling_features, falling_features_df])

    if DATASET == 'REDD':
        power_samples = {'P_t': []}
        features = {'timestamp': [], 'transition': [], 'low_state_max': [],
                    'high_state_max': [], 'low_state_min': [],
                    'high_state_min': []}
        rising_features = {'ris_trans_power_change_P': [], 'ris_trans_spike_P': [], 'ris_trans_duration': []}
        falling_features = {'fall_trans_power_change_P': [], 'fall_trans_spike_P': [], 'fall_trans_duration': []}
    elif DATASET == 'IMDELD':
        power_samples = {'P_t': [], 'Q_t': []}
        features = {'timestamp': [], 'transition': [], 'transition_Q': [],
                    'low_state_max': [], 'high_state_max': [],
                    'low_state_min': [], 'high_state_min': [],
                    'low_state_max_Q': [], 'high_state_max_Q': [],
                    'low_state_min_Q': [], 'high_state_min_Q': []}
        rising_features = {'ris_trans_power_change_P': [], 'ris_trans_spike_P': [],
                           'ris_trans_power_change_Q': [], 'ris_trans_spike_Q': [],
                           'ris_trans_duration': []}
        falling_features = {'fall_trans_power_change_P': [], 'fall_trans_spike_P': [],
                            'fall_trans_power_change_Q': [], 'fall_trans_spike_Q': [],
                            'fall_trans_duration': []}

df_app_transitions.to_csv(f'transitions/{DATASET}/{appliance}.csv')
df_rising_features.to_csv(f'transitions/{DATASET}/{appliance}_rising.csv')
df_falling_features.to_csv(f'transitions/{DATASET}/{appliance}_falling.csv')

# %% Centroids separation
# Load the appliance power transitions
appliance = machine
app_transitions = pd.read_csv(f'transitions/{DATASET}/{appliance}.csv', index_col=0)
app_transitions['watt_transition_low'] = np.abs(np.exp(app_transitions['high_state_min']) - np.exp(app_transitions['low_state_max']))
app_transitions['watt_transition_high'] = np.abs(np.exp(app_transitions['high_state_max']) - np.exp(app_transitions['low_state_min']))
transitions = pd.concat([app_transitions['watt_transition_low'], app_transitions['watt_transition_high']], axis=0)
app_watt_transition = pd.DataFrame(transitions)
agglomerative_clustering = AgglomerativeClustering(n_clusters=10).fit(app_watt_transition)
clf = NearestCentroid()
clf.fit(app_watt_transition, agglomerative_clustering.labels_)
centroids = clf.centroids_
# Lookup list to associate a centroid to the label of the observation
cent_look_up = {idx: centroid[0] for idx, centroid in enumerate(centroids)}
sort_cent = np.sort(centroids, axis=0)[::-1]  # Sorting the centroids in descending order
root = sort_cent[0]
group_cond = root * 0.15
pairwise = pd.DataFrame(
    squareform(pdist(sort_cent)),
)
centroid_pairs = []
for col in range(pairwise.shape[1]):
    if not any(np.array(pairwise[col][col + 1:]) < group_cond) and len(np.array(pairwise[col][col + 1:]) < group_cond) != 0:  # If any of the distances is less than the 15% of the root centroid then that is a unique centroid
        root = sort_cent[col + 1]
        group_cond = root * 0.15
        print(group_cond)
    elif len(np.array(pairwise[col][col + 1:]) < group_cond) != 0:  # Just check if there are elements left in the list
        centroid_pairs.append({col: [index[0] + col + 1 for index in np.where(np.array(pairwise[col][col + 1:]) < group_cond)]})  # Keep track of which centroid must be merged with the ones that are near in distance
        root = sort_cent[col + 1]
        group_cond = root * 0.15
# %% Connected components
# Apart from checking that the distance from a root centroid to another to group them together (15% condition), we check as well that each of the centroids are not connected
G1 = nx.Graph()
G1.add_edges_from([(key, value) for element in centroid_pairs for key in element.keys() for value in element[key]])  # generating a connected graph with the connected centroids
connected_comp = sorted(nx.connected_components(G1), key=len, reverse=True)
# Link again the sorted centroid indexes with the agglomerative clustering centroid indexes
sort_cent_look_up = {sorted_idx: idx for sorted_idx, sorted_centroid in enumerate(sort_cent) for idx, centroid in enumerate(centroids) if centroid == sorted_centroid}
connected_comp = [[sort_cent_look_up[component] for component in element] for element in connected_comp]  # List of lists. The inner lists are the centroids that must be merged together
# Retrieve the indexes of the connected centroids to then know which is the minimum and maximum value of that transition
connected_comp_idx = [np.where(np.in1d(agglomerative_clustering.labels_, component)) for component in connected_comp]
# Build the transition intervals:
# 1) with the centroids that must be merged together
transition_intervals = [(app_watt_transition.iloc[indexes].values.min(), app_watt_transition.iloc[indexes].values.max()) for indexes in connected_comp_idx]
# 2) With the centroids that were not merged together
existing_clusters = np.unique(agglomerative_clustering.labels_)
evaluated_clusters = [element for component in connected_comp for element in component]  # Evaluated clusters
missing_centroids = [[element] for element in existing_clusters if element not in evaluated_clusters]
missing_idx = [np.where(np.in1d(agglomerative_clustering.labels_, component)) for component in missing_centroids]
missing_transition_int = [(app_watt_transition.iloc[indexes].values.min(), app_watt_transition.iloc[indexes].values.max()) for indexes in missing_idx]
transition_intervals = missing_transition_int + transition_intervals
# Delete transitions with a value lower than 40W (i.e. appliance ground state or standby mode)
transition_intervals = [(transition[0], transition[1]) for transition in transition_intervals if transition[0] > 40 and transition[1] > 40]
# Delete the transitions with a single point (it is noise)
transition_intervals = [(transition[0], transition[1]) for transition in transition_intervals if not transition[0] == transition[1]]
df_transition_intervals = pd.DataFrame(transition_intervals)
df_transition_intervals.to_csv(f'transitions/{DATASET}/intervals_{appliance}.csv')
