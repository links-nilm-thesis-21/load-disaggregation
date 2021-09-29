from nilmtk import DataSet
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt


# %% Function section


def pdf(dataframe, data_point):
    """
    Probability that the transient features are matched with the event that is in conflict.
    Probability that is modeled with a non parametric density estimation
    :param dataframe: receives the dataframe column with the values of the transient event feature of a specific appliance
    :param data_point: integer with the information of the transient event feature
    :return: logarithmic probability that data_point is from dataframe
    """
    # Reshape data coming from dataframe to fit the probability model
    data = np.reshape(dataframe.values, (len(dataframe), 1))
    model = KernelDensity(bandwidth=2, kernel='gaussian')
    model.fit(data)
    probability = model.score_samples(np.reshape(data_point, (1, 1)))

    return probability  # Return the log probability


def append_features(rising_events, falling_events, rising_features, falling_features, rising_features_aux, falling_features_aux, ris_compatibility_idx, fall_compatibility_idx):
    # Rising features
    rising_features['ris_transition_P'].append(rising_features_aux['ris_transition_P'].pop(ris_compatibility_idx))
    rising_features['ris_trans_duration'].append(rising_features_aux['ris_trans_duration'].pop(ris_compatibility_idx))
    rising_features['ris_trans_power_change_P'].append(rising_features_aux['ris_trans_power_change_P'].pop(ris_compatibility_idx))
    rising_features['ris_trans_spike_P'].append(rising_features_aux['ris_trans_spike_P'].pop(ris_compatibility_idx))
    # Falling features
    falling_features['fall_transition_P'].append(falling_features_aux['fall_transition_P'].pop(fall_compatibility_idx))
    falling_features['fall_trans_duration'].append(falling_features_aux['fall_trans_duration'].pop(fall_compatibility_idx))
    falling_features['fall_trans_power_change_P'].append(falling_features_aux['fall_trans_power_change_P'].pop(fall_compatibility_idx))
    falling_features['fall_trans_spike_P'].append(falling_features_aux['fall_trans_spike_P'].pop(fall_compatibility_idx))
    # Delete the previous matching conditions from rising and falling set
    # First condition:
    rising_features_aux['P_plus_high'].pop(ris_compatibility_idx)
    rising_features_aux['P_plus_low'].pop(ris_compatibility_idx)
    falling_features_aux['P_minus'].pop(fall_compatibility_idx)
    # Second condition:
    rising_features_aux['delta_P_plus_high'].pop(ris_compatibility_idx)
    rising_features_aux['delta_P_plus_low'].pop(ris_compatibility_idx)
    falling_features_aux['delta_P_minus'].pop(fall_compatibility_idx)
    # Third condition:
    rising_features_aux['feature_idx'].pop(ris_compatibility_idx)
    falling_features_aux['feature_idx'].pop(fall_compatibility_idx)
    # As the events are matched, both are deleted from the set
    rising_events.pop(ris_compatibility_idx)  # The last element of the rising events is deleted
    falling_events.pop(fall_compatibility_idx)  # Extract the falling event that was compatible to delete it
    rising_features_aux['ts'].pop(ris_compatibility_idx)
    falling_features_aux['ts'].pop(fall_compatibility_idx)

    return rising_events, falling_events, rising_features, falling_features, rising_features_aux, falling_features_aux


def append_rising_unmatched(rising_events, unmatched_rising_features, rising_features_aux, ris_compatibility_idx):
    print(f'For the rising event {rising_events[ris_compatibility_idx]}, a match was not found')
    unmatched_rising_features['ris_transition_P'].append(rising_features_aux['ris_transition_P'].pop(ris_compatibility_idx))
    unmatched_rising_features['ris_trans_duration'].append(rising_features_aux['ris_trans_duration'].pop(ris_compatibility_idx))
    unmatched_rising_features['ris_trans_power_change_P'].append(rising_features_aux['ris_trans_power_change_P'].pop(ris_compatibility_idx))
    unmatched_rising_features['ris_trans_spike_P'].append(rising_features_aux['ris_trans_spike_P'].pop(ris_compatibility_idx))
    # Delete the previous matching conditions from rising set
    # First condition:
    rising_features_aux['P_plus_high'].pop(ris_compatibility_idx)
    rising_features_aux['P_plus_low'].pop(ris_compatibility_idx)
    # Second condition:
    rising_features_aux['delta_P_plus_high'].pop(ris_compatibility_idx)
    rising_features_aux['delta_P_plus_low'].pop(ris_compatibility_idx)
    unmatched_rising_features['timestamp'].append(rising_events.pop(ris_compatibility_idx))  # The last element of the rising events is deleted
    # Third condition
    rising_features_aux['feature_idx'].pop(ris_compatibility_idx)
    return rising_events, unmatched_rising_features, rising_features_aux


def append_falling_unmatched(falling_events, unmatched_falling_features, falling_features_aux, fall_compatibility_idx):
    print(f'For the falling event {falling_events[fall_compatibility_idx]}, a match was not found')
    unmatched_falling_features['fall_transition_P'].append(falling_features_aux['fall_transition_P'].pop(fall_compatibility_idx))
    unmatched_falling_features['fall_trans_duration'].append(falling_features_aux['fall_trans_duration'].pop(fall_compatibility_idx))
    unmatched_falling_features['fall_trans_power_change_P'].append(falling_features_aux['fall_trans_power_change_P'].pop(fall_compatibility_idx))
    unmatched_falling_features['fall_trans_spike_P'].append(falling_features_aux['fall_trans_spike_P'].pop(fall_compatibility_idx))
    # Delete the previous matching conditions from falling set
    # First condition:
    falling_features_aux['P_minus'].pop(fall_compatibility_idx)
    # Second condition:
    falling_features_aux['delta_P_minus'].pop(fall_compatibility_idx)
    unmatched_falling_features['timestamp'].append(falling_events.pop(fall_compatibility_idx))  # The falling event is deleted
    # Third condition:
    falling_features_aux['feature_idx'].pop(fall_compatibility_idx)
    return falling_events, unmatched_falling_features, falling_features_aux


# %% Grouping all power intervals
appliances = ['oven1', 'oven2', 'microwave', 'kitchen outlets1', 'kitchen outlets2', 'bathroom gfi', 'washer dryer1', 'washer dryer2', 'fridge', 'dish washer']
df_power_intervals = pd.DataFrame()
app_corr = []  # To store the appliance power interval correspondence
for appliance in appliances:
    power_intervals = pd.read_csv(f'transitions first try/intervals_{appliance}.csv', index_col=0)
    df_power_intervals = pd.concat([df_power_intervals, power_intervals])
    # To store the appliance power interval correspondence
    app_corr_aux = [appliance for i in range(len(power_intervals))]
    app_corr = app_corr + app_corr_aux
# Lookup list to know for which power interval corresponds to the appliance
power_inter_look_up = {i: appliance for i, appliance in zip(range(len(df_power_intervals)), app_corr)}
power_ranges = [(df_power_intervals.iloc[i][0], df_power_intervals.iloc[i][1]) for i in range(len(df_power_intervals))]
# Creating a matrix to store the event correspondences
index = np.arange(len(power_ranges))
event_match_aux = pd.DataFrame(index=index)  # auxiliary event correspondences
event_match = pd.DataFrame(index=index)  # event correspondences
prob_tie_break_ris = pd.DataFrame(index=index)  # probability tie break matrix
rising_or_falling = pd.DataFrame(index=index)  # Record in the event correspondences if it was a rising (+1) or falling event (-1)
event_powers = pd.DataFrame(index=index)  # event power values
prediction_dict = {'timestamp': [], 'appliance': [], 'On_Off': []}  # Creating the dictionary for the predictions
# %% Load dataset

redd = DataSet('./REDD/redd.h5')
building_1_elec = redd.buildings[1].elec
df_main_1_bui_1 = next(building_1_elec[1].load())
df_main_2_bui_1 = next(building_1_elec[2].load())
df_oven_1_bui_1 = next(building_1_elec[3].load())
df_oven_2_bui_1 = next(building_1_elec[4].load())
df_microwave_bui_1 = next(building_1_elec[11].load())
df_kitchen_outlets_1_bui_1 = next(building_1_elec[15].load())
df_kitchen_outlets_2_bui_1 = next(building_1_elec[16].load())
df_bathroom_gfi_bui_1 = next(building_1_elec[12].load())
df_washer_dryer_1_bui_1 = next(building_1_elec[10].load())
df_washer_dryer_2_bui_1 = next(building_1_elec[20].load())
df_fridge_bui_1 = next(building_1_elec[5].load())
df_dish_washer_bui_1 = next(building_1_elec[6].load())
# Merging main1 and main2 (to include the dishwasher in the aggregate measurements)
df_mains_merged = df_main_1_bui_1['power']['apparent'] + df_main_2_bui_1['power']['apparent']
df_mains_merged = pd.DataFrame(df_mains_merged)
# Clean merged power signals just sum the appliances that will be considered for disaggregation
df_clean_merge = df_oven_1_bui_1['power']['active'] + df_oven_2_bui_1['power']['active'] + df_microwave_bui_1['power']['active'] \
                 + df_kitchen_outlets_1_bui_1['power']['active'] + df_kitchen_outlets_2_bui_1['power']['active'] + df_bathroom_gfi_bui_1['power']['active'] \
                 + df_washer_dryer_1_bui_1['power']['active'] + df_washer_dryer_2_bui_1['power']['active'] + df_fridge_bui_1['power']['active']
df_clean_merge = pd.DataFrame(df_clean_merge)

data_trans_main = {'main1': np.log(df_main_1_bui_1['power']['apparent'].mask(df_main_1_bui_1['power']['apparent'] <= 0)).fillna(0),
                   'main2': np.log(df_main_2_bui_1['power']['apparent'].mask(df_main_2_bui_1['power']['apparent'] <= 0)).fillna(0),
                   'merged': np.log(df_mains_merged['apparent'].mask(df_mains_merged['apparent'] <= 0)).fillna(0)}

data_trans_app = {'oven1': np.log(df_oven_1_bui_1['power']['active'].mask(df_oven_1_bui_1['power']['active'] <= 0)).fillna(0),
                  'oven2': np.log(df_oven_2_bui_1['power']['active'].mask(df_oven_2_bui_1['power']['active'] <= 0)).fillna(0),
                  'microwave': np.log(df_microwave_bui_1['power']['active'].mask(df_microwave_bui_1['power']['active'] <= 0)).fillna(0),
                  'kitchen outlets1': np.log(df_kitchen_outlets_1_bui_1['power']['active'].mask(df_kitchen_outlets_1_bui_1['power']['active'] <= 0)).fillna(0),
                  'kitchen outlets2': np.log(df_kitchen_outlets_2_bui_1['power']['active'].mask(df_kitchen_outlets_2_bui_1['power']['active'] <= 0)).fillna(0),
                  'bathroom gfi': np.log(df_bathroom_gfi_bui_1['power']['active'].mask(df_bathroom_gfi_bui_1['power']['active'] <= 0)).fillna(0),
                  'washer dryer1': np.log(df_washer_dryer_1_bui_1['power']['active'].mask(df_washer_dryer_1_bui_1['power']['active'] <= 0)).fillna(0),
                  'washer dryer2': np.log(df_washer_dryer_2_bui_1['power']['active'].mask(df_washer_dryer_2_bui_1['power']['active'] <= 0)).fillna(0),
                  'fridge': np.log(df_fridge_bui_1['power']['active'].mask(df_fridge_bui_1['power']['active'] <= 0)).fillna(0),
                  'dish washer': np.log(df_dish_washer_bui_1['power']['active'].mask(df_dish_washer_bui_1['power']['active'] <= 0)).fillna(0),
                  'clean': np.log(df_clean_merge['active'].mask(df_clean_merge['active'] <= 0)).fillna(0)}

df_power_trans_mains = pd.DataFrame(data_trans_main)
df_power_trans_app = pd.DataFrame(data_trans_app)
# %% Run algorithm with detection of active cycle


def run_algorithm(test_index, window_length, shift_count, power_samples, df_power_trans_mains, df_power_trans_app, event_match_aux, event_match,
                  prob_tie_break_ris, rising_or_falling, event_powers, prediction_dict, num_last_steady_sts=4, min_samples_steady_state=10,
                  max_samples_transient_state=3, max_window_size=1350, main=True):
    """

    :param test_index: Integer that indicates the start sample of the whole time series
    :param window_length: Integer that indicates the length of samples of the whole time series
    :param shift_count: Number of samples that must be shifted the window to start at the first sample of the second steady state
    :param power_samples: dictionary to store the power samples in the window
    :param df_power_trans_mains: dataframe with the mains electricity measurements
    :param df_power_trans_app: dataframe containing the transformed (log transformation) power of each of the appliances
    :param event_match_aux: dataframe matrix of auxiliary event correspondences
    :param event_match: dataframe matrix of event correspondences
    :param prob_tie_break_ris: dataframe probability tie break matrix
    :param rising_or_falling: dataframe to record in the event correspondences matrix if it was a rising (+1) or falling event (-1)
    :param event_powers: dataframe matrix of event power values
    :param prediction_dict: predictions dictionary
    :param num_last_steady_sts: Number of steady states to save the euclidean distances mean
    :param min_samples_steady_state: Integer that indicates the minimum number of data points that should be included in a group to be consider a cluster
    :param max_samples_transient_state: Integer to indicate the maximum number of transient samples to give less weight in the euclidean distance calculation
    :param max_window_size: Least number of samples to build up to 2 clusters
    :param main: Boolean that flags if the test is done on the disaggregated or aggregated power signal
    :return: shift_count, power_samples, prediction_dict
    """
    features = {'transition': [], 'log_transition': []}
    rising_features_aux = {'ris_trans_duration': [], 'ris_trans_spike_P': [],
                           'ris_trans_power_change_P': [], 'ris_transition_P': [],
                           'P_plus_high': [], 'P_plus_low': [],
                           'delta_P_plus_high': [], 'delta_P_plus_low': [],
                           'ts': [], 'feature_idx': []}
    falling_features_aux = {'fall_trans_duration': [], 'fall_trans_spike_P': [],
                            'fall_trans_power_change_P': [], 'fall_transition_P': [],
                            'P_minus': [], 'delta_P_minus': [],
                            'ts': [], 'feature_idx': []}
    rising_features = {'ris_trans_duration': [], 'ris_trans_spike_P': [],
                       'ris_trans_power_change_P': [], 'ris_transition_P': []}
    falling_features = {'fall_trans_duration': [], 'fall_trans_spike_P': [],
                        'fall_trans_power_change_P': [], 'fall_transition_P': []}
    unmatched_rising_features = {'timestamp': [], 'ris_trans_duration': [],
                                 'ris_trans_spike_P': [], 'ris_trans_power_change_P': [],
                                 'ris_transition_P': []}
    unmatched_falling_features = {'timestamp': [], 'fall_trans_duration': [],
                                  'fall_trans_spike_P': [], 'fall_trans_power_change_P': [],
                                  'fall_transition_P': []}
    events_ts = []  # To record the timestamp of when the events happened
    ON_OFF_event = []  # To record if was an On or OFF event
    euc_dist_means = []  # To record the previous euclidean distances from the past 2 consecutive steady states
    euc_dist_stds = []  # To record the previous euclidean distances' standard deviations from the past 2 consecutive steady states
    noise_detector = (min_samples_steady_state / 2) - 1  # Value that is linked with the minimum amount of steady state samples (therefore the frequency) to see if non consecutive samples are noise
    GND_ST = False  # Flag that symbolizes if the ground state was defined or not
    END_ACTIVE_CYCLE = False  # Flag to determine when is the end of an active cycle
    rising_events = []  # To store the rising events unix timestamp
    falling_events = []  # To store the falling events unix timestamp
    in_between_unmatched = []  # To store the event index of the unmatched rising/falling event
    window_size = (min_samples_steady_state * 2) + 1  # Least number of samples to build up to 2 clusters
    weights = [0.4, 0.5, 0.6]  # Give less weight to the 3 most separated samples (largest euclidean distances)
    shift = False  # Flag to shift the power samples to the first sample of the second steady state
    if main:
        data = df_power_trans_mains
        appliance = 'merged'
    else:
        data = df_power_trans_app
        appliance = 'fridge'

    for i in range(len(data[test_index:test_index + window_length])):
        if shift:
            shift_count += min_samples_steady_state  # It must shifted back the min number of samples of the steady state (DBSCAN parameter)
            shift = False  # Flag back to False to search again for other steady states
        power_samples['P_t'].append(data[appliance][test_index:test_index + window_length].iloc[i - shift_count])
        if len(power_samples['P_t']) >= window_size:
            df_power_samples = pd.DataFrame(power_samples, index=data[appliance].iloc[test_index:test_index + window_length].index[(i - shift_count) - window_size + 1:(i - shift_count) + 1])
            ts = df_power_samples.index
            ts = ts.astype('int64')
            # Calculate the euclidean distance between consecutive rows (samples), i.e. 0 and 1, 1 and 2, 2 and 3...
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
                    power_samples = {'P_t': []}  # Empty the power samples to detect new transient states
                    window_size = (min_samples_steady_state * 2) + 1  # Return to the initial window size
            else:
                print(window_size)
                shift = True  # It means that the samples must be shifted back to the first sample of the second steady state
                power_samples = {'P_t': []}  # Empty the power samples to detect new transient states
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
                # Define if it was an ON or OFF event
                ON_or_OFF = np.exp(df_power_samples.iloc[second_steady_st_inx].P_t.mean()) - np.exp(df_power_samples.iloc[first_steady_st_inx].P_t.mean())
                if ON_or_OFF > 0 and np.abs(ON_or_OFF) >= 40:  # Means an ON event because the appliance is consuming more AND filtering the low power events
                    ON_OFF_event.append(1)  # 1 symbolizes an ON event
                    # Power change from one steady state to another (i.e., transition power change)
                    rising_features_aux['ris_transition_P'].append(df_power_samples.iloc[second_steady_st_inx].P_t.mean() - df_power_samples.iloc[first_steady_st_inx].P_t.mean())
                    rising_features_aux['P_plus_high'].append(df_power_samples.iloc[first_steady_st_inx].P_t.max())
                    rising_features_aux['P_plus_low'].append(df_power_samples.iloc[first_steady_st_inx].P_t.min())
                    rising_features_aux['delta_P_plus_high'].append(np.abs(df_power_samples.iloc[second_steady_st_inx].P_t.max() - df_power_samples.iloc[first_steady_st_inx].P_t.min()))
                    rising_features_aux['delta_P_plus_low'].append(np.abs(df_power_samples.iloc[second_steady_st_inx].P_t.min() - df_power_samples.iloc[first_steady_st_inx].P_t.max()))
                    rising_features_aux['ts'].append(ts[first_steady_st_inx[-1]] // 1000000000)
                    rising_events.append(ts[first_steady_st_inx[-1]])  # Store the rising event unix timestamp
                    if not GND_ST:  # Definition of the ground state if it is not defined yet
                        GND_ST = True
                        ground_state_high = df_power_samples.iloc[first_steady_st_inx].P_t.max()  # Upper level of the ground state
                elif ON_or_OFF < 0 and np.abs(ON_or_OFF) >= 40:  # Means an OFF event because the appliance is consuming less AND filtering the low power events
                    ON_OFF_event.append(0)  # 0 symbolizes an OFF event
                    # Power change from one steady state to another (i.e., transition power change)
                    falling_features_aux['fall_transition_P'].append(df_power_samples.iloc[second_steady_st_inx].P_t.mean() - df_power_samples.iloc[first_steady_st_inx].P_t.mean())
                    falling_features_aux['P_minus'].append(df_power_samples.iloc[second_steady_st_inx].P_t.mean())
                    falling_features_aux['delta_P_minus'].append(np.abs(df_power_samples.iloc[second_steady_st_inx].P_t.mean() - df_power_samples.iloc[first_steady_st_inx].P_t.mean()))
                    falling_features_aux['ts'].append(ts[first_steady_st_inx[-1]] // 1000000000)
                    falling_events.append(ts[first_steady_st_inx[-1]])  # Store the falling event unix timestamp
                    # Check end of active cycle
                    candidate_ground_st = df_power_samples.iloc[second_steady_st_inx].P_t.mean()  # Defining the candidate ground state of the second steady state to see if the active cycle is over
                    if GND_ST:  # If ground state is detected check for end of active cycle
                        # End of active cycle is detected if candidate GND st. is lower or equal than the upper level of the first steady state
                        if candidate_ground_st <= ground_state_high + (ground_state_high * 0.05):
                            GND_ST = False  # Return to false the flag for the next active cycle
                            END_ACTIVE_CYCLE = True  # Active cycle has ended for that specific interval

                # Feature extraction (assuming in the time window there is just one transient state)
                # Power change from one steady state to another (i.e., transition power change)
                features['transition'].append(np.exp(df_power_samples.iloc[second_steady_st_inx].P_t.mean()) - np.exp(df_power_samples.iloc[first_steady_st_inx].P_t.mean()))
                features['log_transition'].append(df_power_samples.iloc[second_steady_st_inx].P_t.mean() - df_power_samples.iloc[first_steady_st_inx].P_t.mean())
                if {-1}.issubset(values):  # meaning that noise was detected from the transient states in between steady states
                    trans_idx = [idx for idx, element in enumerate(clusters.labels_) if element == -1]  # Check which are the indexes of the outliers (i.e., -1) (transient samples)
                    if ON_or_OFF > 0 and np.abs(ON_or_OFF) >= 40:  # Define if it is a rising or falling spike to determine its dimensions AND filtering the low power events
                        # Transient duration
                        trans_duration = pd.Timedelta.total_seconds(df_power_samples.index[second_steady_st_inx[0]] - df_power_samples.index[first_steady_st_inx[-1]])  # Timedelta to total seconds with pandas
                        rising_features_aux['ris_trans_duration'].append(trans_duration)
                        # Active and reactive power change of the transient state
                        trans_active_change = np.exp(df_power_samples.iloc[trans_idx].P_t.max()) - np.exp(df_power_samples.iloc[first_steady_st_inx].P_t.mean())
                        rising_features_aux['ris_trans_power_change_P'].append(trans_active_change)
                        # Transient spike for reactive and active powers
                        max_P = np.exp(df_power_samples.iloc[trans_idx].P_t.max())
                        min_P = np.exp(df_power_samples.iloc[second_steady_st_inx].P_t.mean())
                        rising_features_aux['ris_trans_spike_P'].append(max_P - min_P)
                        # Index correspondence with dictionary features
                        rising_features_aux['feature_idx'].append(len(features['transition']) - 1)
                    elif ON_or_OFF < 0 and np.abs(ON_or_OFF) >= 40:
                        # Transient duration
                        trans_duration = pd.Timedelta.total_seconds(df_power_samples.index[second_steady_st_inx[0]] - df_power_samples.index[first_steady_st_inx[-1]])  # Timedelta to total seconds with pandas
                        falling_features_aux['fall_trans_duration'].append(trans_duration)
                        # Active and reactive power change of the transient state
                        trans_active_change = np.exp(df_power_samples.iloc[first_steady_st_inx].P_t.mean()) - np.exp(df_power_samples.iloc[trans_idx].P_t.min())
                        # Transient spike for reactive and active powers
                        max_P = np.exp(df_power_samples.iloc[second_steady_st_inx].P_t.mean())
                        min_P = np.exp(df_power_samples.iloc[trans_idx].P_t.min())
                        # Active and reactive power change of the transient state
                        falling_features_aux['fall_trans_power_change_P'].append(trans_active_change)
                        # Transient spike for reactive and active powers
                        falling_features_aux['fall_trans_spike_P'].append(max_P - min_P)
                        # Index correspondence with dictionary features
                        falling_features_aux['feature_idx'].append(len(features['transition']) - 1)
                else:  # Meaning that there was not a detectable noise between the first and second steady state in the window (i.e. transient state) (depending on high or low freq. data)
                    if ON_or_OFF > 0 and np.abs(ON_or_OFF) >= 40:  # Define if it is a rising or falling spike to determine its dimensions AND filtering the low power events
                        rising_features_aux['ris_trans_duration'].append(0)
                        rising_features_aux['ris_trans_power_change_P'].append(0)
                        rising_features_aux['ris_trans_spike_P'].append(0)
                        # Index correspondence with dictionary features
                        rising_features_aux['feature_idx'].append(len(features['transition']) - 1)
                    elif ON_or_OFF < 0 and np.abs(ON_or_OFF) >= 40:
                        falling_features_aux['fall_trans_duration'].append(0)
                        falling_features_aux['fall_trans_power_change_P'].append(0)
                        falling_features_aux['fall_trans_spike_P'].append(0)
                        # Index correspondence with dictionary features
                        falling_features_aux['feature_idx'].append(len(features['transition']) - 1)
                if END_ACTIVE_CYCLE:  # If active cycle has ended start event matching
                    END_ACTIVE_CYCLE = False  # Return flag to false for the next active cycle
                    for idx, pair in enumerate(power_ranges):  # Scan all the power intervals to match the event
                        for j in range(len(features['transition'])):
                            if pair[0] <= np.abs(features['transition'][j]) <= pair[1]:  # Match the detected events with the power interval transition
                                event_match_aux.loc[idx, j] = 1
                                event_powers.loc[idx, j] = features['transition'][j]
                                if features['transition'][j] > 0:  # Rising event
                                    rising_or_falling.loc[idx, j] = 1
                                else:  # Falling event
                                    rising_or_falling.loc[idx, j] = -1
                            else:
                                if np.abs(features['transition'][j]) <= 40:  # Filter low power events (most likely noise <40W)
                                    continue
                                else:  # unmatched event
                                    # TODO: if the distance is greater than x consider the event as a maintenance event?
                                    # Extract the index from the high or low power interval that is nearest to the unmatched event
                                    candidate_high = [np.linalg.norm(np.abs(features['transition'][j]) - high_range) for high_range in [pow_range[1] for pow_range in power_ranges]]
                                    candidate_low = [np.linalg.norm(np.abs(features['transition'][j]) - low_range) for low_range in [pow_range[0] for pow_range in power_ranges]]
                                    # Index of minimum values from high and low intervals
                                    idx_high = np.argmin(candidate_high)
                                    idx_low = np.argmin(candidate_low)
                                    # Define if the unmatched event is near the high or low intervals looking at which of both is lower
                                    high_or_low = np.argmin((candidate_high[idx_high], candidate_low[idx_low]))
                                    if high_or_low == 0:  # The high interval is nearer the unmatched event
                                        transition_idx = idx_high
                                    else:
                                        transition_idx = idx_low
                                    # Match the event with the appliance power interval
                                    event_match_aux.loc[transition_idx, j] = 1
                                    event_powers.loc[transition_idx, j] = features['transition'][j]
                                    if features['transition'][j] > 0:  # Rising event
                                        rising_or_falling.loc[transition_idx, j] = 1
                                    else:  # Falling event
                                        rising_or_falling.loc[transition_idx, j] = -1
                    if len(event_match_aux.columns) == 0:  # Meaning that in that active cycle the events detected where noise (low power events)
                        continue  # Pass to the next active cycle
                    else:
                        # Filter the event correspondence with the dependent event probability
                        for event in event_match_aux.sum(axis=0).iloc[np.where(event_match_aux.sum(axis=0) > 1)].index:  # The indexing means where the sum of the columns (events) is greater than one. And it means that that event has more than one correspondence
                            # Extracting the appliances that were a match to that event
                            apps = [(power_inter_look_up[idx], idx) for idx in np.where(event_match_aux[event] == 1)[0]]
                            # Empty list to save the probabilities of each appliance
                            app_prob = []
                            for app in apps:
                                df_features = pd.read_csv(f'transitions first try/{app[0]}.csv', index_col=0)
                                if features['transition'][event] > 0:  # Rising event
                                    df_rising_features = pd.read_csv(f'transitions first try/{app[0]}_rising.csv', index_col=0)
                                    # Build for active power in watts
                                    df_features['watt_transition_low'] = np.abs(np.exp(df_features['high_state_min']) - np.exp(df_features['low_state_max']))
                                    df_features['watt_transition_high'] = np.abs(np.exp(df_features['high_state_max']) - np.exp(df_features['low_state_min']))
                                    transitions = pd.concat([df_features['watt_transition_low'], df_features['watt_transition_high']], axis=0)
                                    app_watt_transition = pd.DataFrame(transitions)
                                    # Time feature (feature engineering)
                                    df_features['timestamp'] = df_features['timestamp'] // 1000000000
                                    timestamp_s = df_features['timestamp']
                                    day = 24 * 60 * 60  # 24 hours x 60 minutes x 60 seconds
                                    df_features['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))  # Map it into a sin function
                                    df_features['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))  # Map it into a cos function
                                    # Search for the corresponding rising feature index
                                    rising_idx = np.where(np.array(rising_features_aux['feature_idx']) == event)[0][0]
                                    # Prob of transition power change
                                    prob1 = pdf(df_rising_features['ris_trans_power_change_P'], rising_features_aux['ris_trans_power_change_P'][rising_idx])
                                    # Prob transition spike
                                    prob2 = pdf(df_rising_features['ris_trans_spike_P'], rising_features_aux['ris_trans_spike_P'][rising_idx])
                                    # Prob transient duration
                                    prob3 = pdf(df_rising_features['ris_trans_duration'], rising_features_aux['ris_trans_duration'][rising_idx])
                                    # Prob transition interval
                                    # prob4 = pdf(df_features['transition'], features['log_transition'][event])
                                    prob4 = pdf(app_watt_transition, features['transition'][event])
                                    # Prob time feature
                                    prob5 = pdf(df_features['Day sin'], np.sin(rising_features_aux['ts'][rising_idx] * (2 * np.pi / day)))
                                    prob6 = pdf(df_features['Day cos'], np.cos(rising_features_aux['ts'][rising_idx] * (2 * np.pi / day)))
                                    # Dependent probability (multiplication of the probabilities-->that is the sum of the log probabilities)
                                    app_prob.append(prob1 + prob2 + prob3 + prob4 + prob5 + prob6)
                                else:  # Falling event is not considered because there is no significant data to calculate the probabilities for each appliance (at least 40% of the detected events)
                                    break
                            # Tie break based on rising events
                            if features['transition'][event] > 0:  # Rising event
                                app_idx = np.argmax(app_prob)  # Extract the index of the maximum probability in the list of dependent probabilities
                                which_app_idx = apps[app_idx][1]  # Extract the index of the power interval corresponding to the appliance (is a list of tuples the second element is the interval index)
                                prob_tie_break_ris.loc[which_app_idx, event] = 1
                        # Empty the dictionaries for the next active cycle
                        features = {'transition': [], 'log_transition': []}
                        ####--- Event pair matching: ----####
                        # Linking already the first rising event with the last falling event, we already know they are a match because the both open and close the active cycle respectively (ris_compatibility_idx=0, fall_compatibility_idx=len(falling_events)-1)
                        # Only matching condition-->for both events the appliance must coincide:
                        # Extract which rising event was in the event match aux matrix
                        ris_event_idx = rising_features_aux['feature_idx'][0]
                        # Extract which falling event was in the event match aux matrix
                        fall_event_idx = falling_features_aux['feature_idx'][len(falling_events) - 1]
                        # Extract the row where the probability of the rising event is higher (tie breaker from last step) (if there is a tie breaker because
                        # the event can only have one correspondance, so no need for the tie break)
                        if ris_event_idx in prob_tie_break_ris.columns:  # Meaning that there is more than one appliance related to that rising event
                            high_prob_idx = np.where(prob_tie_break_ris[ris_event_idx] == 1)[0][0]
                        else:  # There is only one appliance related to that event
                            high_prob_idx = np.where(event_match_aux[ris_event_idx] == 1)[0][0]
                        # Possible appliances that match with the falling event
                        poss_fall_app = [power_inter_look_up[idx] for idx in np.where(event_match_aux[fall_event_idx] == 1)[0]]
                        poss_fall_app_idx = [idx for idx in np.where(event_match_aux[fall_event_idx] == 1)[0]]
                        match_cond = power_inter_look_up[high_prob_idx] in poss_fall_app
                        if match_cond:
                            # Search for the falling event index corresponding to the appliance
                            fall_appliance_idx = np.where(power_inter_look_up[high_prob_idx] == np.array(poss_fall_app))
                            fall_appliance_idx = poss_fall_app_idx[fall_appliance_idx[0][0]]
                            # Constructing the correspondence matrix
                            event_match.loc[high_prob_idx, ris_event_idx] = 1
                            event_match.loc[fall_appliance_idx, fall_event_idx] = 1
                            # Assigning the correspondent prediction to the dictionary
                            # For the rising event (index 0)
                            prediction_dict['timestamp'].append(rising_events[0])
                            prediction_dict['appliance'].append(power_inter_look_up[high_prob_idx])
                            prediction_dict['On_Off'].append(1)
                            # For the falling event (index len(falling_events) - 1)
                            prediction_dict['timestamp'].append(falling_events[len(falling_events) - 1])
                            prediction_dict['appliance'].append(power_inter_look_up[fall_appliance_idx])
                            prediction_dict['On_Off'].append(0)
                            rising_events, falling_events, rising_features, falling_features, rising_features_aux, falling_features_aux = append_features(rising_events, falling_events, rising_features, falling_features, rising_features_aux, falling_features_aux, 0, len(falling_events) - 1)
                        while len(rising_events) != 0 and len(falling_events) != 0:  # It means that not all rising/falling events have found a match
                            # Check which is the adjacent falling element of the last rising event
                            ris_compatibility_idx = len(rising_events) - 1  # The last element index of the rising events
                            # Extract which rising event was in the event match aux matrix
                            ris_event_idx = rising_features_aux['feature_idx'][ris_compatibility_idx]
                            if ris_event_idx not in event_match_aux.columns:  # Meaning that in the previous step, that event was filtered and is probably noise (power below certain value)
                                rising_events, unmatched_rising_features, rising_features_aux = append_rising_unmatched(rising_events, unmatched_rising_features, rising_features_aux, ris_compatibility_idx)
                                continue  # Pass to the next iteration
                            subtraction = np.array(falling_events) - np.array(rising_events[-1])  # The subtraction between the unix timestamps that gives the least number is the adjacent event of that rising event
                            # The rising event cannot be associated with the previous falling events (i.e., a negative result from the subtraction), just with the succeeding
                            adjacent_events = [k[0] for k in sorted(enumerate(subtraction), key=lambda x: x[1]) if k[1] > 0]  # Find the indexes of the positive sorted integers
                            if len(adjacent_events) == 0:  # Meaning that for that rising event there are not future falling events (only preceding falling events), therefore, there is not a match for that rising event
                                in_between_unmatched.append((ris_event_idx, rising_events[ris_compatibility_idx], rising_features_aux['ris_transition_P'][ris_compatibility_idx]))
                                rising_events, unmatched_rising_features, rising_features_aux = append_rising_unmatched(rising_events, unmatched_rising_features, rising_features_aux, ris_compatibility_idx)
                                continue  # Pass to the next iteration
                            else:
                                fall_compatibility_idx = adjacent_events[0]
                                # Extract which falling event was in the event match aux matrix
                                fall_event_idx = falling_features_aux['feature_idx'][fall_compatibility_idx]
                                if fall_event_idx not in event_match_aux.columns:  # Meaning that in the previous step, that event was filtered and is probably noise (power below certain value)
                                    falling_events, unmatched_falling_features, falling_features_aux = append_falling_unmatched(falling_events, unmatched_falling_features, falling_features_aux, fall_compatibility_idx)
                                    continue  # Pass to the next iteration
                            # TODO extract the power consumption for the time interval of the match
                            # Check compatibility between events:
                            # First condition:
                            P_plus_low = rising_features_aux['P_plus_low'][ris_compatibility_idx]
                            P_plus_high = rising_features_aux['P_plus_high'][ris_compatibility_idx]
                            condition_1 = P_plus_low - (P_plus_low * 0.02) <= falling_features_aux['P_minus'][fall_compatibility_idx] <= P_plus_high + (P_plus_high * 0.02)
                            # Second condition:
                            delta_P_plus_low = rising_features_aux['delta_P_plus_low'][ris_compatibility_idx]
                            delta_P_plus_high = rising_features_aux['delta_P_plus_high'][ris_compatibility_idx]
                            condition_2 = delta_P_plus_low - (delta_P_plus_low * 0.02) <= falling_features_aux['delta_P_minus'][fall_compatibility_idx] <= delta_P_plus_high + (delta_P_plus_high * 0.02)
                            # Third condition: (APPLIANCE MATCHING)
                            # Extract the row where the probability of the rising event is higher (tie breaker from last step) (if there is a tie breaker because
                            # the event can only have one correspondance, so no need for the tie break)
                            if ris_event_idx in prob_tie_break_ris.columns:  # Meaning that there is more than one appliance related to that rising event
                                high_prob_idx = np.where(prob_tie_break_ris[ris_event_idx] == 1)[0][0]
                            else:  # There is only one appliance related to that event
                                high_prob_idx = np.where(event_match_aux[ris_event_idx] == 1)[0][0]
                            # See if the pair correspond to the same appliance (the event power interval was assigned to the same appliance)
                            # Possible appliances that match with the falling event
                            poss_fall_app = [power_inter_look_up[idx] for idx in np.where(event_match_aux[fall_event_idx] == 1)[0]]
                            poss_fall_app_idx = [idx for idx in np.where(event_match_aux[fall_event_idx] == 1)[0]]
                            # Appliances must coincide
                            condition_3 = power_inter_look_up[high_prob_idx] in poss_fall_app
                            if (condition_1 or condition_2) and condition_3:
                                # Search for the falling event index corresponding to the appliance
                                fall_appliance_idx = np.where(power_inter_look_up[high_prob_idx] == np.array(poss_fall_app))
                                fall_appliance_idx = poss_fall_app_idx[fall_appliance_idx[0][0]]
                                # Constructing the correspondence matrix
                                event_match.loc[high_prob_idx, ris_event_idx] = 1
                                event_match.loc[fall_appliance_idx, fall_event_idx] = 1
                                # Assigning the correspondent prediction to the dictionary
                                # For the rising event (index ris_compatibility_idx)
                                prediction_dict['timestamp'].append(rising_events[ris_compatibility_idx])
                                prediction_dict['appliance'].append(power_inter_look_up[high_prob_idx])
                                prediction_dict['On_Off'].append(1)
                                # For the falling event (index fall_compatibility_idx)
                                prediction_dict['timestamp'].append(falling_events[fall_compatibility_idx])
                                prediction_dict['appliance'].append(power_inter_look_up[fall_appliance_idx])
                                prediction_dict['On_Off'].append(0)
                                rising_events, falling_events, rising_features, falling_features, rising_features_aux, falling_features_aux = append_features(rising_events, falling_events, rising_features, falling_features, rising_features_aux, falling_features_aux, ris_compatibility_idx, fall_compatibility_idx)
                            else:  # Check the remaining events and slide the compatibility index
                                len_rising_before = len(rising_events)  # Save the length of rising events before searching for a match
                                # Pass to the next falling event if the first was not a match and there still falling events
                                for fall_index in adjacent_events[1:]:  # Evaluate the remaining event excluding the first element that was already checked
                                    condition_1 = P_plus_low - (P_plus_low * 0.02) <= falling_features_aux['P_minus'][fall_index] <= P_plus_high + (P_plus_high * 0.02)
                                    condition_2 = delta_P_plus_low - (delta_P_plus_low * 0.02) <= falling_features_aux['delta_P_minus'][fall_index] <= delta_P_plus_high + (delta_P_plus_high * 0.02)
                                    # Extract which falling event was in the event match aux matrix
                                    fall_event_idx = falling_features_aux['feature_idx'][fall_index]
                                    # Possible appliances that match with the falling event
                                    poss_fall_app = [power_inter_look_up[idx] for idx in np.where(event_match_aux[fall_event_idx] == 1)[0]]
                                    poss_fall_app_idx = [idx for idx in np.where(event_match_aux[fall_event_idx] == 1)[0]]
                                    # Appliances must coincide
                                    condition_3 = power_inter_look_up[high_prob_idx] in poss_fall_app
                                    if (condition_1 or condition_2) and condition_3:  # A match was found
                                        # Search for the falling event index corresponding to the appliance
                                        fall_appliance_idx = np.where(power_inter_look_up[high_prob_idx] == np.array(poss_fall_app))
                                        fall_appliance_idx = poss_fall_app_idx[fall_appliance_idx[0][0]]
                                        # Constructing the correspondence matrix
                                        event_match.loc[high_prob_idx, ris_event_idx] = 1
                                        event_match.loc[fall_appliance_idx, fall_event_idx] = 1
                                        # Assigning the correspondent prediction to the dictionary
                                        # For the rising event (index ris_compatibility_idx)
                                        prediction_dict['timestamp'].append(rising_events[ris_compatibility_idx])
                                        prediction_dict['appliance'].append(power_inter_look_up[high_prob_idx])
                                        prediction_dict['On_Off'].append(1)
                                        # For the falling event (index fall_index)
                                        prediction_dict['timestamp'].append(falling_events[fall_index])
                                        prediction_dict['appliance'].append(power_inter_look_up[fall_appliance_idx])
                                        prediction_dict['On_Off'].append(0)
                                        rising_events, falling_events, rising_features, falling_features, rising_features_aux, falling_features_aux = append_features(rising_events, falling_events, rising_features, falling_features, rising_features_aux, falling_features_aux, ris_compatibility_idx, fall_index)
                                        break
                                len_rising_after = len(rising_events)  # Save the length of rising events after searching for a match
                                if len_rising_after == len_rising_before:  # It means that the rising event did not found a match
                                    in_between_unmatched.append((ris_event_idx, rising_events[ris_compatibility_idx], rising_features_aux['ris_transition_P'][ris_compatibility_idx]))
                                    rising_events, unmatched_rising_features, rising_features_aux = append_rising_unmatched(rising_events, unmatched_rising_features, rising_features_aux, ris_compatibility_idx)
                        if len(falling_events) != 0 or len(rising_events) != 0:  # Meaning that not all rising/falling events were matched with a falling/rising event
                            for j in range(len(falling_events)):
                                in_between_unmatched.append((fall_event_idx, falling_events[0], falling_features_aux['fall_transition_P'][0]))
                                falling_events, unmatched_falling_features, falling_features_aux = append_falling_unmatched(falling_events, unmatched_falling_features, falling_features_aux, 0)
                            for j in range(len(rising_events)):
                                in_between_unmatched.append((ris_event_idx, rising_events[0], rising_features_aux['ris_transition_P'][0]))
                                rising_events, unmatched_rising_features, rising_features_aux = append_rising_unmatched(rising_events, unmatched_rising_features, rising_features_aux, 0)

                        # After the event matching pair, search for the unmatched non-filtered events that might have an appliance match (for the odd number of events in an active cycle)
                        if not all(event_idx[0] in event_match.columns for event_idx in in_between_unmatched):  # If at least one of the unmatched events (all command), is NOT in the event match columns we must search for the match
                            # The matched pair events with the corresponding appliance:
                            matched = [(power_inter_look_up[app_idx], event_match.columns[event_idx]) for app_idx, event_idx in zip(np.where(event_match == 1)[0], np.where(event_match == 1)[1])]
                            matched_events = {}
                            for key, value in matched:
                                matched_events.setdefault(key, []).append(value)
                            # Match first the unmatched appliance with the already paired appliances
                            unmatched_poss_apps = [(power_inter_look_up[app_idx], event_idx, app_idx, timestamp, feature) for event_idx, timestamp, feature in in_between_unmatched for app_idx in np.where(event_match_aux[event_idx] == 1)[0]]
                            match_apps = [element for element in unmatched_poss_apps if element[0] in matched_events.keys()]
                            # One last check to match the appliance with the already paired appliances: if the unmatched event is in between the paired events then that unmatched event belongs to the same appliance of the pair events
                            matched_apps = [app for app in match_apps if np.array(matched_events[app[0]]).min() < app[1] < np.array(matched_events[app[0]]).max()]
                            for app in matched_apps:
                                event_match.loc[app[2], app[1]] = 1
                                # Assigning the correspondent prediction to the dictionary
                                # For the rising event (index ris_compatibility_idx)
                                prediction_dict['timestamp'].append(app[3])
                                prediction_dict['appliance'].append(power_inter_look_up[app[2]])
                                if app[4] > 0:  # Rising event
                                    prediction_dict['On_Off'].append(1)
                                else:  # Falling event
                                    prediction_dict['On_Off'].append(0)
                        # Empty the unmatched events list
                        in_between_unmatched = []
                        # Empty the useful dataframes
                        event_match = pd.DataFrame(index=index)
                        event_match_aux = pd.DataFrame(index=index)
                        prob_tie_break_ris = pd.DataFrame(index=index)
                        event_powers = pd.DataFrame(index=index)
    return shift_count, power_samples, prediction_dict


power_samples = {'P_t': []}
min_samples_steady_state = 10
test_index = 0
window_length = 580507  # 580507 is 2011-04-25 13:22:15 for the aggregated signal
shift_count = min_samples_steady_state + 1  # Number of samples that must be shifted the window to start at the first sample of the second steady state
prediction_df = pd.DataFrame()

while shift_count > min_samples_steady_state:
    shift_count = 0
    shift_count, power_samples, prediction_dict = run_algorithm(test_index, window_length, shift_count, power_samples, df_power_trans_mains, df_power_trans_app, event_match_aux, event_match, prob_tie_break_ris, rising_or_falling, event_powers, prediction_dict, min_samples_steady_state=min_samples_steady_state)
    print('shift count:', shift_count)
    test_index = test_index + window_length - (shift_count + len(power_samples['P_t'])) - 1
    window_length = shift_count + len(power_samples['P_t'])
    power_samples = {'P_t': []}
    prediction_dict_df = pd.DataFrame(prediction_dict)
    prediction_df = pd.concat([prediction_df, prediction_dict_df])
    prediction_dict = {'timestamp': [], 'appliance': [], 'On_Off': []}
prediction_df.to_csv('predictions/pred.csv')
