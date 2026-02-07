import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import time
from tqdm import tqdm
import uproot

def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0

def load_real_event_data(path_in):
    print('Loading global hit and track data, compiling events...')
    track_fields = ["tid", "pdg", "vx", "vy", "vz", "vt", "px", "py", "pz"]
    hit_fields = ["tid", "hid", "det", "pdg", "x", "y", "z", "time", "edep", "px", "py", "pz"]
        
    with uproot.open(path_in) as file:
        # pd.DataFrame of track and hit data from mu3e tree - only with chosen fields
        tracks_flat = file['mu3e_mc_tracks'].arrays(track_fields, library="pd")
        hits_flat = file['mu3e_mchits'].arrays(hit_fields, library="pd")

        # jagged awkward array - each entry in jagged array = list of hit indices in mchits tree, each list = one event
        hit_mapping = file["mu3e"]["hit_mc_i"].array(library='ak')

    # dataframe tidying    
    global_track_df = tracks_flat.rename(columns={'tid': 'trackID'})
    global_hit_df = hits_flat.rename(columns={"tid": "trackID", "hid": "hitID"})
        
    return global_track_df, global_hit_df, hit_mapping

def hit_sorter(col):
    return col.abs().astype(np.int64)

def root_to_parquet(
    in_dir: str, 
    train_dir: str, 
    val_dir: str, 
    test_dir: str, 
    event_batch: int,
    event_limit: int=None, 
    save_to_parquet: bool = True
):
    """
    Convert ROOT file to 2 parquet files (tracks and hits) with original eventID column.
    For hits without an associated event, assigns eventID : -1.
    """   
    path_in = Path(in_dir)
    
    path_train = Path(train_dir)
    path_val = Path(val_dir)
    path_test = Path(test_dir)
    
    path_train.mkdir(parents=True, exist_ok=True)
    path_val.mkdir(parents=True, exist_ok=True)
    path_test.mkdir(parents=True, exist_ok=True)
    
    ### loading hit and track dataframes ###
    global_track_df, global_hit_df, hit_mapping = load_real_event_data(path_in)
        
    ### mapping events ###
    print('Mapping events...')
    # flattening into one long numpy array of hit indices - all the hits in mchits that are included in monte carlo eventing
    flat_indices = np.asarray(ak.flatten(hit_mapping), dtype=np.int64)

    # creates array of eventIDs - maps onto flat_indices --- first N entries in event_ids = 0, maps to first N hit indices in flat_indices
    event_ids = np.repeat(np.arange(len(hit_mapping)), ak.num(hit_mapping))

    # above misses out all events that aren't accounted for in the mu3e tree - below assigns those events eventID = -1 -- makes easy to filter out later
    full_event_ids = np.full(len(global_hit_df), -1, dtype=np.int32)   # array of -1, length = total number of hits
    full_event_ids[flat_indices] = event_ids                           # applies eventIDs to all hit idx
    global_hit_df['eventID'] = full_event_ids                          # applied eventIDs to all hits in global hit dataframe

    tid_to_event = global_hit_df[global_hit_df['eventID'] != -1].set_index('trackID')['eventID']
    tid_to_event = tid_to_event[~tid_to_event.index.duplicated(keep='first')]
    global_track_df['eventID'] = global_track_df['trackID'].map(tid_to_event).fillna(-1).astype(int)
    
    ### applying event limit ###
    if event_limit:
        print(f'Limiting to {event_limit} compiled events...')
        global_hit_df = global_hit_df[global_hit_df["eventID"] < event_limit]
        global_track_df = global_track_df[global_track_df["eventID"] < event_limit]
        print('Ordering dataframes...')
    else:
        print('Ordering dataframes...')
        
    ### sorting eventID and trackID ###
    global_hit_df = global_hit_df.sort_values(['eventID', 'trackID'])
    global_track_df = global_track_df.sort_values(['eventID', 'trackID'])
    
    sensor_hits = global_hit_df[global_hit_df['det']==10]
    valid_sensor_hits = sensor_hits[sensor_hits['eventID']!=-1].reset_index(drop=True)
    valid_global_tracks = global_track_df[global_track_df['eventID']!=-1].reset_index(drop=True)
    
    ### dropping split tracks ###
    print('Dropping split tracks...')
    track_event_counts = valid_sensor_hits.groupby('trackID')['eventID'].nunique()
    split_tracks = track_event_counts[track_event_counts > 1].index

    valid_sensor_hits = valid_sensor_hits[~valid_sensor_hits['trackID'].isin(split_tracks)].reset_index(drop=True)
    valid_global_tracks = valid_global_tracks[~valid_global_tracks['trackID'].isin(split_tracks)].reset_index(drop=True)
    
    ### enforcing event level consistency ###
    hit_events = valid_sensor_hits['eventID'].unique()
    particle_events = valid_global_tracks['eventID'].unique()

    common_events = set(hit_events) & set(particle_events)

    valid_sensor_hits = valid_sensor_hits[valid_sensor_hits['eventID'].isin(common_events)]
    valid_global_tracks = valid_global_tracks[valid_global_tracks['eventID'].isin(common_events)]
    
    ### restricting to 4-hit or more particle tracks ###
    print('Restricting to 4-hit particle tracks...')
    counts = valid_sensor_hits['trackID'].value_counts()
    keep_particle_ids = counts[counts >= 4].index.to_numpy()

    valid_sensor_hits = valid_sensor_hits[valid_sensor_hits['trackID'].isin(keep_particle_ids)].reset_index(drop=True)
    valid_global_tracks = valid_global_tracks[valid_global_tracks['trackID'].isin(keep_particle_ids)].reset_index(drop=True)
    
    # sanity check
    bad = valid_sensor_hits.groupby('trackID').size()
    bad = bad[bad < 4]

    assert bad.empty, f"Non-4-hit tracks found: {bad.head()}"
    
    ### sorting by hitID within track ###
    print('Sorting hitID within tracks...')
    valid_sensor_hits = valid_sensor_hits.sort_values(by=['trackID', 'hitID'], key=hit_sorter)
    
    ### merge events into larger batch ###
    print(f'Merging events into batch of {event_batch}...')
    valid_sensor_hits['eventID_old'] = valid_sensor_hits['eventID']
    valid_global_tracks['eventID_old'] = valid_global_tracks['eventID']
    
    valid_sensor_hits['eventID'] = valid_sensor_hits['eventID_old'] // event_batch
    valid_global_tracks['eventID'] = valid_global_tracks['eventID_old'] // event_batch

    valid_global_tracks = valid_global_tracks.sort_values(
        ["eventID", "trackID"]
    ).reset_index(drop=True)

    ### sanity checks ###
    assert valid_sensor_hits.groupby("trackID")["eventID"].nunique().max() == 1
    assert set(valid_sensor_hits["eventID"].unique()) == set(
        valid_global_tracks["eventID"].unique()
    )
    
    print('Separating into train, val, test groups...')
    total_events = valid_sensor_hits['eventID'].max()
    train_hits = valid_sensor_hits[valid_sensor_hits["eventID"] < int(0.75*total_events)]
    train_tracks = valid_global_tracks[valid_global_tracks["eventID"] < int(0.75*total_events)]
    
    val_hits = valid_sensor_hits[(valid_sensor_hits["eventID"] >= int(0.75*total_events)) & (valid_sensor_hits["eventID"] < int(0.875*total_events))]
    val_tracks = valid_global_tracks[(valid_global_tracks["eventID"] >= int(0.75*total_events)) & (valid_global_tracks["eventID"] < int(0.875*total_events))]
    
    test_hits = valid_sensor_hits[valid_sensor_hits["eventID"] >= int(0.875*total_events)]
    test_tracks = valid_global_tracks[valid_global_tracks["eventID"] >= int(0.875*total_events)]

    ### saving to single parquet files ###
    if save_to_parquet:
        print("Saving hits to parquet...")
        train_hits.to_parquet(path_train / "all_hits.parquet", index=False)
        val_hits.to_parquet(path_val / "all_hits.parquet", index=False)    
        test_hits.to_parquet(path_test / "all_hits.parquet", index=False)

        print("Saving tracks to parquet...")
        train_tracks.to_parquet(path_train / "all_tracks.parquet", index=False)
        val_tracks.to_parquet(path_val / "all_tracks.parquet", index=False)    
        test_tracks.to_parquet(path_test / "all_tracks.parquet", index=False)

        print("\n--- Done ---")
        print(f"Saved {len(train_tracks['eventID'].unique())} events to:")
        print(f"  - {path_train / 'all_tracks.parquet'}")
        print(f"  - {path_train / 'all_hits.parquet'}")
        print()
        print(f"Saved {len(val_tracks['eventID'].unique())} events to:")
        print(f"  - {path_val / 'all_tracks.parquet'}")
        print(f"  - {path_val / 'all_hits.parquet'}")
        print()
        print(f"Saved {len(test_tracks['eventID'].unique())} events to:")
        print(f"  - {path_test / 'all_tracks.parquet'}")
        print(f"  - {path_test / 'all_hits.parquet'}")
    else:
        print("\n--- Done ---")
        print(f"Saved {len(valid_sensor_hits['eventID'].unique())} events to:")
        print(f"  - valid_sensor_hits")
        print(f"  - valid_global_tracks")
        print()
    
    return global_track_df, global_hit_df, valid_sensor_hits, valid_global_tracks