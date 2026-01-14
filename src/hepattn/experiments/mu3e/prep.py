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

def load_real_event_data(path_in, path_out):
    print('Loading global hit and track data, compiling events...')
    track_fields = ["tid", "pdg", "vx", "vy", "vz", "vt", "px", "py", "pz"]
    hit_fields = ["tid", "hid", "det", "pdg", "x", "y", "z", "time", "edep", "px", "py", "pz"]
    
    with uproot.open(Path(in_dir)) as file:
        # pd.DataFrame of track and hit data from mu3e tree - only with chosen fields
        tracks_flat = file['mu3e_mc_tracks'].arrays(track_fields, library="pd")
        hits_flat = file['mu3e_mchits'].arrays(hit_fields, library="pd")

        # jagged awkward array - each entry in jagged array = list of hit indices in mchits tree, each list = one event
        hit_mapping = file["mu3e"]["hit_mc_i"].array(library='ak') 

    # dataframe tidying    
    global_track_df = tracks_flat.rename(columns={'tid': 'trackID'})
    global_hit_df = hits_flat.rename(columns={"tid": "trackID", "hid": "hitID"})
        
    return global_track_df, global_hit_df, hit_mapping

def root_to_parquet(in_dir: str, out_dir: str, event_limit: int=None):
    """
    Convert ROOT file to 2 parquet files (tracks and hits) with original eventID column.
    For hits without an associated event, assigns eventID : -1.
    """   
    path_in = Path(in_dir)
    path_out = Path(out_dir)
    path_out.mkdir(parents=True, exist_ok=True)
    
    ### loading hit and track dataframes ###
    global_track_df, global_hit_df, hit_mapping = load_real_event_data(path_in, path_out)
    
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
        
    # sorting eventID and trackID
    global_hit_df = global_hit_df.sort_values(['eventID', 'trackID']).reset_index(drop=True)
    global_track_df = global_track_df.sort_values(['eventID', 'trackID']).reset_index(drop=True)
    
    # save to single parquet files
    print("Saving tracks to parquet...")
    global_track_df.to_parquet(path_out / "all_tracks.parquet", index=False)
    
    print("Saving hits to parquet...")
    global_hit_df.to_parquet(path_out / "all_hits.parquet", index=False)
    
    print("\n--- Done ---")
    print(f"Saved {len(global_track_df['eventID'].unique())} events to:")
    print(f"  - {path_out / 'all_tracks.parquet'}")
    print(f"  - {path_out / 'all_hits.parquet'}")
    
    return global_track_df, global_hit_df