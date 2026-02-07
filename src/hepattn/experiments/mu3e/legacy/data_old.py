from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from lightning import LightningDataModule    #PyTorch lightning datamodule wrapper 
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader, Dataset

# checks file exists and is not empty
def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


class Mu3eDataset(Dataset):
    # sets up PyTorch dataset for Mu3e data, which events to load, how to filter hits/tracks, handles dummy test data
    def __init__(
        self,
        dirpath: str,
        inputs: dict,            # dictionary specifying which hit features to include in input tensors
        targets: dict,           # dictionary specifying which features should be inlcuded as model targets
        num_events: int = -1,
        hit_volume_ids: list = None,
        feature_volume_ids: dict = None,
        particle_min_num_hits: int = None,              # minimum number of hits a track must have to keep
        event_max_num_particles=1000,                   # maximum number of hits 
        strict_max_objects: bool = False,
        hit_eval_path: str | None = None,
        dummy_data: bool = False,
    ):
        super().__init__()
        
        # store dummy_data flag
        self.dummy_data = dummy_data

        # set the global random sampling seed
        self.sampling_seed = 42
        np.random.seed(self.sampling_seed)  # noqa: NPY002

        # if using dummy data, skip file-based initialization
        if self.dummy_data:
            rank_zero_info("Generating dummy data...")
            self.dirpath = Path(dirpath) if dirpath else Path()
            self.hit_eval_path = None
            self.inputs = inputs
            self.targets = targets
            self.num_events = max(num_events, 1) if num_events > 0 else 10
            self.event_names = [f"dummy_event_{i:06d}" for i in range(self.num_events)]
            self.sample_ids = list(range(self.num_events))
            self.hit_volume_ids = hit_volume_ids
            self.particle_min_num_hits = particle_min_num_hits
            self.event_max_num_particles = event_max_num_particles
            return

        # list of event names for mu3e parquet files
        hit_file_path = Path(dirpath) / 'all_hits.parquet'
        event_id_series = pd.read_parquet(hit_file_path, columns=['eventID'])['eventID']
        
        # get unique and sorted eventIDs
        unique_event_ids = event_id_series.unique()
        unique_event_ids.sort()
        
        # create standarised event names and sample ids
        event_names = [f'event{ID:09d}' for ID in unique_event_ids]
        sample_ids = unique_event_ids.tolist()

        # calculate the number of events that will actually be used
        num_events_available = len(event_names)
        
        # sanity checks 
        if num_events > num_events_available:
            msg = f"Requested {num_events} events, but only {num_events_available} are available in the directory {dirpath}."
            raise ValueError(msg)

        if num_events < 0:
            num_events = num_events_available

        if num_events == 0:
            raise ValueError("num_events must be greater than 0")

        # metadata
        self.dirpath = Path(dirpath)
        self.hit_eval_path = hit_eval_path
        self.inputs = inputs
        self.targets = targets
        self.num_events = num_events
        self.event_names = event_names[:num_events]

        self.all_hits = pd.read_parquet(self.dirpath / Path("all_hits.parquet"))
        self.all_tracks = pd.read_parquet(self.dirpath / Path("all_tracks.parquet"))

        hits_counts = self.all_hits.groupby('eventID').size()
        particles_counts = self.all_tracks.groupby('eventID').size()

        valid_events = hits_counts.index.intersection(particles_counts.index).to_list()
        self.sample_ids = valid_events[:num_events]
        
        #global_hit_files = Path(dirpath).glob("all_hits*.parquet")
        #global_track_files = Path(dirpath).glob("all_tracks*.parquet")

        #self.all_hits_list = [pd.read_parquet(p) for p in global_hit_files]
        #self.all_tracks_list = [pd.read_parquet(p) for p in global_track_files]
        
        # setup hit eval file if specified
        if self.hit_eval_path:
            rank_zero_info(f"Using hit eval dataset {self.hit_eval_path}")
            
        # store filtering parameters # 
        
        # hit level cuts
        self.hit_volume_ids = hit_volume_ids
        
        # optional per-feature hit volume selections
        self.feature_volume_ids = feature_volume_ids
        
        # track level cuts
        self.particle_min_num_hits = particle_min_num_hits

        # event level cuts
        self.event_max_num_particles = event_max_num_particles
        self.strict_max_objects = strict_max_objects
    
    def __len__(self):
        return int(self.num_events)
    
    def __getitem__(self, idx):
        if self.dummy_data:
            return self._generate_dummy_data(idx)

        # prepare containers
        inputs = {}
        targets = {}

        # load the event
        hits, particles = self.load_event(idx) 
        num_particles = len(particles)
        
        # build the input hits
        for feature, fields in self.inputs.items():
            feature_hits = hits

            # valid mask is all True for the feature-specific subset
            inputs[f"{feature}_valid"] = torch.full((len(feature_hits),), True).unsqueeze(0)
            targets[f"{feature}_valid"] = inputs[f"{feature}_valid"]
            
            for field in fields:
                inputs[f"{feature}_{field}"] = torch.from_numpy(feature_hits[field].values).unsqueeze(0).half()

        # create the targets for whether a track slot is used or not
        if num_particles > self.event_max_num_particles:
            if self.strict_max_objects:
                message = f"Event {idx} has {num_particles}, but limit is {self.event_max_num_particles}"
                raise ValueError(message)
            particles = particles.iloc[:self.event_max_num_particles]            
            num_particles = self.event_max_num_particles
            
        # create track_valid mask by concatenating True and False arrays
        num_padding = self.event_max_num_particles - num_particles
        targets["particle_valid"] = torch.cat([torch.full((num_particles,), True), torch.full((num_padding,), False)]).unsqueeze(0)

        # create the mask targets
        #selected_particle_ids = torch.from_numpy(particles["trackID"].values)
        #particle_ids = torch.cat([selected_particle_ids, torch.full((num_padding,), -999)])
        #hit_particle_ids = torch.from_numpy(hits["trackID"].values)
        #targets["particle_hit_valid"] = (particle_ids.unsqueeze(-1) == hit_particle_ids.unsqueeze(-2)).unsqueeze(0)

        # create mask targets
        particle_idx = torch.full((self.event_max_num_particles,), -1)
        particle_idx[:num_particles] = torch.arange(num_particles)
        hit_particle_idx = torch.from_numpy(hits["particle_idx"].values)
        targets["particle_hit_valid"] = (particle_idx.unsqueeze(-1) == hit_particle_idx.unsqueeze(-2)).unsqueeze(0) 

        # create the hit filter targets (note this ignores the event_max_num_tracks filtering)
        for target_feature, fields in self.targets.items():
            if "on_valid_particle" in fields:
                targets[f"{target_feature}_on_valid_particle"] = torch.from_numpy(hits["on_valid_particle"].to_numpy()).unsqueeze(0)

        # Add sample ID
        targets["sample_id"] = torch.tensor([self.sample_ids[idx]], dtype=torch.int32)

        # Build the regression targets
        if "particle" in self.targets:
            for field in self.targets["particle"]:
                # Null target/track slots are filled with nans
                x = torch.full((self.event_max_num_particles,), torch.nan)
                x[:num_particles] = torch.from_numpy(particles[field].to_numpy()[: self.event_max_num_particles])
                targets[f"particle_{field}"] = x.unsqueeze(0)
        
        assert len(particles) > 0, f"Event {idx} has no particles"

        return inputs, targets

    def load_event(self, idx):
        sample_id = self.sample_ids[idx]
        event_name = self.event_names[idx]

        # load data for the specific event
        hits = self.all_hits[self.all_hits['eventID'] == sample_id].copy()
        particles = self.all_tracks[self.all_tracks['eventID'] == sample_id].copy()
        
        # make the detector volume selection
        if self.hit_volume_ids:
            hits = hits[hits["det"].isin(self.hit_volume_ids)].copy()

        # add extra hit fields (geometric features)
        hits["r"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2)
        hits["s"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2 + hits["z"] ** 2)
        hits["lambda"] = np.arccos(hits["z"] / hits["s"])                       # use our variable lambda
        hits["phi"] = np.arctan2(hits["y"], hits["x"])
        hits["eta"] = -np.log(np.tan(hits["lambda"] / 2))
        hits["u"] = hits["x"] / (hits["x"] ** 2 + hits["y"] ** 2)
        hits["v"] = hits["y"] / (hits["x"] ** 2 + hits["y"] ** 2)

        # add extra track fields (kinematic features)
        particles["p"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2 + particles["pz"] ** 2)
        particles["pt"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2)
        particles["eta"] = np.arctanh(particles["pz"] / particles["p"])
        particles["lambda"] = np.arccos(particles["pz"] / particles["p"])
        particles["phi"] = np.arctan2(particles["py"], particles["px"])
        particles["coslambda"] = np.cos(particles["lambda"])
        particles["sinlambda"] = np.sin(particles["lambda"])
        particles["cosphi"] = np.cos(particles["phi"])
        particles["sinphi"] = np.sin(particles["phi"])

        # apply track cut based on hit content (minimum hits / naked tracks)
        counts = hits["trackID"].value_counts()
        keep_particle_ids = counts[counts >= self.particle_min_num_hits].index.to_numpy()
        particles = particles[particles["trackID"].isin(keep_particle_ids)].reset_index(drop=True)

        # re-index tracks per event
        particles["particle_idx"] = np.arange(len(particles))
        track_id_to_idx = dict(zip(particles['trackID'].values, particles["particle_idx"].values))
        #assert len(track_id_to_idx) == len(particles), "trackID -> particle_idx mapping is not one-to-one"

        hits["particle_idx"] = hits["trackID"].map(track_id_to_idx)
        hits = hits.dropna(subset=["particle_idx"])
        hits["particle_idx"] = hits["particle_idx"].astype(int)
        #assert hits["particle_idx"].min() >= 0, "particle_idx to hit mapping failed"
        #assert hits["particle_idx"].max() < len(particles), "particle_idx to hit mapping failed"

        # mark which hits are on a valid / reconstructable track, for the hit filter
        hits["on_valid_particle"] = hits["particle_idx"].isin(particles["particle_idx"])

        # sanity checks
        assert len(particles) != 0, "No particles remaining - loosen selection!"
        assert len(hits) != 0, "No hits remaining - loosen selection!"
        assert particles["particle_idx"].nunique() == len(particles), "Non-unique track ids"

        #if len(particles) == 0  and len(hits) == 0:
        #    raise RuntimeError(f"Warning: Event {sample_id} has no valid hit or particles, skipping...")
        #    return None, None

        return hits, particles
    
    def _generate_dummy_data(self, idx):
        """Generate completely random dummy data for CI testing."""
        inputs = {}
        targets = {}

        # Create random number generator
        rng = np.random.default_rng(self.sampling_seed + idx)

        # Generate random number of hits (between 10 and 100)
        num_hits = rng.integers(10, 101)

        # Generate random number of tracks (up to event_max_num_tracks)
        num_particles = rng.integers(1, min(self.event_max_num_particles + 1, 101))

        # Build the input hits with random data
        for feature, fields in self.inputs.items():
            inputs[f"{feature}_valid"] = torch.full((num_hits,), True).unsqueeze(0)
            targets[f"{feature}_valid"] = inputs[f"{feature}_valid"]

            for field in fields:
                # Generate random normal data for all fields
                data = rng.standard_normal(num_hits)
                inputs[f"{feature}_{field}"] = torch.from_numpy(data).unsqueeze(0).to(torch.float32)

        # Build the targets for whether a track slot is used or not
        targets["particle_valid"] = torch.full((self.event_max_num_particles,), False)
        targets["particle_valid"][:num_particles] = True
        targets["particle_valid"] = targets["particle_valid"].unsqueeze(0)

        # Build dummy track IDs
        particle_ids = torch.arange(num_particles, dtype=torch.long)
        particle_ids = torch.cat([particle_ids, -999 * torch.ones(self.event_max_num_particles - num_particles)])

        # Assign random track IDs to hits
        hit_particle_ids = torch.randint(0, num_particles, (num_hits,))

        # Create the mask targets
        targets["particle_hit_valid"] = (particle_ids.unsqueeze(-1) == hit_particle_ids.unsqueeze(-2)).unsqueeze(0)

        # Create the hit filter targets (random boolean)
        targets["hit_on_valid_particle"] = torch.randint(0, 2, (num_hits,), dtype=torch.bool).unsqueeze(0)

        # Add sample ID
        targets["sample_id"] = torch.tensor([idx], dtype=torch.int32)

        # Build the regression targets
        if "particle" in self.targets:
            for field in self.targets["particle"]:
                # Generate random track data
                x = torch.full((self.event_max_num_particles,), torch.nan)
                data = rng.standard_normal(num_particles)
                x[:num_particles] = torch.from_numpy(data)
                targets[f"particle_{field}"] = x.unsqueeze(0)

        return inputs, targets

class Mu3eDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        test_dir: str | None = None,
        pin_memory: bool = True,
        hit_eval_train: str | None = None,
        hit_eval_val: str | None = None,
        hit_eval_test: str | None = None,
        **kwargs,
    ):
        super().__init__()

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.pin_memory = pin_memory
        self.hit_eval_train = hit_eval_train
        self.hit_eval_val = hit_eval_val
        self.hit_eval_test = hit_eval_test
        self.kwargs = kwargs

    def setup(self, stage: str):
        if stage in {"fit", "test"}:
            self.train_dataset = Mu3eDataset(
                dirpath=self.train_dir,
                num_events=self.num_train,
                hit_eval_path=self.hit_eval_train,
                **self.kwargs,
            )

        if stage == "fit":
            self.val_dataset = Mu3eDataset(
                dirpath=self.val_dir,
                num_events=self.num_val,
                hit_eval_path=self.hit_eval_val,
                **self.kwargs,
            )

        # Only print train/val dataset details when actually training
        if stage == "fit":
            rank_zero_info(f"Created training dataset with {len(self.train_dataset):,} events")
            rank_zero_info(f"Created validation dataset with {len(self.val_dataset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"

            self.test_dataset = Mu3eDataset(
                dirpath=self.test_dir,
                num_events=self.num_test,
                hit_eval_path=self.hit_eval_test,
                **self.kwargs,
            )
            rank_zero_info(f"Created test dataset with {len(self.test_dataset):,} events")

    def get_dataloader(self, stage: str, dataset: Mu3eDataset, shuffle: bool):
        return DataLoader(
            dataset=dataset,
            batch_size=None,
            collate_fn=None,
            sampler=None,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dataset, stage="fit", shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dataset, stage="test", shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dataset, stage="test", shuffle=False)
