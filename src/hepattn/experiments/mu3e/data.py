from __future__ import annotations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from lightning import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader, Dataset

def pad_last_two_dim(x, L):  ### BROKEN NEEDS TO BE FIXED ###
    """Pads the last two dimensions to L x L (for N x N matrices)."""
    if x.size(-1) == L and x.size(-2) == L:
        return x
    pad_val = False if x.dtype == torch.bool else 0
    new_shape = (*x.shape[:-2], L, L)
    y = x.new_full(new_shape, pad_val)
    y[..., :x.size(-2), :x.size(-1)] = x
    return y

def is_valid_file(path):
    """ returns True if path exists and is non-empty"""
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0

def pad_last_dim(x, L):
    """ pads the last dimension of x to length L."""
    if x.size(-1) == L:
        return x
    pad_val = False if x.dtype == torch.bool else 0
    y = x.new_full((*x.shape[:-1], L), pad_val)
    y[..., :x.size(-1)] = x
    return y

def collate_dict(dicts):
    """ collates list of dicts of tensors into single dict. 
        pads tensors along last dimension before concatenating """
    out = {}
    for k in dicts[0]:
        xs = [d[k] for d in dicts]

        if torch.is_tensor(xs[0]) and xs[0].dim() > 0:
            Lmax = max(x.size(-1) for x in xs)
            xs = [pad_last_dim(x, Lmax) for x in xs]

        out[k] = torch.cat(xs, dim=0) if torch.is_tensor(xs[0]) else xs
    
    return out

def collate_dict_diagnostic(dicts):
    """ same as collate_dict but with error logging for shape mismatches
        prints informative messages before raising """
    out = {}
    for k in dicts[0]:
        xs = [d[k] for d in dicts]

        if torch.is_tensor(xs[0]) and xs[0].dim() > 0:
            # --- DIAGNOSTIC START ---
            shapes_before = [x.shape for x in xs]
            # Check if dimensions other than the last one are inconsistent
            # (e.g., trying to cat (1, 83) and (83,) will fail)
            dims_consistent = all(len(s) == len(shapes_before[0]) for s in shapes_before)
            
            if not dims_consistent:
                print(f"\n❌ DIMENSION MISMATCH on Key: '{k}'")
                print(f"Some tensors are 1D, some are 2D. Shapes: {shapes_before[:5]}")
            # --- DIAGNOSTIC END ---

            Lmax = max(x.size(-1) for x in xs)
            xs = [pad_last_dim(x, Lmax) for x in xs]

        if torch.is_tensor(xs[0]):
            try:
                out[k] = torch.cat(xs, dim=0)
            except RuntimeError as e:
                # This catches the "83 vs 43" error specifically
                print(f"\n" + "="*60)
                print(f"FAILED TO CONCATENATE KEY: '{k}'")
                print(f"Error: {e}")
                print(f"Lmax used for padding: {Lmax if 'Lmax' in locals() else 'N/A'}")
                print(f"Shapes AFTER padding (first 5): {[x.shape for x in xs[:5]]}")
                print("="*60 + "\n")
                raise e
        else:
            out[k] = xs
    
    return out

def mu3e_collate(batch):
    """ custom collate function for mu3e dataloader.
        unpacks (input, target) pairsand collates each separately. """
    inputs_list, targets_list = zip(*batch)
    return collate_dict_diagnostic(inputs_list), collate_dict_diagnostic(targets_list)

class Mu3eDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        targets: dict,
        num_events: int = -1,
        hit_volume_ids: list | None = None,
        feature_volume_ids: dict | None = None,
        particle_min_num_hits=None,
        event_max_num_particles=1000,
        strict_max_objects: bool = False,
        hit_eval_path: str | None = None,
        dummy_data: bool = False,    
    ):
        super().__init__()

        # Store dummy_data flag
        self.dummy_data = dummy_data

        # Set the global random sampling seed
        self.sampling_seed = 42
        np.random.seed(self.sampling_seed)  # noqa: NPY002

        # If using dummy data, skip file-based initialization
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

        # Load in event data
        self.dirpath = Path(dirpath)
        self.all_hits = pd.read_parquet(self.dirpath / Path("all_hits.parquet"))
        self.all_tracks = pd.read_parquet(self.dirpath / Path("all_tracks.parquet"))

        hits_counts = self.all_hits.groupby('eventID').size()
        particles_counts = self.all_tracks.groupby('eventID').size()

        valid_events = sorted(hits_counts.index.intersection(particles_counts.index).to_list())
        event_names = [f'event{ID:09d}' for ID in valid_events]
        num_events_available = len(valid_events)

        if num_events > num_events_available:
            msg = f"Requested {num_events} events, but only {num_events_available} are available in the directory {dirpath}."
            raise ValueError(msg)

        if num_events < 0:
            num_events = num_events_available

        if num_events == 0:
            raise ValueError("num_events must be greater than 0")

        self.sample_ids = valid_events[:num_events]

        # Metadata
        self.hit_eval_path = hit_eval_path
        self.inputs = inputs
        self.targets = targets
        self.num_events = num_events
        self.event_names = event_names[:num_events]

        # Setup hit eval file if specified
        if self.hit_eval_path:
            rank_zero_info(f"Using hit eval dataset {self.hit_eval_path}")

        # Hit level cuts
        self.hit_volume_ids = hit_volume_ids
        # Optional per-feature hit volume selections
        self.feature_volume_ids = feature_volume_ids

        # Particle level cuts
        self.particle_min_num_hits = particle_min_num_hits

        # Event level cuts
        self.event_max_num_particles = event_max_num_particles
        self.strict_max_objects = strict_max_objects

    def __len__(self):
        return int(self.num_events)
    
    def load_event(self, idx):
        sample_id = self.sample_ids[idx]

        hits = self.all_hits[self.all_hits['eventID'] == sample_id].copy()
        particles = self.all_tracks[self.all_tracks['eventID'] == sample_id].copy()

        # Make the detector volume selection
        if self.hit_volume_ids:
            hits = hits[hits["det"].isin(self.hit_volume_ids)].copy()

        # Scale spatial coordinates into 0-1 range
        for coord in ["x", "y", "z"]:
            hits[coord] *= 0.01

        for coord in ["px", "py", "pz"]:
            particles[coord] *= 0.01
            hits[coord] *= 0.01
            
        # Add extra hit fields
        hits["r"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2)
        hits["s"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2 + hits["z"] ** 2)
        hits["eta"] = np.arctanh(hits["z"] / hits["s"])

        hits["lambda"] = np.arccos(hits["z"] / hits["s"])
        hits["coslambda"] = hits["z"] / hits["s"]
        hits["sinlambda"] = hits["r"] / hits["s"]

        hits["phi"] = np.arctan2(hits["y"], hits["x"])
        hits["cosphi"] = hits["x"] / hits["s"]
        hits["sinphi"] = hits["y"] / hits["s"]
        
        hits["u"] = hits["x"] / (hits["x"] ** 2 + hits["y"] ** 2)
        hits["v"] = hits["y"] / (hits["x"] ** 2 + hits["y"] ** 2)

        # Add extra particle fields
        particles["p"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2 + particles["pz"] ** 2)
        particles["pt"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2)
        particles["eta"] = np.arctanh(particles["pz"] / particles["p"])
        particles["signed_pt"] = particles["charge"] * particles["pt"]

        particles["lambda"] = np.arccos(particles["pz"] / particles["p"])
        particles["coslambda"] = particles["pz"] / particles["p"]
        particles["sinlambda"] = particles["pt"] / particles["p"]

        particles["phi"] = np.arctan2(particles["py"], particles["px"])
        particles["sinphi"] = particles["py"] / particles["pt"] 
        particles["cosphi"] = particles["px"] / particles["pt"]

        # Apply particle cut based on hit content
        counts = hits["trackID"].value_counts()
        keep_particle_ids = counts[counts >= self.particle_min_num_hits].index.to_numpy()
        particles = particles[particles["trackID"].isin(keep_particle_ids)]

        # Re-index tracks per event
        particles["particle_idx"] = np.arange(len(particles))
        trackID_to_idx = dict(zip(particles['trackID'].values, particles['particle_idx'].values))
        assert len(trackID_to_idx) == len(particles), "trackID to particle_idx mapping is not one-to-one"

        hits["particle_idx"] = hits["trackID"].map(trackID_to_idx)
        hits = hits.dropna(subset=['particle_idx'])
        hits['particle_idx'] = hits['particle_idx'].astype(int)
        assert hits["particle_idx"].min() >= 0, "particle_idx to hit mapping failed"
        assert hits["particle_idx"].max() < len(particles), "particle_idx to hit mapping failed"

        # Mark which hits are on a valid / reconstructable particle, for the hit filter
        hits["on_valid_particle"] = hits["particle_idx"].isin(particles["particle_idx"])

        # Sanity checks
        assert len(particles) != 0, "No particles remaining - loosen selection!"
        assert len(hits) != 0, "No hits remaining - loosen selection!"
        assert particles["trackID"].nunique() == len(particles), "Non-unique particle ids"

        return hits, particles

    def __getitem__(self, idx):
        if self.dummy_data:
            return self._generate_dummy_data(idx)
        
        # Load the event
        hits, particles = self.load_event(idx)
        num_particles = len(particles)

        # Truncate particles if above event_max_num_particles
        if num_particles > self.event_max_num_particles:
            if self.strict_max_objects:
                message = f"Event {idx} has {num_particles}, but limit is {self.event_max_num_particles}"
                raise ValueError(message)
            particles = particles.iloc[:self.event_max_num_particles]
            # remove hits pointing to truncated particles
            valid_particle_idx = set(particles['particle_idx'].values)
            hits = hits[hits['particle_idx'].isin(valid_particle_idx)].copy()

        num_particles = len(particles)
        
        assert hits['particle_idx'].isin(particles['particle_idx']).all(), "Some hits point to particles that were truncated"

        # prepare input and target containers
        inputs = {}
        targets = {}

        # Build the input hits
        for feature, fields in self.inputs.items():
            feature_hits = hits

            # Valid mask is all True for the feature-specific subset
            inputs[f"{feature}_valid"] = torch.full((len(feature_hits),), True).unsqueeze(0)
            targets[f"{feature}_valid"] = inputs[f"{feature}_valid"]

            for field in fields:
                inputs[f"{feature}_{field}"] = torch.from_numpy(feature_hits[field].values).unsqueeze(0).half()

        # Create particle_valid mask by concatenating True and False arrays
        num_padding = self.event_max_num_particles - num_particles
        targets["particle_valid"] = torch.cat([
            torch.full((num_particles,), True), 
            torch.full((num_padding,), False)
        ]).unsqueeze(0)

        # Create the particle_hit_valid mask    
        particle_idx = torch.full((self.event_max_num_particles,), -1)
        particle_idx[:num_particles] = torch.arange(num_particles)
        hit_particle_idx = torch.from_numpy(hits["particle_idx"].values)
        targets["particle_hit_valid"] = (particle_idx.unsqueeze(-1) == hit_particle_idx.unsqueeze(-2)).unsqueeze(0)

        # Create the hit filter targets (note this ignores the event_max_num_particles filtering)
        for target_feature, fields in self.targets.items():
            if "on_valid_particle" in fields:
                targets[f"{target_feature}_on_valid_particle"] = torch.from_numpy(
                    hits["on_valid_particle"].to_numpy()
                ).unsqueeze(0)

        # Add sample ID
        targets["sample_id"] = torch.tensor([self.sample_ids[idx]], dtype=torch.int32)

        # Build the regression targets
        if "particle" in self.targets:
            for field in self.targets["particle"]:
                # Null target/particle slots are filled with nans
                # This acts as a sanity check that we correctly mask out null slots in the loss
                x = torch.full((self.event_max_num_particles,), torch.nan)
                x[:num_particles] = torch.from_numpy(
                    particles[field].to_numpy()[: self.event_max_num_particles]
                )
                targets[f"particle_{field}"] = x.unsqueeze(0)

        # Get particle index for each hit
        #hit_particle_idxs = hits["particle_idx"].values
        #num_hits = len(hit_particle_idxs)

        # Generate Succession Mask (N_hits, N_hits)
        #succession_mask = torch.zeros((num_hits, num_hits), dtype=torch.bool)

        # two hits are successors if they share particle_idx and are adjacent
        #same_particle = (hit_particle_idxs[:-1] == hit_particle_idxs[1:])

        # fill [i, i+1] positions
        #indices = np.where(same_particle)[0]
        #succession_mask[indices, indices+1] = True

        #targets["hit_succession_mask"] = succession_mask.unsqueeze(0)

        # __getitem__ shape 
        #print(f"\n[DEBUG __getitem__ idx={idx}]")
        #for k, v in inputs.items():
        #    if torch.is_tensor(v):
        #        print(f"  input[{k}]: {tuple(v.shape)} {v.dtype}")
        #    elif isinstance(v, dict):
        #        for kk, vv in v.items():
        #            if torch.is_tensor(vv):
        #                print(f"  input[{k}][{kk}]: {tuple(vv.shape)} {vv.dtype}")

        #for k, v in targets.items():
        #    if torch.is_tensor(v):
        #        print(f"  target[{k}]: {tuple(v.shape)} {v.dtype}")
        #    elif isinstance(v, dict):
        #        for kk, vv in v.items():
        #            if torch.is_tensor(vv):
        #                print(f"  target[{k}][{kk}]: {tuple(vv.shape)} {vv.dtype}")
                
        #for k, v in targets.items():
        #    if torch.is_tensor(v) and v.dim() == 1:
        #        # This is the danger zone! 
        #        # If one event has (83,) and another has (43,), cat(dim=0) will CRASH.
        #        print(f"⚠️ DANGER: Key '{k}' is 1D with shape {v.shape} and type {v.dtype}")

        return inputs, targets

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
            batch_size=150,
            collate_fn=mu3e_collate,
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