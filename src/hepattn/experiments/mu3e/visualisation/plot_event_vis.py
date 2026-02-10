import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from hepattn.experiments.mu3e.data import Mu3eDataModule
from hepattn.experiments.mu3e.visualisation.event_vis import plot_mu3e_dual_view

config_path = Path('/home/xzcapnai/hepattn/hepattn/src/hepattn/experiments/mu3e/configs/mu3e_tracking.yaml')
out_dir = Path('/home/xzcapnai/hepattn/hepattn/src/hepattn/experiments/mu3e/visualisation/plots')
out_dir.mkdir(parents=True, exist_ok=True)

with config_path.open() as f:
    config =yaml.safe_load(f)

dm = Mu3eDataModule(**dict(config["data"]))
dm.setup("fit")
loader = dm.train_dataloader()

inputs, targets = next(iter(loader))

fig, eventID = plot_mu3e_dual_view(inputs, targets)

fig.savefig(out_dir / f'event_{eventID}_display.png')
plt.close(fig)