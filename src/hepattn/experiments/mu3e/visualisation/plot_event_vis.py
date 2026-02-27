import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from hepattn.experiments.mu3e.data import Mu3eDataModule
from hepattn.experiments.mu3e.visualisation.event_vis import plot_mu3e_dual_view, plot_mu3e_tri_view

config_path = Path('/home/xzcapnai/hepattn/hepattn/src/hepattn/experiments/mu3e/configs/mu3e_tracking.yaml')
outdir = Path('/home/xzcapnai/hepattn/hepattn/src/hepattn/experiments/mu3e/visualisation/plots')
outdir.mkdir(parents=True, exist_ok=True)

with config_path.open() as f:
    config = yaml.safe_load(f)

dm = Mu3eDataModule(**dict(config["data"]))
dm.setup("fit")
loader = dm.train_dataloader()

inputs, targets = next(iter(loader))
batch_number = inputs['hit_x'].shape[0]

fig, eventID = plot_mu3e_dual_view(inputs, targets, batch_number)
fig_2, eventID_2 = plot_mu3e_tri_view(inputs, targets, batch_number)

fig.savefig(outdir / f'dual_event_{eventID}_display.png')
fig_2.savefig(outdir / f'tri_event_{eventID}.display.png')
plt.close(fig)
plt.close(fig_2)