import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

def t2n(x):
    '''torch to numpy convertor'''
    return x.detach().cpu().numpy()

def plot_mu3e_dual_view(inputs, targets):
    '''event visualiser - takes torch dataloader input and target dicts as input
       plots what model "sees"
       '''
    b = 0  # since batch_size = 1
    plt.style.use('default')

    # extract event id
    eventID = int(targets['sample_id'][b].item()) # using sample_id from targets dict
    
    # setup figure
    fig, (ax1, ax2) = plt.subplots(
        1, 2, 
        figsize=(15, 5), 
        dpi=200, 
        sharey=True, 
        gridspec_kw={'width_ratios': [200, 460]}
    )
    
    # scale by 100 for mm scale
    h_x = inputs["hit_x"][b].detach().cpu().numpy() * 100
    h_y = inputs["hit_y"][b].detach().cpu().numpy() * 100
    h_z = inputs["hit_z"][b].detach().cpu().numpy() * 100
    valid_mask = inputs["hit_valid"][b].detach().cpu().numpy().astype(bool)

    # XY Plane
    ax1.set_xlim(-100, 100)
    ax1.set_ylim(-100, 100)
    ax1.set_aspect('equal')
    ax1.plot(0, 0, 'x', color='black', alpha=0.5)
    for r in [23.3, 29.8, 73.9, 86.3]:
        ax1.add_patch(Circle((0, 0), r, color='black', lw=0.3, fill=False))
    ax1.add_patch(Circle((0, 0), 19, color='green', alpha=0.1, label='Target'))

    # ZY Plane
    ax2.set_xlim(-230, 210)
    ax2.set_ylim(-100, 100)
    ax2.set_aspect('equal')
    ax2.plot(0, 0, 'x', color='black', alpha=0.5)
    # detector layer Z-bounds
    ax2.hlines(y=[+23.3, -23.3], xmin=-62.35,  xmax=62.35,  color='black', lw=0.3)
    ax2.hlines(y=[+29.8, -29.8], xmin=-62.35,  xmax=62.35,  color='black', lw=0.3)
    ax2.hlines(y=[+73.9, -73.9], xmin=-175.95, xmax=175.95, color='black', lw=0.3)
    ax2.hlines(y=[+86.3, -86.3], xmin=-186.3,  xmax=186.3,  color='black', lw=0.3)
    
    # stopping target
    target1 = Polygon(np.array([[-50, 0], [0, 19], [0, -19]]), closed=True, color='green', alpha=0.1)
    target2 = Polygon(np.array([[50, 0], [0, 19], [0, -19]]), closed=True, color='green', alpha=0.1)
    ax2.add_patch(target1)
    ax2.add_patch(target2)

    # plot tracks
    particle_hit_val = targets["particle_hit_valid"][b].detach().cpu().numpy()
    particle_valid = targets["particle_valid"][b].detach().cpu().numpy().astype(bool)
    num_particles = np.sum(particle_valid)
    
    for p_idx in range(particle_hit_val.shape[0]):
        if not particle_valid[p_idx]: 
            continue
        
        # mask hits belonging to this particle that are also valid
        this_p_mask = particle_hit_val[p_idx].astype(bool) & valid_mask
        if not np.any(this_p_mask): 
            continue
        
        x_p, y_p, z_p = h_x[this_p_mask], h_y[this_p_mask], h_z[this_p_mask]

        # plot XY Projection
        line, = ax1.plot(x_p, y_p, '--', alpha=0.2, lw=1)
        color = line.get_color()
        ax1.plot(x_p, y_p, '.', color=color,markersize=5)

        # plot ZY Projection
        ax2.plot(z_p, y_p, '--', alpha=0.2, lw=1, color=color)
        ax2.plot(z_p, y_p, '.', color=color, markersize=5, label=f'P{p_idx}')

    # labels + polishing
    plt.suptitle(f'Mu3e Event {eventID} | Particles: {num_particles}', fontsize=14)
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('y [mm]')
    ax2.set_xlabel('z [mm]')
    ax1.grid(alpha=0.1)
    ax2.grid(alpha=0.1)
    ax2.legend(loc='center left', fontsize='x-small', title="Truth Tracks")
    
    plt.tight_layout()

    return fig, eventID