import numpy as np
import torch
import matplotlib.pyplot as plt
import textwrap
import matplotlib.animation as animation
import matplotlib.patches as _mpl_patches

from mplsoccer import Pitch as SoccerPitchMPL


def fn_poss_idx(ball_xy, player_xy, trsh=0.1, eps=1e-6):
    ball_dists = torch.linalg.norm(ball_xy - player_xy, dim=-1)  # (B, T, N-1)
    ball_dists = torch.cat([torch.zeros_like(ball_dists[:, :, [0]]), ball_dists], dim=-1)
    ball_dists[:, :, 0] = trsh
    nan_mask = torch.isnan(ball_dists)
    if nan_mask.any():
        # Put 1000 in the nan values
        ball_dists[nan_mask] = 1000

    poss_matrix = ((ball_dists == ball_dists.min(dim=-1).values.unsqueeze(-1)) & (ball_dists <= trsh)).float()   # (B, T, N)
    more_than_one_poss = poss_matrix.sum(-1) > 1
    if more_than_one_poss.any():
        # Put ball as possessor if there is more than one possessor
        # It's like a duel or handoff between team-mates
        poss_matrix[more_than_one_poss] = 0
        b_idx, t_idx = torch.where(more_than_one_poss)
        poss_matrix[b_idx, t_idx, 0] = 1
    assert (poss_matrix.sum(-1) == 1).all()

    poss_idx = torch.argmax(poss_matrix, dim=-1)

    return poss_matrix, poss_idx

def poss_processing(seq_unnorm, labels, poss_threshold=1.5):

    poss_matrix, poss_idx = fn_poss_idx(
        seq_unnorm[:, :, [0], :2], seq_unnorm[:, :, 1:, :2], trsh=poss_threshold)
    for i in range(len(labels)):
        labels[i]['poss_idx'] = poss_idx[i]
        if 'state_int' in labels[i].keys():
            # If the state is 'pass', set the possessor to 0
            pass_frames = (labels[i]['state_int'] == 1)
            labels[i]['poss_idx'][pass_frames] = 0

    return labels

def poss_conditioning(sequences, labels, name='guidposs'):
    sum_ = 0
    for i in range(len(labels)):
        high_cond = labels[i]['poss_idx'][labels[i]['poss_idx'] != 0]
        if name == 'guidposs':
            high_cond = torch.unique_consecutive(high_cond)
        else:
            raise NotImplementedError(f'Guidance {name} not implemented')
        sum_ += high_cond.shape[0]
        # Fill with 0s the high_cond values until the end of the sequence
        high_cond = torch.cat([high_cond, torch.zeros(sequences[i].shape[0] - len(high_cond))])
        labels[i]['high_cond'] = high_cond
    print(f'Average number of players touching the ball: {sum_ / len(labels)}')

    return labels



def _draw_pitch(ax, pitch_type, pitch_size):
	"""Draw a sports surface on ax.

	pitch_type:
	- 'soccer' (soccer)
	- 'nba'
	- 'american_football'

	pitch_size:
	- (length, width) in the same units as your trajectories.
	"""
	L, W = float(pitch_size[0]), float(pitch_size[1])

	if pitch_type in ['soccer']:
		# Draw soccer pitch using mplsoccer (as requested).
		pad_left = 0.0
		pad_right = 0.0
		pitch = SoccerPitchMPL(
			pitch_type='uefa',
			pitch_color='#c2d59d',
			line_color='white',
			stripe=False,
			pitch_length=L,
			pitch_width=W,
			pad_left=pad_left,
			pad_right=pad_right,
			axis=True,
		)
		pitch.draw(ax=ax)
		return ax

	if pitch_type == 'nba':
		# Matplotlib-only NBA court. We reuse the provided 94x50 ft geometry and scale to pitch_size.
		ft_L, ft_W = 94.0, 50.0
		sx = L / ft_L
		sy = W / ft_W
		tr = lambda x, y: (x * sx, y * sy)

		line = '#000000'
		wood = '#f6edd9'
		hoop_col = '#f55b33'

		# Floor
		ax.add_patch(_mpl_patches.Rectangle((0, 0), L, W, fc=wood, ec='none', zorder=0))

		# Boundaries
		ax.plot([0, L], [0, 0], color=line, lw=2)
		ax.plot([0, L], [W, W], color=line, lw=2)
		ax.plot([0, 0], [0, W], color=line, lw=2)
		ax.plot([L, L], [0, W], color=line, lw=2)

		# Center line and circle
		ax.plot([L / 2, L / 2], [0, W], color=line, lw=2)
		cx, cy = tr(ft_L / 2, ft_W / 2)
		ax.add_patch(_mpl_patches.Circle((cx, cy), 6.0 * min(sx, sy), fill=False, ec=line, lw=2))

		# Key dimensions (in feet, then scaled)
		hoop_r = 0.75
		hoop_x_l, hoop_x_r = 5.25, ft_L - 5.25
		hoop_y = ft_W / 2
		back_x = 4.0
		back_half = 3.0
		lane_w = 16.0
		lane_y0 = (ft_W - lane_w) / 2
		ft_x = 19.0
		ft_r = 6.0
		ra_r = 4.0
		three_r = 23.75
		corner_y_margin = 3.0
		corner_three_x = 14.0

		# Hoops and backboards
		hx, hy = tr(hoop_x_l, hoop_y)
		ax.add_patch(_mpl_patches.Circle((hx, hy), hoop_r * min(sx, sy), fill=False, ec=hoop_col, lw=2))
		bx, _ = tr(back_x, 0)
		ax.plot([bx, bx], [tr(0, hoop_y - back_half)[1], tr(0, hoop_y + back_half)[1]], color=line, lw=2)

		hx, hy = tr(hoop_x_r, hoop_y)
		ax.add_patch(_mpl_patches.Circle((hx, hy), hoop_r * min(sx, sy), fill=False, ec=hoop_col, lw=2))
		bx, _ = tr(ft_L - back_x, 0)
		ax.plot([bx, bx], [tr(0, hoop_y - back_half)[1], tr(0, hoop_y + back_half)[1]], color=line, lw=2)

		# Paint (lanes) and free-throw
		x0, y0 = tr(0, lane_y0)
		x1, y1 = tr(ft_x, lane_y0 + lane_w)
		ax.add_patch(_mpl_patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=line, lw=2))
		x0, y0 = tr(ft_L - ft_x, lane_y0)
		x1, y1 = tr(ft_L, lane_y0 + lane_w)
		ax.add_patch(_mpl_patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=line, lw=2))

		fx, fy = tr(ft_x, hoop_y)
		ax.add_patch(_mpl_patches.Circle((fx, fy), ft_r * min(sx, sy), fill=False, ec=line, lw=2))
		fx, fy = tr(ft_L - ft_x, hoop_y)
		ax.add_patch(_mpl_patches.Circle((fx, fy), ft_r * min(sx, sy), fill=False, ec=line, lw=2))

		# Restricted area arcs
		hx, hy = tr(hoop_x_l, hoop_y)
		ax.add_patch(_mpl_patches.Arc((hx, hy), 2 * ra_r * sx, 2 * ra_r * sy, angle=0, theta1=-90, theta2=90, ec=line, lw=2))
		hx, hy = tr(hoop_x_r, hoop_y)
		ax.add_patch(_mpl_patches.Arc((hx, hy), 2 * ra_r * sx, 2 * ra_r * sy, angle=0, theta1=90, theta2=270, ec=line, lw=2))

		# 3-point lines
		# Left side corners
		ax.plot([0, tr(corner_three_x, 0)[0]], [tr(0, corner_y_margin)[1], tr(0, corner_y_margin)[1]], color=line, lw=2)
		ax.plot([0, tr(corner_three_x, 0)[0]], [tr(0, ft_W - corner_y_margin)[1], tr(0, ft_W - corner_y_margin)[1]], color=line, lw=2)
		# Left arc
		theta1_l = np.degrees(np.arctan2(corner_y_margin - hoop_y, corner_three_x - hoop_x_l))
		theta2_l = np.degrees(np.arctan2(ft_W - corner_y_margin - hoop_y, corner_three_x - hoop_x_l))
		hx, hy = tr(hoop_x_l, hoop_y)
		ax.add_patch(_mpl_patches.Arc((hx, hy), 2 * three_r * sx, 2 * three_r * sy, angle=0, theta1=theta1_l, theta2=theta2_l, ec=line, lw=2))

		# Right side corners
		ax.plot([L, tr(ft_L - corner_three_x, 0)[0]], [tr(0, corner_y_margin)[1], tr(0, corner_y_margin)[1]], color=line, lw=2)
		ax.plot([L, tr(ft_L - corner_three_x, 0)[0]], [tr(0, ft_W - corner_y_margin)[1], tr(0, ft_W - corner_y_margin)[1]], color=line, lw=2)
		# Right arc
		theta1_r = np.degrees(np.arctan2(ft_W - corner_y_margin - hoop_y, ft_L - corner_three_x - hoop_x_r))
		theta2_r = np.degrees(np.arctan2(corner_y_margin - hoop_y, ft_L - corner_three_x - hoop_x_r))
		hx, hy = tr(hoop_x_r, hoop_y)
		ax.add_patch(_mpl_patches.Arc((hx, hy), 2 * three_r * sx, 2 * three_r * sy, angle=0, theta1=theta1_r, theta2=theta2_r, ec=line, lw=2))

		ax.set_xlim(0, L)
		ax.set_ylim(0, W)
		ax.set_aspect('equal')
		ax.axis('off')
		return ax

	if pitch_type == 'american_football':
		# Clean fallback: 120x53.3 (including endzones)
		field_length = 120.0
		field_width = 53.3
		endzone_depth = 10.0
		sx = L / field_length
		sy = W / field_width
		tr = lambda x, y: (x * sx, y * sy)

		# Field + endzones
		ax.add_patch(_mpl_patches.Rectangle((0, 0), L, W, facecolor="#85ad7f", zorder=0))
		ax.add_patch(_mpl_patches.Rectangle(tr(0, 0), endzone_depth * sx, W, facecolor="#436845", zorder=1))
		ax.add_patch(_mpl_patches.Rectangle(tr(field_length - endzone_depth, 0), endzone_depth * sx, W, facecolor="#436845", zorder=1))

		# Boundaries
		ax.plot([0, L], [0, 0], color='white', lw=2)
		ax.plot([0, L], [W, W], color='white', lw=2)
		ax.plot([0, 0], [0, W], color='white', lw=2)
		ax.plot([L, L], [0, W], color='white', lw=2)

		# Major yard lines (every 10 yards)
		for x in range(int(endzone_depth), int(field_length - endzone_depth) + 1, 10):
			xp, _ = tr(x, 0)
			ax.plot([xp, xp], [0, W], color='white', lw=2)

		# Hash marks (every yard)
		for x in range(int(endzone_depth), int(field_length - endzone_depth) + 1):
			xp, _ = tr(x, 0)
			# Bottom
			ax.plot([xp, xp], [tr(0, 1)[1], tr(0, 2)[1]], color='white', lw=1)
			# Top
			ax.plot([xp, xp], [tr(0, field_width - 2)[1], tr(0, field_width - 1)[1]], color='white', lw=1)
			# Middle
			ax.plot([xp, xp], [tr(0, 23.16)[1], tr(0, 24.16)[1]], color='white', lw=1)
			ax.plot([xp, xp], [tr(0, field_width - 24.16)[1], tr(0, field_width - 23.16)[1]], color='white', lw=1)

		# Yard numbers
		yard_numbers = {
			20: '10', 30: '20', 40: '30', 50: '40', 60: '50',
			70: '40', 80: '30', 90: '20', 100: '10'
		}
		for x_pos, number in yard_numbers.items():
			xp, _ = tr(x_pos, 0)
			ax.text(xp, tr(0, 12)[1], number, color='white', fontsize=20, ha='center', va='center', rotation=180)
			ax.text(xp, tr(0, field_width - 12)[1], number, color='white', fontsize=20, ha='center', va='center')

		ax.set_xlim(0, L)
		ax.set_ylim(0, W)
		ax.set_aspect('equal')
		ax.axis('off')
		return ax

	# Default blank
	ax.set_aspect('equal')
	ax.axis('off')
	return ax


def plot_traj(dataset, index=0, save_path=None, show=False, mp4_path=None, gif_path=None, fps=10):
	seq_norm = dataset.trajs[index][..., :2].detach().cpu()  # (T, N, 2)
	# NOTE: dataset.unnormalize_batch currently outputs NFL in *yards* when units='meters'.
	# Keep plotting coordinates consistent with the pitch we draw.
	if str(dataset.name_sport).lower() == 'nfl':
		units_label = 'yards'
		seq = dataset.unnormalize_batch(seq_norm).cpu().numpy()
		pitch_type = 'american_football'
		pitch_size = (120.0, 53.3)
		# (0,0) is at the middle of the pitch
	elif str(dataset.name_sport).lower() == 'nba':
		units_label = 'm'
		seq = dataset.unnormalize_batch(seq_norm).cpu().numpy()
		pitch_type = 'nba'
		pitch_size = (28.0, 15.0)
		# (0,0) is at the middle of the court
	else:
		units_label = 'm'
		seq = dataset.unnormalize_batch(seq_norm).cpu().numpy()
		pitch_type = 'soccer'
		pitch_size = (105.0, 68.0)
  		# translate to have (0,0) at bottom-left
		seq += np.array([[pitch_size[0] / 2, pitch_size[1] / 2]])

	caption = None
	if isinstance(dataset.labels[index], dict):
		caption = dataset.labels[index].get('caption_text')

	# Use a sport-specific figure aspect so the surface looks natural.
	# NFL (120x53.3) is much wider than tall.
	if pitch_type == 'american_football':
		figsize = (9.0, 5.0)
	elif pitch_type == 'nba':
		figsize = (8.5, 5.0)
	else:
		figsize = (7.5, 6.5)
	fig, ax = plt.subplots(figsize=figsize)
	_draw_pitch(ax, pitch_type=pitch_type, pitch_size=pitch_size)
 
	# Simple animation: show agent positions over time.
	if pitch_type in ['soccer', 'american_football']:
		colors = ['#ffd166'] + [('tab:blue' if 1 <= j <= 11 else 'm') for j in range(1, seq.shape[1])]
	elif pitch_type == 'nba':
		colors = ['#ffd166'] + [('tab:blue' if 1 <= j <= 5 else 'm') for j in range(1, seq.shape[1])]
	else:
		raise NotImplementedError(f"Pitch type {pitch_type} not implemented.")

	# 0 is the ball in this codebase
	ball = seq[:, 0]
	ax.plot(ball[:, 0], ball[:, 1], color=colors[0], linewidth=2.0, label='ball')

	# players
	for j in range(1, seq.shape[1]):
		p = seq[:, j]
		ax.plot(p[:, 0], p[:, 1], linewidth=1.0, alpha=0.75, color=colors[j])
  
	scat = ax.scatter(seq[-1, :, 0], seq[-1, :, 1], s=50, c=colors, zorder=3)
	# Label each scatter point with the agent id.
	label_texts = []
	for j in range(1, seq.shape[1]):
		label_texts.append(
			ax.text(
				seq[-1, j, 0],
				seq[-1, j, 1],
				str(j),
				fontsize=8,
				ha='center',
				va='center',
				color='white',
				zorder=4,
			)
		)
	ax.set_aspect('equal', adjustable='box')
	# Pitch handles axis style, but keep a title for context.
	# ax.set_title(f"{dataset.name_sport.upper()} | {dataset.split_name} | idx={index}")

	if caption:
		wrapped = textwrap.fill(str(caption), width=95)
		# Reserve a small bottom margin for the caption (avoid excessive whitespace).
		# Soccer tends to have more unused vertical space, so keep it tighter.
		if pitch_type == 'soccer':
			fig.tight_layout(rect=[0, 0.05, 1, 1])
			caption_y = 0.02
			fontsize = 11
		else:
			fig.tight_layout(rect=[0, 0.13, 1, 1])
			caption_y = 0.03
			fontsize = 12
		fig.text(0.5, caption_y, wrapped, ha='center', va='bottom', fontsize=fontsize)
	else:
		fig.tight_layout()

	if save_path:
		fig.savefig(save_path, dpi=200, bbox_inches='tight')
		print(f"Saved: {save_path}")

	if mp4_path or gif_path:
		# remove the previous scatter and labels
		scat.remove()
		for txt in label_texts:
			txt.remove()

		scat = ax.scatter(seq[0, :, 0], seq[0, :, 1], s=50, c=colors, zorder=3)
		# Label each scatter point with the agent id.
		label_texts = []
		for j in range(1, seq.shape[1]):
			label_texts.append(
				ax.text(
					seq[0, j, 0],
					seq[0, j, 1],
					str(j),
					fontsize=8,
					ha='center',
					va='center',
					color='white',
					zorder=4,
				)
			)

		def _update(t):
			scat.set_offsets(seq[t])
			for n, txt in enumerate(label_texts):
				j = n + 1
				txt.set_position((seq[t, j, 0], seq[t, j, 1]))
			return (scat, *label_texts)

		ani = animation.FuncAnimation(fig, _update, frames=seq.shape[0], interval=1000 / fps, blit=True)

		if mp4_path:
			try:
				writer = animation.FFMpegWriter(fps=fps)
				ani.save(mp4_path, writer=writer)
				print(f"Saved: {mp4_path}")
			except Exception as e:
				raise RuntimeError(
					"Failed to write MP4. Ensure ffmpeg is installed and available on PATH. "
					f"Original error: {e}"
				)

		if gif_path:
			try:
				writer = animation.PillowWriter(fps=fps)
				ani.save(gif_path, writer=writer)
				print(f"Saved: {gif_path}")
			except Exception as e:
				raise RuntimeError(
					"Failed to write GIF. Ensure pillow is installed (pip install pillow). "
					f"Original error: {e}"
				)

	if show:
		plt.show()
	else:
		plt.close(fig)
