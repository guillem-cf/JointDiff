import numpy as np
import copy
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pickle as pkl
import json

from utils import poss_processing, poss_conditioning, plot_traj
from metrics import compute_SADE_SFDE, compute_ADE_FDE, compute_ACC


class AllSports(Dataset):
    """Dataloader for the Sports Trajectory datasets"""
    def __init__(
        self, name_sport, split_name, obs_len=10, poss_trsh=1.5, guidance='guidposs', captions=False, llm='T5-base', fut_obs=None, device='cuda', text_embeddings=False, data_root=None,
    ):
        """
        Args:
            name_sport (str): Sport identifier. One of 'nba', 'nfl', 'bundes'.
            split_name (str): Dataset split. One of 'train', 'test'.
            obs_len (int): Number of observed (input) frames.
            poss_trsh (float): Distance threshold (in meters) for ball-possession assignment.
            guidance (str): Possession guidance mode. Default 'guidposs'.
            captions (bool): Whether to load text captions (NFL & Bundesliga only).
            llm (str): HuggingFace model id for text embeddings (e.g. 'T5-base').
            fut_obs (int or None): If 1, last frame is visible (imputation task). None for forecasting.
            device (str): Device for text-embedding computation ('cuda' or 'cpu').
            text_embeddings (bool): If True, pre-compute T5 embeddings for captions.
            data_root (str or None): Path to the 'all_sports' data folder. Auto-detected if None.
        """
        super(AllSports, self).__init__()

        assert name_sport in ["nba", "nfl", "bundes"]
        assert split_name in ["train", "test"]
        assert fut_obs in [None, 1]  # "None" for forecasting, "1" for imputation

        self.split_name = split_name
        self.obs_len = obs_len
        self.fut_obs = fut_obs
        self.num_classes = 0
        self.use_team_GT = []
        self.name_sport = name_sport

        # Root folder that contains the "all_sports" data structure.
        if data_root is None:
            data_root = Path(__file__).resolve().parent / 'all_sports'
        else:
            data_root = Path(data_root)

        self.data_root = data_root

        self.captions_root = None

        if name_sport in ['nba']:  # NBA dataset (basketball)
            assert captions is False, "Captions not supported for NBA dataset"
            if split_name == 'train':
                data_path = self.data_root / 'nba_led' / 'nba_train.npy'
            else:
                data_path = self.data_root / 'nba_led' / 'nba_test.npy'
            self.process_nba(str(data_path), split_name)
        elif name_sport == 'nfl':  # NFL dataset  (American football)
            if split_name == 'train':
                data_path = self.data_root / 'nfl' / 'processed' / 'train_clean.p'
                captions_path = self.data_root / 'nfl' / 'processed' / 'captions' / 'one_json_refined' / 'train_captions_refined.json'
            else:
                data_path = self.data_root / 'nfl' / 'processed' / 'test_clean.p'
                captions_path = self.data_root / 'nfl' / 'processed' / 'captions' / 'one_json_refined' / 'test_captions_refined.json'
            self.captions_root = captions_path
            self.process_nfl(str(data_path))
        elif name_sport == 'bundes':  # Bundesliga dataset  (football / soccer)
            if split_name == 'train':
                data_path = self.data_root / 'bundesliga' / 'processed' / 'train.npy'
                captions_path = self.data_root / 'bundesliga' / 'processed' / 'captions' / 'one_json_refined' / 'train_captions_refined.json'
            else:
                data_path = self.data_root / 'bundesliga' / 'processed' / 'test.npy'
                captions_path = self.data_root / 'bundesliga' / 'processed' / 'captions' / 'one_json_refined' / 'test_captions_refined.json'
            self.captions_root = captions_path
            self.process_bundes(str(data_path), split_name)
        else:
            raise NotImplementedError(f'Sport {name_sport} not implemented')
            
        self.pred_players = list(range(1, self.trajs.shape[2]))
    
        self.generate_labels(poss_trsh, guidance, name_sport)
        
        if captions:
            if self.captions_root is None:
                raise ValueError(f"Captions requested but not available for sport: {name_sport}")
            self.load_captions_text(self.captions_root)
            if text_embeddings:
                # assert guidance == 'guidtextemb', "Captions only supported with 'guidtextemb' guidance"
                self.generate_text_embeddings(self.captions_root, llm=llm, device=device)

    def load_captions_text(self, captions_path):
        """Attach caption text (captions_ref) to labels without computing embeddings."""
        captions_path = Path(captions_path)
        if not captions_path.exists():
            raise FileNotFoundError(f"Captions file not found at {captions_path}")

        with open(captions_path, 'r') as f:
            captions_json = json.load(f)

        assert len(captions_json) == len(self.labels), "Captions length does not match trajectories length"

        for i in range(len(self.labels)):
            caption_text = captions_json[i]['captions_ref'].replace('player', 'Player')
            self.labels[i]['caption_text'] = caption_text

        
    def generate_text_embeddings(self, captions, llm='T5-base', device='cuda'):
        """Generate text embeddings for the dataset"""
        try:
            from transformers import T5Tokenizer, T5EncoderModel
        except ImportError:
            raise ImportError(
                "The 'transformers' library is required for text embeddings. "
                "Install it with: pip install transformers"
            )
        captions = Path(captions)
        if not captions.exists():
            raise FileNotFoundError(f"Captions file not found at {captions}")
        # Upload jsons
        with open(captions, 'r') as f:
            captions_json = json.load(f)
        
        assert len(captions_json) == len(self.labels), "Captions length does not match trajectories length"
        
        captions_ref = []
        captions = []
        for i in range(len(captions_json)):
            captions_ref.append(captions_json[i]['captions_ref'].replace('player', 'Player'))
            # captions.append(captions_json[i]['caption'])

        # Load T5 tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(llm)
        self.t5_encoder = T5EncoderModel.from_pretrained(llm).to(device)
        self.t5_encoder.eval()
        
        tokens = self.tokenizer.batch_encode_plus(
            captions_ref, padding='longest')
        
        embeddings = []
        batch_size = 100
        for i in range(0, len(tokens['input_ids']), batch_size):
            input_ids = torch.tensor(tokens['input_ids'][i:i + batch_size]).to(device)
            attention_mask = torch.tensor(tokens['attention_mask'][i:i + batch_size]).to(device)
            with torch.no_grad():
                outputs = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings.extend(outputs.last_hidden_state.cpu())
        
        # Clear GPU memory
        # torch.cuda.empty_cache()
        # we need the model in eval model to compute bert score-like metrics
        
        # Insert embeddings into labels
        for i in range(len(self.labels)):
            self.labels[i]['caption_text'] = captions_ref[i]
            self.labels[i]['caption_tok'] = torch.tensor(tokens['input_ids'][i])
            self.labels[i]['caption_emb'] = embeddings[i]

    def generate_labels(self, poss_trsh, guidance, sport_name):
        if sport_name == 'nfl':
            conv_units = 0.9144  # Yards to meters
        else:
            conv_units = 1.0
        self.conv_units = conv_units
        # Generate labels for possessor
        self.labels = []
        for i in range(len(self.trajs)):
            self.labels.append({})
        self.labels = poss_processing(self.trajs * self.r_fact * conv_units, self.labels, poss_threshold=poss_trsh)
        if guidance is not None:
            self.labels = poss_conditioning(self.trajs, self.labels, name=guidance)
            
        assert len(self.labels) == len(self.trajs), "Labels length does not match trajectories length"

    def process_nba(self, data_root, split_name,):
        """Process the NBA dataset"""
        self.trajs = np.load(data_root) #(N,30,11,2)
        self.trajs /= (94/28) # from feet to meters 

        # Same splits as LED: https://arxiv.org/abs/2303.10895
        if split_name == 'train':
            self.trajs = self.trajs[:32500]
        else:
            self.trajs = self.trajs[:12500]
        
        # Normalize the trajectories
        court_dims = np.array([28, 15])  # Basketball court is 28x15 meters
        self.traj_mean = court_dims / 2
        self.r_fact = 5
        self.trajs = (self.trajs - self.traj_mean) / self.r_fact

        # Put ball first
        ball = self.trajs[:, :, -1]
        self.trajs = np.concatenate([ball[:, :, None, :], self.trajs[:, :, :-1]], axis=2)
        
        # From nuympy to torch
        self.trajs = torch.from_numpy(self.trajs).type(torch.float)
        
        # Add attributes
        self.trajs_attr = torch.zeros_like(self.trajs[:, :, :, :1]).repeat(1, 1, 1, 3)
        self.trajs_attr[:, :, 0, 0] = 0  # Ball
        self.trajs_attr[:, :, 1:6, 0] = 1  # Team 1 players
        self.trajs_attr[:, :, 6:11, 0] = 1  # Team 2 players
        self.trajs_attr[..., -2] = 1  # No NaN
        self.trajs_attr[:, :self.obs_len, :, -1] = 0  # Input
        self.trajs_attr[:, self.obs_len:, :, -1] = 1  # Output
        if self.fut_obs is not None:
            self.trajs_attr[:, -self.fut_obs:, :, -1] = 0  # Future observation visible
        
    def process_nfl(self, data_root):
        """Process the NFL dataset"""
        with open(data_root, 'rb') as f:
            self.trajs = pkl.load(f)
        
        # Normalize the trajectories
        court_dims = np.array([120, 53.3])  # Football field is 120x53.3 yards
        self.r_fact = 60
        self.traj_mean = court_dims / 2
        self.trajs = (self.trajs - self.traj_mean) / self.r_fact  # bound to [-1, 1]
        
        # From numpy to torch
        self.trajs = torch.from_numpy(self.trajs).type(torch.float)
        
        # Add attributes
        self.trajs_attr = torch.zeros_like(self.trajs[:, :, :, :1]).repeat(1, 1, 1, 3)
        self.trajs_attr[:, :, 0, 0] = 0  # Ball
        self.trajs_attr[:, :, 1:12, 0] = 1  # Team 1 players
        self.trajs_attr[:, :, 12:, 0] = 1  # Team 2 players
        self.trajs_attr[..., -2] = 1  # No NaN
        self.trajs_attr[:, :self.obs_len, :, -1] = 0  # Input
        self.trajs_attr[:, self.obs_len:, :, -1] = 1  # Output
        if self.fut_obs is not None:
            self.trajs_attr[:, -self.fut_obs:, :, -1] = 0  # Future observation visible
        
    def process_bundes(self, data_root, split_name, augmentation=True):
        """Process the Bundesliga dataset"""
        self.trajs = np.load(data_root)
        
        # Normalize the trajectories
        # court_dims = np.array([105, 68])  # Football field is 105x68 meters
        self.r_fact = 52.5
        self.trajs /= self.r_fact  # bound to [-1, 1]
        self.traj_mean = np.array([0, 0])  # Center of the field
        
        # From numpy to torch
        self.trajs = torch.from_numpy(self.trajs).type(torch.float)
        
        if split_name == 'train' and augmentation:
            # assert captions is False, "Captions not supported for Bundesliga dataset"
            # Data augmentation
            data = copy.deepcopy(self.trajs)
            data[..., 0] = -data[..., 0]  # Flip x-coordinates
            data[..., 1] = -data[..., 1]  # Flip y-coordinates
            self.trajs = torch.cat([self.trajs, data], dim=0)
        
        # Add attributes
        self.trajs_attr = torch.zeros_like(self.trajs[:, :, :, :1]).repeat(1, 1, 1, 3)
        self.trajs_attr[:, :, 0, 0] = 0  # Ball
        self.trajs_attr[:, :, 1:12, 0] = 1  # Team 1 players
        self.trajs_attr[:, :, 12:, 0] = 1  # Team 2 players
        self.trajs_attr[..., -2] = 1  # No NaN
        self.trajs_attr[:, :self.obs_len, :, -1] = 0  # Input
        self.trajs_attr[:, self.obs_len:, :, -1] = 1  # Output
        if self.fut_obs is not None:
            self.trajs_attr[:, -self.fut_obs:, :, -1] = 0  # Future observation visible
            
    def unnormalize_batch(self, xy_norm):
        """Unnormalize trajectories back to metric coordinates.

        Accepts either:
        - torch.Tensor with shape (..., 2) or (..., D>=2)
        - numpy.ndarray with shape (..., 2) or (..., D>=2)

        Returns the same type as input. If D>2, only the first two channels
        (x,y) are unnormalized and the rest are preserved.
        """

        if torch.is_tensor(xy_norm):
            xy = xy_norm[..., :2] * float(self.r_fact)
            mean = torch.as_tensor(self.traj_mean, dtype=xy_norm.dtype, device=xy_norm.device)
            xy = xy + mean

            if xy_norm.shape[-1] > 2:
                out = xy_norm.clone()
                out[..., :2] = xy
                return out
            return xy

        raise TypeError(f"Unsupported trajs type: {type(xy_norm)}")

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, index):
        agents_traj = self.trajs[index]
        agents_attr = self.trajs_attr[index]
        agents = torch.cat([agents_traj, agents_attr], dim=-1)  # Concatenate trajectory and attributes
        
        agents_out = agents
        agents_in = agents_out[:0]
        
        labels = self.labels[index]

        return agents_in, agents_out, labels


if __name__ == "__main__":
    
    # data_root = Path(__file__).resolve().parent / 'all_sports'
    save_path = Path(__file__).resolve().parent / 'examples'
    # Create output directory if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)
    
    idx = {'nba': 7, 'bundes': 0, 'nfl': 3}  # Example indices for visualization
    
    
    for name_sport in ['nba', 'nfl', 'bundes']:
        print(f"\n################ {name_sport.upper()} ################")
        captions = True if name_sport in ['nfl', 'bundes'] else False
        dataset = AllSports(
            name_sport=name_sport,
            split_name='test',
            obs_len=10,
            fut_obs=None,   # Set fut_obs=1 for imputation task (last frame visible), None for forecasting
            captions=captions,
            text_embeddings=False,
        )

        plot_traj(dataset, index=idx[name_sport],
                save_path=str(save_path / f"output_{name_sport}.png"),
                mp4_path=str(save_path / f"output_{name_sport}.mp4"),
                # gif_path=str(save_path / f"output_{name_sport}.gif"),
                fps=5,
                show=False)
    
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

        metrics = {}
        for n, data in enumerate(dataloader):
            agents_in, agents_out, labels = data
            
            if n == 0:
                print("Agents in shape:", agents_in.shape)  # dummy tensor with shape [B, 0, N, D]
                print("Agents out shape:", agents_out.shape)
                print("Labels keys:", labels.keys())
            
            gt_xy = agents_out[..., :2]   # [B, T, N, 2]
            gt_poss = labels['poss_idx']  # [B, T]
            mask = agents_out[..., -1]    # [B, T, N] where 0 indicates input and 1 indicates output
            
            # Model prediction
            # pred_xy, pred_poss = model(gt_xy, mask, labels)
            pred_xy = gt_xy.clone().unsqueeze(1).repeat(1, 20, 1, 1, 1)  # [B, modes, T, N, 2] where we create 20 identical modes for testing
            pred_xy = pred_xy + torch.randn_like(pred_xy) * 0.1  # Add noise for testing
            pred_poss = gt_poss.clone().unsqueeze(1).repeat(1, 20, 1)  # [B, modes, T] where we create 20 identical modes for testing
            pred_poss = torch.randint(0, gt_poss.max()+1, pred_poss.shape)
            
            # Unnormalize trajectories back to metric coordinates
            gt_xy = dataset.unnormalize_batch(gt_xy)  # [B, T, N, 2]
            pred_xy = dataset.unnormalize_batch(pred_xy)  # [B, modes, T, N, 2]
            norm = torch.norm(pred_xy - gt_xy.unsqueeze(1), dim=-1)  # [B, modes, T, N]
            
            # Compute metrics
            scene_metrics = compute_SADE_SFDE(norm, mask)
            individual_metrics = compute_ADE_FDE(norm, mask)
            possessor_metrics = compute_ACC(pred_poss, gt_poss, mask[..., 0])
            batch_metrics = {**scene_metrics, **individual_metrics, **possessor_metrics}
            
            if metrics == {}:
                metrics = {key: [] for key in batch_metrics.keys()}
            for key in metrics.keys():
                metrics[key].append(batch_metrics[key])

        # Average metrics over batches
        for key in metrics.keys():
            metrics[key] = torch.cat(metrics[key], dim=0).mean().item()
        
        print(f"\nMetrics on {name_sport.upper()} test set:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")