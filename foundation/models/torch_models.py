from typing import Dict as TypingDict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box, Dict
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_torch_modelv2 import RecurrentTorchModel
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import add_time_dimension

_WORLD_MAP_NAME = "world-map"
_WORLD_IDX_MAP_NAME = "world-idx_map"
_MASK_NAME = "action_mask"


def get_flat_obs_size(obs_space: Box) -> int:
    if isinstance(obs_space, Box):
        return int(np.prod(obs_space.shape))
    if not isinstance(obs_space, Dict):
        raise TypeError

    def rec_size(space: Dict, total: int = 0) -> int:
        for subspace in space.spaces.values():
            if isinstance(subspace, Box):
                total += int(np.prod(subspace.shape))
            elif isinstance(subspace, Dict):
                total = rec_size(subspace, total)
            else:
                raise TypeError
        return total

    return rec_size(obs_space)


def apply_logit_mask_torch(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    invalid = (1.0 - mask.float()) * -1.0e7
    return logits + invalid


class _Branch(nn.Module):
    def __init__(
        self,
        *,
        use_conv: bool,
        conv_in_channels: int,
        conv_shape: Optional[tuple],
        num_conv: int,
        num_fc: int,
        fc_dim: int,
        cell_size: int,
        non_conv_dim: int,
        embedding_vocab: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.use_conv = use_conv
        self.embedding_dim = embedding_dim
        self.conv_shape = conv_shape
        self.non_conv_dim = non_conv_dim

        conv_out_dim = 0
        if use_conv:
            assert conv_shape is not None
            self.embedding = nn.Embedding(embedding_vocab, embedding_dim)
            layers: List[nn.Module] = []
            in_channels = conv_in_channels
            layers.append(nn.Conv2d(in_channels, 16, kernel_size=3, stride=2))
            layers.append(nn.ReLU())
            current_channels = 16
            for _ in range(num_conv - 1):
                layers.append(nn.Conv2d(current_channels, 32, kernel_size=3, stride=2))
                layers.append(nn.ReLU())
                current_channels = 32
            layers.append(nn.Flatten())
            self.conv_model = nn.Sequential(*layers)
            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, conv_shape[0], conv_shape[1])
                conv_out_dim = int(self.conv_model(dummy).shape[-1])
        else:
            self.embedding = None
            self.conv_model = None

        dense_input_dim = conv_out_dim + non_conv_dim
        self.fc_layers = nn.ModuleList()
        last_dim = dense_input_dim
        for _ in range(num_fc):
            self.fc_layers.append(nn.Linear(last_dim, fc_dim))
            last_dim = fc_dim
        self.layer_norm = nn.LayerNorm(last_dim if last_dim > 0 else 1)
        self.lstm = nn.LSTM(last_dim if last_dim > 0 else 1, cell_size, batch_first=True)

    def compute_features(
        self,
        *,
        obs: TypingDict[str, torch.Tensor],
        non_conv_keys: List[str],
        idx_channels: int,
    ) -> torch.Tensor:
        tensors: List[torch.Tensor] = []
        if self.use_conv:
            conv_map = obs[_WORLD_MAP_NAME]
            conv_idx = obs[_WORLD_IDX_MAP_NAME]
            batch, time = conv_map.shape[:2]
            channels = conv_map.shape[2]
            height, width = conv_map.shape[3:5]

            conv_map_flat = conv_map.reshape(batch * time, channels, height, width)
            conv_idx_flat = conv_idx.reshape(batch * time, idx_channels, height, width)
            embedded = self.embedding(conv_idx_flat.long())
            embedded = embedded.view(
                batch * time,
                idx_channels * self.embedding_dim,
                height,
                width,
            )
            conv_input = torch.cat([conv_map_flat, embedded], dim=1)
            conv_features = self.conv_model(conv_input)
            conv_features = conv_features.view(batch, time, -1)
            tensors.append(conv_features)

        non_conv: List[torch.Tensor] = []
        for key in non_conv_keys:
            tensor = obs[key]
            non_conv.append(tensor.reshape(tensor.shape[0], tensor.shape[1], -1))
        if non_conv:
            stacked = torch.cat(non_conv, dim=-1) if len(non_conv) > 1 else non_conv[0]
            tensors.append(stacked)

        if not tensors:
            raise ValueError("No observation features available for the model.")

        features = torch.cat(tensors, dim=-1) if len(tensors) > 1 else tensors[0]
        x = features
        for layer in self.fc_layers:
            x = torch.relu(layer(x))
        x = self.layer_norm(x)
        return x

    def forward_lstm(
        self,
        x: torch.Tensor,
        state_h: torch.Tensor,
        state_c: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, max_t, _ = x.shape
        h0 = state_h.unsqueeze(0)
        c0 = state_c.unsqueeze(0)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (hn, cn) = self.lstm(packed, (h0, c0))
        padded, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=max_t
        )
        return padded, hn.squeeze(0), cn.squeeze(0)


class TorchConvLSTM(RecurrentTorchModel):
    custom_name = "torch_conv_lstm"

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        if hasattr(obs_space, "original_space"):
            obs_space = obs_space.original_space
        if not isinstance(obs_space, Dict):
            raise TypeError("Observation space should be a gym Dict.")

        custom_config = (
            model_config.get("custom_model_config")
            or model_config.get("custom_options")
            or {}
        )
        self.cell_size = int(custom_config.get("lstm_cell_size", 128))
        self.num_conv = int(custom_config.get("num_conv", 0))
        self.num_fc = int(custom_config.get("num_fc", 0))
        self.fc_dim = int(custom_config.get("fc_dim", 0))
        self.embedding_vocab = int(custom_config.get("input_emb_vocab", 0))
        self.embedding_dim = int(custom_config.get("idx_emb_dim", 0))
        self.generic_name = custom_config.get("generic_name")

        self.input_keys: List[str] = list(obs_space.spaces.keys())
        self.mask_key = _MASK_NAME

        self.use_conv = _WORLD_MAP_NAME in obs_space.spaces
        if self.use_conv:
            map_space = obs_space.spaces[_WORLD_MAP_NAME]
            idx_space = obs_space.spaces[_WORLD_IDX_MAP_NAME]
            self.conv_shape = (map_space.shape[1], map_space.shape[2])
            self.map_channels = map_space.shape[0]
            self.idx_channels = idx_space.shape[0]
            conv_in_channels = self.map_channels + self.idx_channels * self.embedding_dim
        else:
            self.conv_shape = None
            self.map_channels = 0
            self.idx_channels = 0
            conv_in_channels = 0

        excluded = {_WORLD_MAP_NAME, _WORLD_IDX_MAP_NAME, self.mask_key}
        if self.generic_name is None:
            self.non_conv_keys = [k for k in self.input_keys if k not in excluded]
        elif isinstance(self.generic_name, (tuple, list)):
            self.non_conv_keys = list(self.generic_name)
        elif isinstance(self.generic_name, str):
            self.non_conv_keys = [self.generic_name]
        else:
            raise TypeError("generic_name must be a string or sequence.")

        self.non_conv_dim = 0
        for key in self.non_conv_keys:
            space = obs_space.spaces[key]
            self.non_conv_dim += int(np.prod(space.shape))

        self.policy_branch = _Branch(
            use_conv=self.use_conv,
            conv_in_channels=conv_in_channels,
            conv_shape=self.conv_shape,
            num_conv=self.num_conv,
            num_fc=self.num_fc,
            fc_dim=self.fc_dim,
            cell_size=self.cell_size,
            non_conv_dim=self.non_conv_dim,
            embedding_vocab=self.embedding_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.value_branch = _Branch(
            use_conv=self.use_conv,
            conv_in_channels=conv_in_channels,
            conv_shape=self.conv_shape,
            num_conv=self.num_conv,
            num_fc=self.num_fc,
            fc_dim=self.fc_dim,
            cell_size=self.cell_size,
            non_conv_dim=self.non_conv_dim,
            embedding_vocab=self.embedding_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.policy_output = nn.Linear(self.cell_size, num_outputs)
        self.value_output = nn.Linear(self.cell_size, 1)
        self._value_out: Optional[torch.Tensor] = None

    def get_initial_state(self) -> List[np.ndarray]:
        return [
            np.zeros(self.cell_size, dtype=np.float32),
            np.zeros(self.cell_size, dtype=np.float32),
            np.zeros(self.cell_size, dtype=np.float32),
            np.zeros(self.cell_size, dtype=np.float32),
        ]

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        time_major_inputs = [
            add_time_dimension(obs[key], seq_lens, framework="torch")
            for key in self.input_keys
        ]
        model_out, new_state = self.forward_rnn(time_major_inputs, state, seq_lens)
        return model_out.reshape(-1, self.num_outputs), new_state

    def forward_rnn(self, inputs, state, seq_lens):
        obs = {k: v for k, v in zip(self.input_keys, inputs)}
        device = next(self.parameters()).device

        seq_lens = torch.as_tensor(seq_lens, dtype=torch.int64, device=device)

        state = [
            s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32)
            for s in state
        ]
        state = [s.to(device) for s in state]

        policy_features = self.policy_branch.compute_features(
            obs=obs, non_conv_keys=self.non_conv_keys, idx_channels=self.idx_channels
        )
        value_features = self.value_branch.compute_features(
            obs=obs, non_conv_keys=self.non_conv_keys, idx_channels=self.idx_channels
        )

        policy_out, state_h_p, state_c_p = self.policy_branch.forward_lstm(
            policy_features, state[0], state[1], seq_lens
        )
        value_out, state_h_v, state_c_v = self.value_branch.forward_lstm(
            value_features, state[2], state[3], seq_lens
        )

        logits = self.policy_output(policy_out)
        mask = obs[self.mask_key].to(device)
        logits = apply_logit_mask_torch(logits, mask)

        values = self.value_output(value_out)
        self._value_out = values

        return logits, [state_h_p, state_c_p, state_h_v, state_c_v]

    def value_function(self):
        if self._value_out is None:
            raise ValueError("Value function requested before forward pass.")
        return self._value_out.reshape(-1)


ModelCatalog.register_custom_model(TorchConvLSTM.custom_name, TorchConvLSTM)


class TorchLinear(TorchModelV2, nn.Module):
    custom_name = "torch_linear"

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        if hasattr(obs_space, "original_space"):
            obs_space = obs_space.original_space
        mask_space = obs_space.spaces[_MASK_NAME]
        self.mask_shape = mask_space.shape
        self.obs_size = get_flat_obs_size(obs_space)

        custom_config = (
            model_config.get("custom_model_config")
            or model_config.get("custom_options")
            or {}
        )
        self.fc_dim = int(custom_config.get("fc_dim", 0))
        self.num_fc = int(custom_config.get("num_fc", 0))
        self.use_fc_value = custom_config.get("fully_connected_value", False)

        self.logits_layer = nn.Linear(self.obs_size, num_outputs)

        self.value_layers = nn.ModuleList()
        last_dim = self.obs_size
        if self.use_fc_value:
            for _ in range(self.num_fc):
                self.value_layers.append(nn.Linear(last_dim, self.fc_dim))
                last_dim = self.fc_dim
        self.value_out_layer = nn.Linear(last_dim, 1)
        self._value_out: Optional[torch.Tensor] = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"]
        mask = input_dict["obs"][_MASK_NAME]
        logits = self.logits_layer(obs)
        logits = apply_logit_mask_torch(logits, mask)

        value_input = obs
        for layer in self.value_layers:
            value_input = torch.relu(layer(value_input))
        self._value_out = self.value_out_layer(value_input)
        return logits, state

    def value_function(self):
        if self._value_out is None:
            raise ValueError("Value function requested before forward call.")
        return self._value_out.reshape(-1)


ModelCatalog.register_custom_model(TorchLinear.custom_name, TorchLinear)


class RandomAction(TorchModelV2, nn.Module):
    custom_name = "random"

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        if hasattr(obs_space, "original_space"):
            obs_space = obs_space.original_space
        assert isinstance(obs_space, Dict)
        self.num_outputs = num_outputs
        self._mask_shape = obs_space.spaces[_MASK_NAME].shape
        self._value_out: Optional[torch.Tensor] = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        batch = input_dict["obs_flat"].shape[0]
        logits = input_dict["obs_flat"].new_zeros((batch, self.num_outputs))
        mask = obs[_MASK_NAME]
        logits = apply_logit_mask_torch(logits, mask)
        self._value_out = input_dict["obs_flat"].new_zeros(batch, 1)
        return logits, state

    def value_function(self):
        if self._value_out is None:
            raise ValueError("Value function requested before forward call.")
        return self._value_out.reshape(-1)


ModelCatalog.register_custom_model(RandomAction.custom_name, RandomAction)
