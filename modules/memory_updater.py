from torch import nn
import torch


class MemoryUpdater(nn.Module):
    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        pass


class AttentionCell(nn.Module):
    """
    自定義的 Attention Cell，介面模仿 RNNCell，但內部使用 Transformer Block。
    邏輯：Memory (Query) 去關注 Message (Key/Value) 來進行更新。
    """
    def __init__(self, input_dim, hidden_dim, num_heads=4, dropout=0.1):
        super(AttentionCell, self).__init__()
        
        # 1. 如果 Message (input) 維度跟 Memory (hidden) 不同，先做投影
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim) 
            if input_dim != hidden_dim 
            else nn.Identity()
        )

        # 2. Multi-Head Attention
        # 為了避免維度錯誤，若 hidden_dim 不能被 num_heads 整除，強制設為 1
        if hidden_dim % num_heads != 0:
            num_heads = 1
            
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )

        # 3. Transformer 標準結構: Add & Norm + FFN
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden):
        # inputs (message): [batch_size, input_dim]
        # hidden (memory):  [batch_size, hidden_dim]

        # 投影輸入
        inputs = self.input_proj(inputs)  # -> [batch, hidden_dim]

        # 調整形狀以符合 MultiheadAttention: [batch, seq_len=1, dim]
        query = hidden.unsqueeze(1)  # Memory 作為 Query
        key = inputs.unsqueeze(1)    # Message 作為 Key
        value = inputs.unsqueeze(1)  # Message 作為 Value

        # Attention Block
        # attn_output: [batch, 1, hidden_dim]
        attn_output, _ = self.multihead_attn(query, key, value)
        
        # Residual + Norm 1
        # squeeze(1) 把 seq_len 維度拿掉，變回 [batch, hidden_dim]
        x = hidden + self.dropout(attn_output.squeeze(1))
        x = self.norm1(x)

        # FFN Block
        ffn_output = self.ffn(x)
        
        # Residual + Norm 2
        new_memory = self.norm2(x + self.dropout(ffn_output))

        return new_memory

class SequenceMemoryUpdater(MemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(SequenceMemoryUpdater, self).__init__()
        self.memory = memory
        self.layer_norm = torch.nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        self.device = device

    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return

        assert (
            (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item()
        ), ("Trying to " "update memory to time in the past")

        memory = self.memory.get_memory(unique_node_ids)
        self.memory.last_update[unique_node_ids] = timestamps

        updated_memory = self.memory_updater(unique_messages, memory)

        self.memory.set_memory(unique_node_ids, updated_memory)

    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

        assert (
            (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item()
        ), ("Trying to " "update memory to time in the past")

        updated_memory = self.memory.memory.data.clone()
        updated_memory[unique_node_ids] = self.memory_updater(
            unique_messages, updated_memory[unique_node_ids]
        )

        updated_last_update = self.memory.last_update.data.clone()
        updated_last_update[unique_node_ids] = timestamps

        return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(GRUMemoryUpdater, self).__init__(
            memory, message_dimension, memory_dimension, device
        )

        self.memory_updater = nn.GRUCell(
            input_size=message_dimension, hidden_size=memory_dimension
        )


class RNNMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(RNNMemoryUpdater, self).__init__(
            memory, message_dimension, memory_dimension, device
        )

        self.memory_updater = nn.RNNCell(
            input_size=message_dimension, hidden_size=memory_dimension
        )

class AttentionMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(AttentionMemoryUpdater, self).__init__(
            memory, message_dimension, memory_dimension, device
        )
        
        self.memory_updater = AttentionCell(
            input_dim=message_dimension,
            hidden_dim=memory_dimension,
            num_heads=4
        ).to(device)

def get_memory_updater(
    module_type, memory, message_dimension, memory_dimension, device
):
    if module_type == "gru":
        return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
    elif module_type == "rnn":
        return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
    elif module_type == "attention":  # 新增這個選項
        return AttentionMemoryUpdater(memory, message_dimension, memory_dimension, device)
    else:
        raise ValueError("Updater type {} not implemented".format(module_type))