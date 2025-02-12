import torch
import torch.nn as nn

class RNNAgent(nn.Module):
    def __init__(self, hidden_dim=128, conv_layers=[16, 32, 64, 64]):
        super(RNNAgent, self).__init__()
        
        # CNN feature extractor for image input
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(1, conv_layers[0], kernel_size=3, stride=2, padding=1))
        for i in range(len(conv_layers) - 1):
            self.conv_layers.append(
                nn.Conv2d(conv_layers[i], conv_layers[i + 1], kernel_size=3, stride=2, padding=1)
            )
            self.conv_layers.append(nn.ReLU())
        self.conv_layers.append(nn.AdaptiveAvgPool2d((4, 4)))  # Reduce to (B, 64, 4, 4)
        self.conv = nn.Sequential(*self.conv_layers)
        
        # Fully connected layer to flatten CNN output
        self.fc_image = nn.Linear(64 * 4 * 4, hidden_dim)
        
        # RNN for temporal processing
        self.rnn = nn.GRU(input_size=hidden_dim + 7, hidden_size=hidden_dim, batch_first=True)
        
        # Fully connected output layer
        self.fc_out = nn.Linear(hidden_dim, 7)
        
    def forward(self, image, vector, hidden=None):
        # Extract features from image
        B, T, C, H, W = image.shape  # (B, T, 1, 256, 256)
        image = image.view(B * T, C, H, W)  # Merge batch & time dims for CNN processing
        img_features = self.conv(image)
        img_features = img_features.view(B * T, -1)  # Flatten
        img_features = self.fc_image(img_features)  # (B*T, hidden_dim)
        
        # Reshape back to (B, T, hidden_dim)
        img_features = img_features.view(B, T, -1)
        
        # Concatenate image features with input vector
        rnn_input = torch.cat((img_features, vector), dim=-1)  # (B, T, hidden_dim + 3)
        
        # Pass through RNN while preserving hidden state across time steps
        rnn_out, hidden = self.rnn(rnn_input, hidden)  # (B, T, hidden_dim)
        
        # Output layer
        output = self.fc_out(rnn_out[:, -1, :])  # Take the last timestep output
        
        return output, hidden
    

class PositionModel(nn.Module):
    def __init__(self, hidden_size=[128, 128]):
        super(PositionModel, self).__init__()
        
        # CNN feature extractor for image input
        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(nn.Linear(3, hidden_size[0]))
        self.linear_layers.append(nn.ReLU())
        for i in range(len(hidden_size) - 1):
            self.linear_layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            self.linear_layers.append(nn.ReLU())
        self.linear_layers.append(nn.Linear(hidden_size[-1], 7))  # Reduce to (B, 64, 4, 4)
        self.fnn = nn.Sequential(*self.linear_layers)
        
        
    def forward(self, vector):
        # Extract features from state
        output = self.fnn(vector)
        
        return output
    

class ImitationAgent:
    def __init__(self, cfg, num_envs, device, img_size=(200, 150)):
        
        self.rnn_model = RNNAgent(
            hidden_dim=cfg['model']['hidden_size'],
            conv_layers=cfg['model']['conv_layers']).to(device)
        
        self.lr = cfg['train']['lr']
        self.batch_size = cfg['train']['batch_size']
        self.optimizer = torch.optim.Adam(self.rnn_model.parameters(), lr=self.lr)
        self.history_length = cfg['train']['history_length']
        self.buffer_size = cfg['train']['buffer_size']
        self.hidden_size = cfg['model']['hidden_size']
        self.max_iter_per_episode = cfg['train']['max_iter_per_episode']
        self.num_envs = num_envs
        self.img_size = img_size
        self.device = device

        self.reset()

        
    def reset(self):
        self.buffer_images = torch.zeros(self.num_envs, self.buffer_size + self.history_length, 1, self.img_size[1], self.img_size[0]).to(self.device)
        self.buffer_vectors = torch.zeros(self.num_envs, self.buffer_size + self.history_length, 7).to(self.device)
        self.buffer_gt_output = torch.zeros(self.num_envs, self.buffer_size + self.history_length, 7).to(self.device)
        self.buffer_hidden_states = torch.zeros(self.num_envs, self.buffer_size + self.history_length, self.hidden_size).to(self.device)


    def record(self, images, vectors, gt_output, hidden_states):
        """
        Record the data into buffers while shifting existing values.
        """
        # Squeeze unnecessary dimensions
        images = images.squeeze(2)
        vectors = vectors.squeeze(1)
        hidden_states = hidden_states.squeeze(0)

        # Define buffer attributes and new data
        buffers = [
            (self.buffer_images, images),
            (self.buffer_vectors, vectors),
            (self.buffer_gt_output, gt_output),
            (self.buffer_hidden_states, hidden_states),
        ]

        # Efficiently shift and update buffers
        for buffer, new_data in buffers:
            buffer[:, self.history_length:-1].copy_(buffer[:, self.history_length+1:])  # In-place shift
            buffer[:, -1].copy_(new_data)  # In-place assignment


    def train(self):
        '''
        train the model
        '''
        
        # train batch by batch
        indices = torch.randperm(self.buffer_size)
        num_batch = self.buffer_size // self.batch_size
        num_batch = min(num_batch, self.max_iter_per_episode)
        for b in range(num_batch):
            batch_indices = indices[b*self.batch_size: (b+1)*self.batch_size] # (batch_size)
            history_batch_indices = batch_indices.reshape((-1, 1)) + torch.arange(self.history_length).reshape((1, -1)) # (batch_size, history_length)
            batch_images = self.buffer_images[:, history_batch_indices, :, :, :] # (num_envs, batch_size, history_length, 1, 256, 256)
            batch_vectors = self.buffer_vectors[:, history_batch_indices, :] # (num_envs, batch_size, history_length, 7)
            batch_gt_output = self.buffer_gt_output[:, batch_indices + self.history_length, :] # (num_envs, batch_size, 7)
            batch_hidden_states = self.buffer_hidden_states[:, batch_indices, :] # (num_envs, batch_size, hidden_size)
            batch_images = batch_images.reshape((-1, *batch_images.shape[2:])) # (num_envs * batch_size, history_length, 1, 256, 256)
            batch_vectors = batch_vectors.reshape((-1, *batch_vectors.shape[2:]))
            batch_gt_output = batch_gt_output.reshape((-1, *batch_gt_output.shape[2:]))
            batch_hidden_states = batch_hidden_states.reshape((-1, *batch_hidden_states.shape[2:])).unsqueeze(0) # (1, num_envs * batch_size, hidden_size)

            # get output
            batch_output, _ = self.rnn_model(batch_images, batch_vectors, batch_hidden_states) # (num_envs * batch_size, 7)

            self.optimizer.zero_grad()
            # batch_output[:, :3] *= 100
            # batch_gt_output[:, :3] *= 100
            loss = torch.mean((batch_output - batch_gt_output) ** 2)
            loss.backward()
            self.optimizer.step()

            print('batch', b, 'loss:', loss.item())

    def train_offline(self):
        indices = torch.randperm(self.buffer_size)
        num_batch = 1
        for b in range(num_batch):
            batch_indices = indices[b*self.batch_size: (b+1)*self.batch_size] # (batch_size)
            history_batch_indices = batch_indices.reshape((-1, 1)) + torch.arange(self.history_length).reshape((1, -1)) # (batch_size, history_length)
            batch_images = self.buffer_images[:, history_batch_indices, :, :, :] # (num_envs, batch_size, history_length, 1, 256, 256)
            batch_vectors = self.buffer_vectors[:, history_batch_indices, :] # (num_envs, batch_size, history_length, 7)
            batch_gt_output = self.buffer_gt_output[:, batch_indices + self.history_length, :] # (num_envs, batch_size, 7)
            batch_hidden_states = self.buffer_hidden_states[:, batch_indices, :] # (num_envs, batch_size, hidden_size)
            batch_images = batch_images.reshape((-1, *batch_images.shape[2:])) # (num_envs * batch_size, history_length, 1, 256, 256)
            batch_vectors = batch_vectors.reshape((-1, *batch_vectors.shape[2:]))
            batch_gt_output = batch_gt_output.reshape((-1, *batch_gt_output.shape[2:]))
            batch_hidden_states = batch_hidden_states.reshape((-1, *batch_hidden_states.shape[2:])).unsqueeze(0) # (1, num_envs * batch_size, hidden_size)

            # get output
            batch_output, _ = self.rnn_model(batch_images, batch_vectors, batch_hidden_states) # (num_envs * batch_size, 7)

            self.optimizer.zero_grad()
            # batch_output[:, :3] *= 100
            # batch_gt_output[:, :3] *= 100
            loss = torch.mean((batch_output - batch_gt_output) ** 2)
            loss.backward()
            self.optimizer.step()

            print('batch', b, 'loss:', loss.item())


    def predict(self, images, vectors, hidden_states):
        '''
        predict the output
        images: (num_envs, T, 1, 256, 256)
        vectors: (num_envs, T, 7)
        hidden_states: (num_envs, hidden_size)
        '''
        return self.rnn_model(images, vectors, hidden_states)
        


class PositionAgent:
    def __init__(self, cfg, num_envs, device):
        
        self.model = PositionModel(
            hidden_size=cfg['model']['hidden_size']).to(device)
        
        self.lr = cfg['train']['lr']
        self.batch_size = cfg['train']['batch_size']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.buffer_size = cfg['train']['buffer_size']
        self.hidden_size = cfg['model']['hidden_size']
        self.max_iter_per_episode = cfg['train']['max_iter_per_episode']
        self.num_envs = num_envs
        self.device = device
        
        self.reset()

        
    def reset(self):
        self.buffer_vectors = torch.zeros(self.num_envs, self.buffer_size, 3).to(self.device)
        self.buffer_gt_output = torch.zeros(self.num_envs, self.buffer_size, 7).to(self.device)


    def record(self, vectors, gt_output):
        """
        Record the data into buffers while shifting existing values.
        """
        # Define buffer attributes and new data
        buffers = [
            (self.buffer_vectors, vectors),
            (self.buffer_gt_output, gt_output),
        ]

        # Efficiently shift and update buffers
        for buffer, new_data in buffers:
            buffer[:, :-1].copy_(buffer[:, 1:])  # In-place shift
            buffer[:, -1].copy_(new_data)  # In-place assignment


    def train(self):
        '''
        train the model
        '''
        
        # train batch by batch
        indices = torch.randperm(self.buffer_size)
        num_batch = self.buffer_size // self.batch_size
        num_batch = min(num_batch, self.max_iter_per_episode)
        for b in range(num_batch):
            batch_indices = indices[b*self.batch_size: (b+1)*self.batch_size] # (batch_size)
            batch_vectors = self.buffer_vectors[:, batch_indices, :] # (num_envs, batch_size, history_length, 7)
            batch_gt_output = self.buffer_gt_output[:, batch_indices, :] # (num_envs, batch_size, 7)
            batch_vectors = batch_vectors.reshape((-1, *batch_vectors.shape[2:]))
            batch_gt_output = batch_gt_output.reshape((-1, *batch_gt_output.shape[2:]))

            # get output
            batch_output = self.model(batch_vectors) # (num_envs * batch_size, 7)

            self.optimizer.zero_grad()
            batch_output[:, :3] 
            batch_gt_output[:, :3] *= 1000
            loss = torch.mean((batch_output - batch_gt_output) ** 2)
            loss.backward()
            self.optimizer.step()

            print('batch', b, 'loss:', loss.item())

    def train_offline(self):
        indices = torch.randperm(self.buffer_size)
        num_batch = 1
        for b in range(num_batch):
            batch_indices = indices[b*self.batch_size: (b+1)*self.batch_size] # (batch_size)
            batch_vectors = self.buffer_vectors[:, batch_indices, :] # (num_envs, batch_size, history_length, 7)
            batch_gt_output = self.buffer_gt_output[:, batch_indices, :] # (num_envs, batch_size, 7)
            batch_vectors = batch_vectors.reshape((-1, *batch_vectors.shape[2:]))
            batch_gt_output = batch_gt_output.reshape((-1, *batch_gt_output.shape[2:]))

            # get output
            batch_output = self.model(batch_vectors) # (num_envs * batch_size, 7)

            self.optimizer.zero_grad()
            batch_output[:, :3]
            batch_gt_output[:, :3] *= 1000
            loss = torch.mean((batch_output - batch_gt_output) ** 2)
            loss.backward()
            self.optimizer.step()

            print('batch', b, 'loss:', loss.item())


    def predict(self, vectors):
        '''
        predict the output
        images: (num_envs, T, 1, 256, 256)
        vectors: (num_envs, T, 7)
        hidden_states: (num_envs, hidden_size)
        '''
        return self.model(vectors)