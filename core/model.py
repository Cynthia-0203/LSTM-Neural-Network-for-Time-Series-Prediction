import os
import numpy as np
import datetime as dt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Model(nn.Module):
	"""A class for an building and inferencing an lstm model"""

	def __init__(self):
		super(Model, self).__init__()
		self.layers = nn.ModuleList()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.to(self.device)
	def forward(self, x):
        # # 定义前向传播过程
		# for layer in self.layers:
		# 	x, _ = layer(x)
		# return x
		
		for layer in self.layers:
			if isinstance(layer, nn.LSTM):
				x, _ = layer(x)
			elif isinstance(layer, nn.Sequential):
				x = x[:, -1, :]  # 提取最后一个时间步
			else:
				x = layer(x)
		return x


	# def load_model(self, filepath):
	# 	print('[Model] Loading model from file %s' % filepath)
	# 	self.load_state_dict(torch.load(filepath, map_location=self.device))
	# 	self.eval()

	def build_model(self, configs):
		
		for layer in configs['model']['layers']:
			# LSTM 层
			if layer['type'] == 'lstm':
				neurons = layer.get('neurons')
				input_dim = layer.get('input_dim')
				input_timesteps = layer.get('input_timesteps')
				return_seq = layer.get('return_seq')
				if input_dim is None:
					input_dim = neurons
				lstm_layer = nn.LSTM(input_dim, neurons, batch_first=True)
				
				if not return_seq:
					self.layers.append(nn.Sequential(nn.Flatten()))  # 展平输出
				else:
					self.layers.append(lstm_layer)
			# Dropout 层
			elif layer['type'] == 'dropout':
				dropout_rate = layer.get('rate')
				self.layers.append(nn.Dropout(dropout_rate))
			# Dense (全连接) 层
			elif layer['type'] == 'dense':
				neurons = layer.get('neurons')
				activation = layer.get('activation')
				self.layers.append(nn.Linear(100, neurons))
				if activation == 'relu':
					self.layers.append(nn.ReLU())
				elif activation == 'sigmoid':
					self.layers.append(nn.Sigmoid())
				elif activation == 'tanh':
					self.layers.append(nn.Tanh())
				elif activation == 'linear':
					self.layers.append(nn.Identity())
				
		self.loss_fn = self.get_loss(configs['model']['loss'])
		self.optimizer = self.get_optimizer(configs['model']['optimizer'])
		self.save_dir = configs['model']['save_dir']
		self.to(self.device)
		print('[Model] Model Compiled')

	def get_loss(self, loss_name):
		if loss_name == 'mse':
			return nn.MSELoss()
		elif loss_name == 'mae':
			return nn.L1Loss()
		elif loss_name == 'huber':
			return nn.HuberLoss()
	def get_optimizer(self, optimizer_name):
        # 根据配置中的优化器名称返回相应的优化器
		if optimizer_name == 'adam':
			return optim.Adam(self.parameters())
		elif optimizer_name == 'sgd':
			return optim.SGD(self.parameters(), lr=0.01)  # 可以调整学习率
		elif optimizer_name == 'rmsprop':
			return optim.RMSprop(self.parameters())


	def train_model(self, x, y, epochs, batch_size, save_dir):
		
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		self.train()
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		
		x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
		y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
		
		dataset = TensorDataset(x_tensor, y_tensor)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

		for epoch in range(epochs):
			epoch_loss = 0.0

			for batch_x, batch_y in dataloader:
				# 前向传播
				self.optimizer.zero_grad()
				
				predictions = self(batch_x)

				# 计算损失
				loss = self.loss_fn(predictions, batch_y)

				# 反向传播和参数更新
				loss.backward()
				self.optimizer.step()

				epoch_loss += loss.item()

			# avg_loss = epoch_loss / len(dataloader)
			print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')
		
		torch.save(self, save_fname)
		print('[Model] Training Completed. Model saved as %s' % save_fname)
		

	def predict_point_by_point(self, data):
		print('[Model] Predicting Point-by-Point...')
		# input_tensor = torch.from_numpy(data).float().to(self.device)
		# predicted = self(input_tensor)
		# predicted = predicted.detach().cpu().numpy().reshape(-1)
		# return predicted
		self.eval()
		with torch.no_grad():
			x_tensor = torch.FloatTensor(data).to(self.device)
			predictions = self(x_tensor)
		return predictions.cpu().numpy()

	def predict_sequences_multiple(self, data, window_size, prediction_len):
		# Predict sequence of 50 steps before shifting prediction run forward by 50 steps
		print('[Model] Predicting Sequences Multiple...')
		self.eval()
		prediction_seqs = []
		with torch.no_grad():
			for i in range(int(len(data)/prediction_len)):
				# curr_frame = data[i*prediction_len]
				curr_frame = torch.FloatTensor(data[i * prediction_len]).to(self.device)
				predicted = []
				# for j in range(prediction_len):
					# input_tensor = torch.from_numpy(curr_frame[np.newaxis, :, :]).float().to(self.device)
					# output = self(input_tensor)
					# predicted.append(output[0, 0].item())
					# curr_frame = curr_frame[1:]
					# curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
				for _ in range(prediction_len):
					pred = self(curr_frame.unsqueeze(0))[0].item()
					predicted.append(pred)
					curr_frame = torch.roll(curr_frame, -1, dims=0)
					curr_frame[-1] = pred
				prediction_seqs.append(predicted)
		return prediction_seqs

	def predict_sequence_full(self, data, window_size):
		#Shift the window by 1 new prediction each time, re-run predictions on new window
		print('[Model] Predicting Sequences Full...')
		self.eval()
		predicted = []
		with torch.no_grad():
			curr_frame = torch.FloatTensor(data[0]).to(self.device)
			for i in range(len(data)):
				# input_tensor = torch.from_numpy(curr_frame[np.newaxis, :, :]).float().to(self.device)
				# output = self(input_tensor)
				# predicted.append(output[0, 0].item())
				# curr_frame = curr_frame[1:]
				# curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
				pred = self(curr_frame.unsqueeze(0))[0].item()
				predicted.append(pred)
				
				# 更新输入帧
				curr_frame = torch.roll(curr_frame, -1, dims=0)
				curr_frame[-1] = pred
		return predicted
