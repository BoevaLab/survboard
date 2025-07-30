import copy
import os
from random import shuffle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter


class Attention(nn.Module):
	def __init__(self, m_length, modalities, device=None):
		"""Attention-based multimodal fusion Part
		Parameters
		----------
		m_length: int
			Weight vector length, corresponding to the modality representation
			length.

		modalities: list
			The list of used modality.

		"""
		super(Attention, self).__init__()
		self.m_length = m_length
		self.data_modalities = modalities
		self.device = device
		# contrast a pipeline for different modality weight matrix
		self.pipeline = {}
		for modality in self.data_modalities:
			self.pipeline[modality] = nn.Linear(self.m_length, self.m_length, bias=False).to(self.device)

	def _scale_for_missing_modalities(self, x, out):
		batch_dim = x.shape[1]
		for i in range(batch_dim):
			patient = x[:, i, :]
			zero_dims = 0
			for modality in patient:
				if modality.sum().data == 0:
					zero_dims += 1

			if zero_dims > 0:
				scaler = zero_dims + 1
				out[i, :] = scaler * out[i, :]

		return out

	def forward(self, multimodal_input):
		"""
		multimodal_input: dictionary
			A dictionary of used modality data, like:
			{'clinical':tensor(sample_size, m_length), 'mRNA':tensor(,),}
		"""
		attention_weight = tuple()
		multimodal_features = tuple()
		for modality in self.data_modalities:
			attention_weight += (torch.tanh(self.pipeline[modality](multimodal_input[modality])),)
			multimodal_features += (multimodal_input[modality],)

		# Across feature
		attention_matrix = F.softmax(torch.stack(attention_weight), dim=0)
		fused_vec = torch.sum(torch.stack(multimodal_features) * attention_matrix, dim=0)

		fused_vec = self._scale_for_missing_modalities(torch.stack(multimodal_features), fused_vec)
		return fused_vec



class ClinicalEmbeddingNet(nn.Module):
	def __init__(self, m_length, clinical_length):

		"""Clinical Embedding Network
		Parameters
		----------
		n_continuous: int (Default:1)
			The number of continuous variables.

		m_length: int
			Representation length.

		embedding_size: list (Default:[(33, 17), (2, 1), (6, 3), (145, 50)])
			Embedding size = (original categorical dimension, embedded dimension)

		"""
		super(ClinicalEmbeddingNet, self).__init__()
		# Embedding Layer

		# Linear Layer
		self.hidden1 = nn.Linear(clinical_length, m_length)
		# batch normalization
		self.bn1 = nn.BatchNorm1d(clinical_length)

	def forward(self, x_continuous):

		x2 = self.bn1(x_continuous)
		x = self.hidden1(x2)

		return x




class CnvNet(nn.Module):
	"""Extract representations from CNV modality"""
	def __init__(self, cnv_length, m_length):
		"""CNV Nerual Network with fully connected layers
		Parameters
		----------
		cnv_length: int
			The input dimension of miRNA modality.

		m_length: int
			Output dimension.

		"""
		super(CnvNet, self).__init__()

		# Linear Layers
		self.cnv_hidden1 = nn.Linear(cnv_length, 1300)
		self.cnv_hidden2 = nn.Linear(1300, 700)
		self.cnv_hidden3 = nn.Linear(700, 300)
		self.cnv_hidden4 = nn.Linear(300, m_length)

		# Batch Normalization Layers
		self.bn1 = nn.BatchNorm1d(1300)
		self.bn2 = nn.BatchNorm1d(700)
		self.bn3 = nn.BatchNorm1d(300)
		self.bn4 = nn.BatchNorm1d(m_length)

		# Dropout Layer
		self.dropout_layer1 = nn.Dropout(p=0.3)
		self.dropout_layer2 = nn.Dropout(p=0.3)
		self.dropout_layer3 = nn.Dropout(p=0.4)


	def forward(self, cnv_input):
		cnv = torch.relu(self.bn1(self.cnv_hidden1(cnv_input)))
		cnv = self.dropout_layer1(cnv)
		cnv = torch.relu(self.bn2(self.cnv_hidden2(cnv)))
		cnv = torch.relu(self.bn3(self.cnv_hidden3(cnv)))
		cnv = self.dropout_layer2(cnv)
		cnv = torch.relu(self.bn4(self.cnv_hidden4(cnv)))

		cnv = self.dropout_layer3(cnv)

		return cnv



class FixedAttention(nn.Module):
	def __init__(self, m_length, modalities, device=None):
		"""Fixed Attention-based multimodal fusion Part
		Parameters
		----------
		m_length: int
			Weight vector length, corresponding to the modality representation
			length.

		modalities: list
			The list of used modality.

		"""
		super(FixedAttention, self).__init__()
		self.m_length = m_length
		self.data_modalities = modalities
		self.device = device
		# contrast a pipeline for different modality weight matrix

	def forward(self, multimodal_input):
		"""
		multimodal_input: dictionary
			A dictionary of used modality data, like:
			{'clinical':tensor(sample_size, m_length), 'mRNA':tensor(,),}
		"""
		attention_weight = tuple()
		multimodal_features = tuple()
		for modality in self.data_modalities:
			attention_weight += (torch.ones(multimodal_input[modality].shape[0], self.m_length).to(self.device),)
			multimodal_features += (multimodal_input[modality],)

		# Across feature
		attention_matrix = F.softmax(torch.stack(attention_weight), dim=0)
		fused_vec = torch.sum(torch.stack(multimodal_features) * attention_matrix, dim=0)

		return fused_vec





class MirnaNet(nn.Module):
	"""Extract representations from miRNA modality"""
	def __init__(self, mirna_length, m_length):
		"""miRNA Nerual Network with fully connected layers
		Parameters
		----------
		mirna_length: int
			The input dimension of miRNA modality.

		m_length: int
			Output dimension.

		"""
		super(MirnaNet, self).__init__()

		# Linear Layers
		self.mirna_hidden1 = nn.Linear(mirna_length, 400)
		self.mirna_hidden2 = nn.Linear(400, m_length)

		# Batch Normalization Layers
		self.bn1 = nn.BatchNorm1d(400)
		self.bn2 = nn.BatchNorm1d(m_length)

		# Dropout Layer
		self.dropout_layer1 = nn.Dropout(p=0.3)
		self.dropout_layer2 = nn.Dropout(p=0.4)

	def forward(self, mirna_input):
		mirna = torch.relu(self.bn1(self.mirna_hidden1(mirna_input)))
		mirna = self.dropout_layer1(mirna)
		mirna = torch.relu(self.bn2(self.mirna_hidden2(mirna)))

		mirna = self.dropout_layer2(mirna)

		return mirna




class MrnaNet(nn.Module):
	"""Extract representations from mRNA modality"""
	def __init__(self, mrna_length, m_length):
		"""miRNA Nerual Network with fully connected layers
		Parameters
		----------
		mrna_length: int
			The input dimension of mRNA modality.

		m_length: int
			Output dimension.

		"""
		super(MrnaNet, self).__init__()

		# Linear Layers
		self.mrna_hidden1 = nn.Linear(mrna_length, m_length)
		self.bn1 = nn.BatchNorm1d(m_length)

		self.dropout_layer2 = nn.Dropout(p=0.4)

	def forward(self, mrna_input):
		mrna = torch.relu(self.bn1(self.mrna_hidden1(mrna_input)))
		mrna = self.dropout_layer2(mrna)

		return mrna




class Loss(nn.Module):
	"""docstring for Loss"""
	def __init__(self, trade_off=0.3, mode='total'):
		"""
		Parameters
		----------
		trade_off: float (Default:0.3)
			To balance the unsupervised loss and cox loss.

		mode: str (Default:'total')
			To determine which loss is used.
		"""
		super(Loss, self).__init__()
		self.trade_off = trade_off
		self.mode = mode


	def _negative_log_likelihood_loss(self, pred_hazard, event, time):
		risk = pred_hazard['hazard']
		_, idx = torch.sort(time, descending=True)
		event = event[idx]
		risk = risk[idx].squeeze()

		hazard_ratio = torch.exp(risk)
		log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-6)
		uncensored_likelihood = risk - log_risk
		censored_likelihood = uncensored_likelihood * event

		num_observed_events = torch.sum(event) + 1e-6
		neg_likelihood = -torch.sum(censored_likelihood) / num_observed_events

		return neg_likelihood


	def _random_match(self, batch_size):
		idx = list(range(batch_size))
		split_size = int(batch_size * 0.5)
		shuffle(idx)
		x1, x2 = idx[:split_size], idx[split_size:]
		if len(x1) != len(x2):
			x1.append(x2[0])

		return x1, x2


	def _contrastive_loss1(self, x1_idx, x2_idx, representation, modalities, margin=0.2):
		"""
		Only one modality
		"""
		con_loss = 0
		modality = modalities[0]
		for idx1, idx2 in zip(x1_idx, x2_idx):
			dis_x_y = torch.cosine_similarity(representation[modality][idx1],
													representation[modality][idx2], dim=0)
			con_loss += torch.pow(torch.clamp(margin+dis_x_y, min=0.0), 2)

		return con_loss / len(x1_idx)


	def _contrastive_loss2(self, x1_idx, x2_idx, representation, modalities, margin=0.2):
		"""
		More than one modality
		"""
		con_loss = 0
		for idx1, idx2 in zip(x1_idx, x2_idx):
			dis_x_x = 0
			dis_y_y = 0
			for i in range(len(modalities)-1):
				for j in range(i+1, len(modalities)):
					dis_x_x += torch.cosine_similarity(representation[modalities[i]][idx1],
														representation[modalities[j]][idx1], dim=0)
					dis_y_y += torch.cosine_similarity(representation[modalities[i]][idx2],
														representation[modalities[j]][idx2], dim=0)
			dis_x_y = 0
			for modality in modalities:
				dis_x_y += torch.cosine_similarity(representation[modality][idx1],
													representation[modality][idx2], dim=0)
			con_loss += torch.pow(torch.clamp(margin+dis_x_y-0.5*dis_x_x-0.5*dis_y_y, min=0.0), 2)

		return con_loss / len(x1_idx)


	def _unsupervised_similarity_loss(self, representation, modalities, t=1):
		k = 0
		similarity_loss = 0
		if len(modalities) > 1:
			while k < t:
				x1_idx, x2_idx = self._random_match(representation[modalities[0]].shape[0])
				similarity_loss += self._contrastive_loss2(x1_idx, x2_idx, representation, modalities)
				k += 1
		else:
			while k < t:
				x1_idx, x2_idx = self._random_match(representation[modalities[0]].shape[0])
				similarity_loss += self._contrastive_loss1(x1_idx, x2_idx, representation, modalities)
				k += 1

		return similarity_loss / t


	def _cross_entropy_loss(self, pred_hazard, event):
		return F.nll_loss(pred_hazard['score'], event)


	def forward(self, representation, modalities, pred_hazard, event, time):
		"""
		When mode = 'total' we use the proposed loss function,
		mode = 'only_cox' we remove the unsupervised loss.
		"""
		if self.mode == 'total':
			loss = self._cross_entropy_loss(pred_hazard, event) + self._negative_log_likelihood_loss(pred_hazard, event, time) + self.trade_off * self._unsupervised_similarity_loss(representation, modalities)
		elif self.mode == 'only_cox':
			loss = self._negative_log_likelihood_loss(pred_hazard, event, time)

		return loss


class Net(nn.Module):
	def __init__(self, modalities, m_length, fusion_method='attention', device=None,
					input_modality_dim={'clinical':4, 'mRNA':1579, 'miRNA':743, 'CNV':2711}):
		"""
		Parameters
		----------
		modalities: list
			Used modalities.

		m_length: int
			Representation length.

		"""
		super(Net, self).__init__()
		self.data_modalities = modalities
		self.m_length = m_length
		self.dim = input_modality_dim
		self.device = device

		self.submodel_pipeline = {}
		# clinical -----------------------------------------------#
		if 'clinical' in self.data_modalities:
			self.clinical_submodel = ClinicalEmbeddingNet(m_length=self.m_length, clinical_length=self.dim["clinical"])
			self.submodel_pipeline['clinical'] = self.clinical_submodel

		# mRNA ---------------------------------------------------#
		if 'mRNA' in self.data_modalities:
			self.mRNA_submodel = MrnaNet(mrna_length=self.dim['mRNA'], m_length=self.m_length)
			self.submodel_pipeline['mRNA'] = self.mRNA_submodel

		# meth ---------------------------------------------------#
		if 'meth' in self.data_modalities:
			self.meth_submodel = MrnaNet(mrna_length=self.dim['meth'], m_length=self.m_length)
			self.submodel_pipeline['meth'] = self.meth_submodel


		# mut ---------------------------------------------------#
		if 'mut' in self.data_modalities:
			self.mut_submodel = CnvNet(cnv_length=self.dim['mut'], m_length=self.m_length)
			self.submodel_pipeline['mut'] = self.mut_submodel

        
		# miRNA --------------------------------------------------#
		if 'miRNA' in self.data_modalities:
			self.miRNA_submodel = MirnaNet(mirna_length=self.dim['miRNA'], m_length=self.m_length)
			self.submodel_pipeline['miRNA'] = self.miRNA_submodel

		if 'rppa' in self.data_modalities:
			self.rppa_submodel = MirnaNet(mirna_length=self.dim['rppa'], m_length=self.m_length)
			self.submodel_pipeline['rppa'] = self.rppa_submodel

		# CNV ----------------------------------------------------#
		if 'CNV' in self.data_modalities:
			self.CNV_submodel = CnvNet(cnv_length=self.dim['CNV'], m_length=self.m_length)
			self.submodel_pipeline['CNV'] = self.CNV_submodel

		# Fusion -------------------------------------------------#
		if len(self.data_modalities) > 1:
			if fusion_method == 'attention':
				self.fusion = Attention(m_length=self.m_length, modalities=self.data_modalities, device=self.device)
			else:
				self.fusion = FixedAttention(m_length=self.m_length, modalities=self.data_modalities, device=self.device)


		# Survival prediction
		self.hazard_layer1 = nn.Linear(m_length, 1)

		self.label_layer1 = nn.Linear(m_length, 2)


	def forward(self, x):
		"""
		Parameters
		----------
		x: dictionary
			Input data from different modality, like:
			{'clinical': tensor, 'mRNA':tensor, }
		"""
		# Extract representations from different modality
		representation = {}
		flag = 1
		for modality in x:
			if modality in ['clinical_categorical', 'clinical_continuous']:
				if flag:
					representation['clinical'] = self.submodel_pipeline['clinical'](x['clinical_continuous'])
					flag = 0
				continue
			representation[modality] = self.submodel_pipeline[modality](x[modality])
		# fusion part
		if len(self.data_modalities) > 1:
			x = self.fusion(representation)
		else:
			x = representation[self.data_modalities[0]]

		# survival predict
		hazard = self.hazard_layer1(x)

		score = F.log_softmax(self.label_layer1(x), dim=1)

		return {'hazard':hazard, 'score':score}, representation
	



class ModelCoach:
	def __init__(self, model, modalities, dataloaders, optimizer, criterion, device=None):
		self.model = model
		self.modalities = modalities
		self.dataloaders = dataloaders
		self.optimizer = optimizer
		self.criterion = criterion.to(device)

		# self.best_perf = {'epoch a': 0.0, 'epoch b': 0.0, 'epoch c': 0.0}
		# self.best_wts = {'epoch a': None, 'epoch b': None, 'epoch c': None}
		
		self.best_perf = {'best_score': 0.0}
		self.best_wts = {'best_wts': None}

		self.current_perf = {'epoch a': 0}
		self.device = device


	def _data2device(self, data):
		for modality in data:
			data[modality] = data[modality].to(self.device)

		return data

	def _compute_loss(self, representation, modalities, pred_hazard, event, time):
		loss = self.criterion(representation=representation, modalities=modalities, pred_hazard=pred_hazard, event=event, time=time)

		return loss

	def _log_info(self, phase, logger, epoch, epoch_loss, epoch_c_index):
		info = {phase + '_loss': epoch_loss,
				phase + '_c_index': epoch_c_index}

		for tag, value in info.items():
			logger.add_scalar(tag, value, epoch)

	def _process_data_batch(self, data, data_label, phase):
		"""
		Train model using a batch.
		"""
		data = self._data2device(data)
		event = data_label['label'][:, 0].to(self.device)
		time = data_label['label'][:, 1].to(self.device)

		with torch.set_grad_enabled(phase == 'train'):
			hazard, representation = self.model(data)
			loss = self._compute_loss(representation, self.modalities, hazard, event, time)

			if phase == 'train':
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

		return loss, hazard['hazard'], time, event


	def _run_training_loop(self, num_epochs, scheduler, info_freq, log_dir):
		logger = SummaryWriter(log_dir)
		log_info = True

		if info_freq is not None:
			def print_header():
				sub_header = ' Epoch     Loss     Ctd     Loss     Ctd'
				print('-' * (len(sub_header) + 2))
				print('             Training        Validation')
				print('           ------------     ------------')
				print(sub_header)
				print('-' * (len(sub_header) + 2))

			print()
			print_header()

		for epoch in range(1, num_epochs+1):
			if info_freq is None:
				print_info = False
			else:
				print_info = (epoch == 1) or (epoch % info_freq == 0)

			for phase in ['train', 'val']:
				if phase == 'train':
					self.model.train()
				else:
					self.model.eval()

				running_loss = []

				if print_info or log_info:
					running_sample_time = torch.FloatTensor().to(self.device)
					running_sample_event = torch.LongTensor().to(self.device)
					running_hazard = torch.FloatTensor().to(self.device)

				for data, data_label in self.dataloaders[phase]:
					loss, hazard, time, event = self._process_data_batch(data, data_label, phase)

					running_loss.append(loss.item())
					running_sample_time = torch.cat((running_sample_time, time.data.float()))
					running_sample_event = torch.cat((running_sample_event, event.long().data))
					running_hazard = torch.cat((running_hazard, hazard.detach()))

				epoch_loss = torch.mean(torch.tensor(running_loss))

				epoch_c_index = concordance_index(running_sample_time.cpu().numpy(), -running_hazard.cpu().numpy(), running_sample_event.cpu().numpy())

				if print_info:
					if phase == 'train':
						message = f' {epoch}/{num_epochs}'
					space = 10 if phase == 'train' else 27
					message += ' ' * (space - len(message))
					message += f'{epoch_loss:.4f}' 
					space = 19 if phase == 'train' else 36
					message += ' ' * (space - len(message))
					message += f'{epoch_c_index:.3f}' 

					if phase == 'val':
						print(message)


				if log_info:
					self._log_info(phase=phase, logger=logger, epoch=epoch,
									epoch_loss=epoch_loss, epoch_c_index=epoch_c_index)

				if phase == 'val':
					if scheduler:
						scheduler.step(epoch_c_index)

					# Record current performance
					k = list(self.current_perf.keys())[0]
					self.current_perf['epoch' + str(epoch)] = self.current_perf.pop(k)
					self.current_perf['epoch' + str(epoch)] = epoch_c_index

					# Record top best model
					# for k, v in self.best_perf.items():
					if epoch_c_index > self.best_perf['best_score']:
							# self.best_perf['best_score'] = self.best_perf.pop(k)
						self.best_perf['best_score'] = epoch_c_index
							# self.best_wts['best_wts'] = self.best_wts.pop(k)
						self.best_wts['best_wts'] = copy.deepcopy(self.model.state_dict())
							# break

	def train(self, num_epochs, scheduler, info_freq, log_dir):
		self._run_training_loop(num_epochs, scheduler, info_freq, log_dir)
		print('>>>>> Best validation C-indices:')
		for k, v in self.best_perf.items():
			print(f'     {v} ({k})')
			


class _BaseModelWithDataLoader:
    def __init__(self, modalities, m_length, dataloaders, fusion_method='attention', device=None, input_modality_dim={}):
        self.data_modalities = modalities
        self.m_length = m_length
        self.dataloaders = dataloaders
        self.device = device
        self.fusion_method = fusion_method
        self.input_modality_dim = input_modality_dim

        self._instantiate_model()
        self.model_blocks = [name for name, _ in self.model.named_children()]



    def _instantiate_model(self, move2device=True):
        print('Instantiate Survival model...')
        self.model = Net(self.data_modalities, self.m_length, self.fusion_method, self.device, self.input_modality_dim)

        if move2device:
            self.model = self.model.to(self.device)


class Model(_BaseModelWithDataLoader):
    def __init__(self, modalities, m_length, dataloaders, fusion_method='attention', trade_off=0.3, mode='total', device=None, input_modality_dim={}):
        super().__init__(modalities, m_length, dataloaders, fusion_method, device, input_modality_dim)

        self.optimizer = Adam
        self.loss = Loss(trade_off=trade_off, mode=mode)

    def fit(self, num_epochs, lr, info_freq, log_dir, lr_factor=0.1, scheduler_patience=5):
        self._instantiate_model()
        optimizer = self.optimizer(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='max', factor=lr_factor,
            patience = scheduler_patience, #verbose=True, 
            threshold=1e-3,
            threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)


        model_coach = ModelCoach(
            model=self.model, modalities=self.data_modalities,
            dataloaders=self.dataloaders, optimizer=optimizer,
            criterion=self.loss, device=self.device)

        model_coach.train(num_epochs, scheduler, info_freq, log_dir)

        self.model = model_coach.model
        self.best_model_weights = model_coach.best_wts
        self.best_c_index = model_coach.best_perf
        self.current_c_index = model_coach.current_perf

    def save_weights(self, saved_epoch, prefix, weight_dir):
        print('Saving model weights to file:')
        if saved_epoch == 'current':
            epoch = list(self.current_concord.keys())[0]
            value = self.current_concord[epoch]
            file_name = os.path.join(
                weight_dir,
                f'{prefix}_{epoch}_c_index{value:.2f}.pth')
        else:
            file_name = os.path.join(
                weight_dir,
                f'{prefix}_{saved_epoch}_' + \
                f'c_index{self.best_concord_values[saved_epoch]:.2f}.pth')
            self.model.load_state_dict(self.best_model_weights[saved_epoch])

        torch.save(self.model.stat_dict(), file_name)
        print(' ', file_name)

    def test(self):
        self.model.load_state_dict(self.best_model_weights['best_wts'])
        self.model = self.model.to(self.device)

    def load_weights(self, path):
        print('Load model weights:')
        print(path)
        self.model.load_state_dict(torch.load(path))
        self.model = self.model.to(self.device)

    def predict(self, data, data_label):
        for modality in data:
            data[modality] = data[modality].to(self.device)
        event = data_label['label'][:, 0].to(self.device)
        time = data_label['label'][:, 1].to(self.device)

        return self.model(data), event, time
	




def setup_seed(seed):
	"""
	Set random seed for torch.
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True

def test_gpu():
	"""
	Detect the hardware: GPU or CPU?
	"""
	print('GPU？', torch.cuda.is_available())
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('The device is ', device)
	return device

def evaluate_model(c_index_arr):
	"""
	Calculate the Mean and the Standard Deviation about c-index array.
	"""
	m = np.sum(c_index_arr, axis=0) / len(c_index_arr)
	s = np.std(c_index_arr)
	return m, s



def get_dataloaders(mydataset, train_sampler, val_sampler, test_sampler, batch_size):
	"""
	Parameters
	----------
	mydataset: Dataset

	train_sampler: array
		Patient indexs in train set.

	val_sampler: array
		Patient indexs in validation set.

	test_sampler: array
		Patient indexs in test set.

	batch_size: int
		Number of patients in each batch.

	Return
	------
	A dictionary of train/validation/test set, like
		{'train': train_loader, 'val': val_loader, 'test': test_loader}
	"""
	
	dataloaders = {}
	dataloaders['train'] = DataLoader(mydataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_sampler))
	dataloaders['val'] = DataLoader(mydataset, batch_size=len(val_sampler), sampler=SubsetRandomSampler(val_sampler))
	dataloaders['test'] = DataLoader(mydataset, batch_size=len(test_sampler), sampler=test_sampler)

	print('Dataset sizes (# patients):')
	print('train: ', len(train_sampler))
	print('  val: ', len(val_sampler))
	print(' test: ', len(test_sampler))
	print()
	print('Batch size: ', batch_size)

	return dataloaders


def compose_run_tag(model, lr, dataloaders, log_dir, suffix=''):
	"""
	Make the tag about modality and learning rate.
	"""
	def add_string(string, addition, sep='_'):
		if not string:
			return addition
		else: return string + sep + addition

	data = None
	for modality in model.data_modalities:
		data = add_string(data, modality)

	run_tag = f'{data}_lr{lr}'

	run_tag += suffix

	print(f'Run tag: "{run_tag}"')

	tb_log_dir = os.path.join(log_dir, run_tag)

	return run_tag


def save_5fold_results(c_index_arr, run_tag):
	"""
	Save the results after 5 fold cross validation.
	"""
	m, s = evaluate_model(c_index_arr)
	with open(f'proposed_{run_tag}.txt', 'w') as file:
		file.write(str(c_index_arr))
		file.write(f"\n Mean: {m}")
		file.write(f"\n Std: {s}")
	file.close()
	



def preprocess_clinical_data(clinical_df):

    clin_data_continuous = clinical_df.iloc[:, np.where(clinical_df.dtypes != "object")[0]]
    clin_data_categorical = clinical_df.iloc[:, np.where(clinical_df.dtypes == "object")[0]]

    return clin_data_categorical, clin_data_continuous


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        """
        Parameters
        ----------
        modalities: list
            Used modalities

        data_path: str
            The path of used data.

        Returns
        -------
        data: dictionary
            {'clin_data_categorical': ,..,'mRNA': ...}

        data_label: dictionary
            {'label':[[event, time]]}
        """
        super(MyDataset, self).__init__()
        feature_names = data.drop(["OS", "OS_days"], axis=1).columns
        column_types = pd.Series(feature_names).str.rsplit("_").apply(lambda x: x[0]).values
        modalities = np.unique(column_types)
        self.data_modalities = modalities        
        target_data = data[["OS", "OS_days"]]
        
        clin_data_continuous = data.iloc[:, np.where(column_types == "clinical")[0]]
        self.target = target_data.values.tolist()

        # clinical
        if 'clinical' in self.data_modalities:
            self.clin_cont = clin_data_continuous.values.tolist()

        # mRNA
        if 'gex' in self.data_modalities:
            data_mrna = data.iloc[:, np.where(column_types == "gex")[0]]
            self.data_mrna = data_mrna.values.tolist()

        # miRNA
        if 'mirna' in self.data_modalities:
            data_mirna = data.iloc[:, np.where(column_types == "mirna")[0]]
            self.data_mirna = data_mirna.values.tolist()

        # miRNA
        if 'rppa' in self.data_modalities:
            data_rppa = data.iloc[:, np.where(column_types == "rppa")[0]]
            self.data_rppa = data_rppa.values.tolist()

        # CNV
        if 'cnv' in self.data_modalities:
            data_cnv = data.iloc[:, np.where(column_types == "cnv")[0]]
            self.data_cnv = data_cnv.values.tolist()

        # CNV
        if 'meth' in self.data_modalities:
            data_meth = data.iloc[:, np.where(column_types == "meth")[0]]
            self.data_meth = data_meth.values.tolist()

    
        if 'mut' in self.data_modalities:
            data_mut = data.iloc[:, np.where(column_types == "mut")[0]]
            self.data_mut = data_mut.values.tolist()

    def __len__(self):
        return len(self.clin_cat)

    def __getitem__(self, index):
        data = {}
        data_label = {}
        target_y = np.array(self.target[index], dtype='int')
        target_y = torch.from_numpy(target_y)
        data_label['label'] = target_y.type(torch.LongTensor)

        
        if 'clinical' in self.data_modalities:
            clin_conti = np.array(self.clin_cont[index]).astype(np.float32)
            clin_conti = torch.from_numpy(clin_conti)
            data['clinical_continuous'] = clin_conti


        if 'gex' in self.data_modalities:
            mrna = np.array(self.data_mrna[index])
            mrna = torch.from_numpy(mrna)
            data['mRNA'] = mrna.type(torch.FloatTensor)

        if 'mut' in self.data_modalities:
            mut = np.array(self.data_mut[index])
            mut = torch.from_numpy(mut)
            data['mut'] = mut.type(torch.FloatTensor)

        if 'rppa' in self.data_modalities:
            rppa = np.array(self.data_rppa[index])
            rppa = torch.from_numpy(rppa)
            data['rppa'] = rppa.type(torch.FloatTensor)

        if 'meth' in self.data_modalities:
            meth = np.array(self.data_meth[index])
            meth = torch.from_numpy(meth)
            data['meth'] = meth.type(torch.FloatTensor)

        if 'mirna' in self.data_modalities:
            mirna = np.array(self.data_mirna[index])
            mirna = torch.from_numpy(mirna)
            data['miRNA'] = mirna.type(torch.FloatTensor)

        if 'cnv' in self.data_modalities:
            cnv = np.array(self.data_cnv[index])
            cnv = torch.from_numpy(cnv)
            data['CNV'] = cnv.type(torch.FloatTensor)

        return data, data_label
