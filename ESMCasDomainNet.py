
import torch,math,esm,pickle,datetime
from torch.utils.data import DataLoader, Dataset
# from torch.optim import Adam
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score


def local_load_model(model_path="esm2_t36_3B_UR50D.pt"):
	model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	# device = "cpu"
	model = model.to(device)
	return model, alphabet,device

class ProteinDataset(Dataset):
    def __init__(self, data_raw):
        self.sequences, self.labels = zip(*data_raw)  # Unpack the data
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return sequence, label

class ESM_MultiInput(nn.Module):
    def __init__(self, esm_model, alphabet, num_labels=4, freeze_layers=3):
        super(ESM_MultiInput, self).__init__()
        self.esm_model = esm_model
        self.num_layers = esm_model.num_layers
        self.batch_converter = alphabet.get_batch_converter()
        self.num_labels = num_labels

        print(f'embed_dim: {esm_model.embed_dim}\nfreeze_layers: {freeze_layers}\nnum_layers: {self.num_layers}')

        # ¶³½áÇ°ÃæËùÓÐµÄ²ã
        for i, param in enumerate(self.esm_model.parameters()):
            param.requires_grad = False
        
        # ½â¶³×îºó freeze_layers ²ã
        # Ê¹ÓÃÊÊºÏ ESM2 Ä£ÐÍµÄ²ãË÷Òý·½Ê½
        for i in range(self.num_layers - freeze_layers, self.num_layers):
            for param in self.esm_model.layers[i].parameters():  # ·ÃÎÊ²ãµÄÕýÈ··½Ê½
                param.requires_grad = True

        # ·ÖÀàÆ÷£¬½«Ã¿¸öÎ»ÖÃµÄ±íÊ¾Ó³Éäµ½ num_labels Î¬¶È
        self.classifier = nn.Linear(esm_model.embed_dim, num_labels)

    def forward(self, original_batch, device='cpu'):
        # Ensure original_batch is a list of tuples with (sequence_name, sequence)
        batch_size = len(original_batch)
        if not isinstance(original_batch[0], tuple):
            original_batch = [(f"seq_{i}", seq.strip()) for i, seq in enumerate(original_batch)]

        # Tokenize the batch using ESM's batch_converter
        _, _, tokenized_batch = self.batch_converter(original_batch)
        tokenized_batch = tokenized_batch[:, 1:-1]  # Remove <cls> and <eos> tokens
        tokenized_batch = tokenized_batch.to(device)

        # Extract representations from ESM
        with torch.no_grad():
            x_original = self.esm_model(
                tokenized_batch, repr_layers=[self.num_layers], return_contacts=False
            )['representations'][self.num_layers]  # Shape: [batch_size, seq_length, feature_dim]
        
        # ¶ÔÃ¿¸öÎ»ÖÃµÄ±íÊ¾½øÐÐ·ÖÀà£¬Êä³öÃ¿¸öÎ»ÖÃµÄ num_labels Àà±ðµÄ logit
        predictions = self.classifier(x_original)  # Shape: [batch_size, seq_length, num_labels]
        
        max_seq_len = max(len(seq.strip()) for _, seq in original_batch)

        return predictions, max_seq_len


class ESM_MultiInput_MLP(nn.Module):
	def __init__(self, esm_model, alphabet, num_labels=4, freeze_layers=3, dropout_rate=0.1):
		super(ESM_MultiInput_MLP, self).__init__()
		self.esm_model = esm_model
		self.num_layers = esm_model.num_layers
		self.batch_converter = alphabet.get_batch_converter()
		self.num_labels = num_labels

		print(f'embed_dim: {esm_model.embed_dim}\nfreeze_layers: {freeze_layers}\nnum_layers: {self.num_layers}')

		# ¶³½áÇ°ÃæËùÓÐµÄ²ã
		for param in self.esm_model.parameters():
			param.requires_grad = False

		# ½â¶³×îºó freeze_layers ²ã
		for i in range(self.num_layers - freeze_layers, self.num_layers):
			for param in self.esm_model.layers[i].parameters():
				param.requires_grad = True

		# MLP ²ã£ºÁ½²ãÈ«Á¬½ÓÍøÂç + ReLU + Dropout
		# self.mlp = nn.Sequential(
			# nn.Linear(esm_model.embed_dim, hidden_dim),
			# nn.ReLU(),
			# nn.Dropout(dropout_rate),
			# nn.Linear(hidden_dim, num_labels)
		# )
				
		# self.mlp = nn.Sequential(
			# nn.Linear(esm_model.embed_dim, 512),  # µÚÒ»²ã£ºÈ«Á¬½Ó²ã£¨ÊäÈë -> 512£©
			# nn.ReLU(),                            # µÚ¶þ²ã£º¼¤»îº¯Êý
			# nn.Dropout(dropout_rate),                       # µÚÈý²ã£ºDropout²ã
			# nn.Linear(512, 512),                   # µÚËÄ²ã£ºÈ«Á¬½Ó²ã£¨512 -> 512£©
			# nn.ReLU(),                             # µÚÎå²ã£º¼¤»îº¯Êý
			# nn.Dropout(dropout_rate),                        # µÚÁù²ã£ºDropout²ã
			# nn.Linear(512, num_labels)              # µÚÆß²ã£ºÈ«Á¬½Ó²ã£¨256 -> Êä³ö£©
		# )

		self.mlp = nn.Sequential(
			nn.Linear(esm_model.embed_dim, 1024),     # µÚÒ»²ã£ºÈ«Á¬½Ó²ã£¨ÊäÈë -> 1024£©
			nn.ReLU(),                                 # µÚ¶þ²ã£ºReLU¼¤»îº¯Êý
			nn.Dropout(dropout_rate),                 # µÚÈý²ã£ºDropout²ã

			nn.Linear(1024, 1024),                    # µÚËÄ²ã£ºÒþ²Ø²ã£¨1024 -> 1024£©
			nn.ReLU(),                                 # µÚÎå²ã£ºReLU¼¤»îº¯Êý
			nn.Dropout(dropout_rate),                 # µÚÁù²ã£ºDropout²ã

			nn.Linear(1024, num_labels)               # µÚÊ®²ã£ºÊä³ö²ã£¨1024 -> Êä³ö£©
		)

	def forward(self, original_batch, device='cpu'):
		# È·±£ original_batch ÊÇ (sequence_name, sequence) ¸ñÊ½µÄÔª×éÁÐ±í
		if not isinstance(original_batch[0], tuple):
			original_batch = [(f"seq_{i}", seq.strip()) for i, seq in enumerate(original_batch)]

		# Ê¹ÓÃ ESM µÄ batch_converter ½øÐÐ tokenization
		_, _, tokenized_batch = self.batch_converter(original_batch)
		tokenized_batch = tokenized_batch[:, 1:-1]  # È¥µô <cls> ºÍ <eos>
		tokenized_batch = tokenized_batch.to(device)

		# ÌáÈ¡ ESM ±íÊ¾²ã
		with torch.no_grad():
			x_original = self.esm_model(
				tokenized_batch, repr_layers=[self.num_layers], return_contacts=False
			)['representations'][self.num_layers]  # [batch_size, seq_length, feature_dim]

		# Í¨¹ý MLP ½øÐÐ·ÖÀà
		predictions = self.mlp(x_original)  # [batch_size, seq_length, num_labels]

		max_seq_len = max(len(seq.strip()) for _, seq in original_batch)

		return predictions, max_seq_len


class ESM_MultiInput_0(nn.Module):
	def __init__(self, esm_model, alphabet, num_labels=4):
		super(ESM_MultiInput, self).__init__()
		self.esm_model = esm_model
		self.num_layers = esm_model.num_layers
		self.batch_converter = alphabet.get_batch_converter()
		# self.hidden_dim = hidden_dim
		self.num_labels = num_labels
		print(f'embed_dim: {esm_model.embed_dim}')
		# ÓÃÓÚ´¦ÀíÔ­Ê¼ÐòÁÐµÄÏßÐÔ²ã
		# self.linear_original = nn.Linear(esm_model.embed_dim, hidden_dim)
		
		# ·ÖÀàÆ÷£¬½«Ã¿¸öÎ»ÖÃµÄ±íÊ¾Ó³Éäµ½ num_labels Î¬¶È
		# self.classifier = nn.Linear(hidden_dim, num_labels)
		self.classifier = nn.Linear(esm_model.embed_dim, num_labels)

	def forward(self, original_batch, device='cpu'):
		# Ensure original_batch is a list of tuples with (sequence_name, sequence)
		batch_size = len(original_batch)
		if not isinstance(original_batch[0], tuple):
			original_batch = [(f"seq_{i}", seq.strip()) for i, seq in enumerate(original_batch)]

		# Tokenize the batch using ESM's batch_converter
		_, _, tokenized_batch = self.batch_converter(original_batch)
		tokenized_batch = tokenized_batch[:, 1:-1]  # Remove <cls> and <eos> tokens
		tokenized_batch = tokenized_batch.to(device)

 
		# Extract representations from ESM
		with torch.no_grad():
			x_original = self.esm_model(
				tokenized_batch, repr_layers=[self.num_layers], return_contacts=False
			)['representations'][self.num_layers]  # Shape: [batch_size, seq_length, feature_dim]
		
		# ½«Ã¿¸öÎ»ÖÃµÄ±íÊ¾Í¨¹ýÏßÐÔ²ãÓ³Éäµ½ hidden_dim
		# x_original = self.linear_original(x_original)  # Shape: [batch_size, seq_length, hidden_dim]
		# ¶ÔÃ¿¸öÎ»ÖÃµÄ±íÊ¾½øÐÐ·ÖÀà£¬Êä³öÃ¿¸öÎ»ÖÃµÄ num_labels Àà±ðµÄ logit
		predictions = self.classifier(x_original)  # Shape: [batch_size, seq_length, num_labels]
		
		
		max_seq_len = max(len(seq.strip()) for _, seq in original_batch)

		return predictions,max_seq_len





def train_model(model, train_loader, val_loader, epochs=10, learning_rate=1e-4,device='cuda',weights = torch.tensor([0.01,1,0.7,1,1.5])):


	current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	train_log=open(f'{current_time}_train.log','a')


	train_log.write(f'Computed class weights: {weights}\n')
	train_log.write(f'learning_rate {learning_rate}\n')
	train_log.close()

	# ½«Ä£ÐÍ·Åµ½Ö¸¶¨Éè±¸£¨CPU »ò GPU£©
	model.to(device)

	# ¶¨ÒåÊý¾Ý¼ÓÔØÆ÷
	# train_loader = DataLoader(ProteinDataset(*train_data), batch_size=batch_size, shuffle=True)
	# val_loader = DataLoader(ProteinDataset(*val_data), batch_size=batch_size, shuffle=False)

	# ¶¨ÒåÓÅ»¯Æ÷ºÍËðÊ§º¯Êý
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	# criterion = nn.CrossEntropyLoss()

	# weights = torch.tensor([0.01,1,0.7,1,1.5])
	criterion = nn.CrossEntropyLoss(weight=weights.to(device),ignore_index=-1)
	# criterion = nn.CrossEntropyLoss(weight=weights.to(device))
	criterion2 = nn.CrossEntropyLoss()
	# ´æ´¢ÑµÁ·ºÍÑéÖ¤µÄËðÊ§Óë×¼È·ÂÊ
	train_losses = []
	val_losses = []
	train_accuracies = []
	val_accuracies = []

	# ÑµÁ·¹ý³Ì
	for epoch in range(epochs):
		train_log=open(f'{current_time}_train.log','a')
		model.train()  # ÉèÖÃÄ£ÐÍÎªÑµÁ·Ä£Ê½ optimizer
		train_loss = 0.0
		correct_train = 0
		correct_train2 = 0
		correct_train3 = 0
		total_train = 0
		total_train2 = 0
		total_train3 = 0
		all_train_preds = []
		all_train_labels = []
		
		for sequences, labels in train_loader:
			# print(f'{labels.shape} {mask_padding.shape}')
			# print(sequences)
			# sequences, labels = sequences.to(device), labels.to(device)
			# Çå¿ÕÌÝ¶È
			# print(labels.tolist())
			labels=labels.to(device)
			optimizer.zero_grad()
			# Ç°Ïò´«²¥
			predictions,max_seq_len = model(original_batch=sequences, device=device)			
			# print(predictions)
			# print(new_labels.shape)
			# ¼ÆËãËðÊ§
			# loss = criterion(predictions, new_labels)
			# print(labels.shape)
			if max_seq_len < labels.size(1):
				labels = labels[:, :max_seq_len]			
			# print(labels.shape)					
			# loss = criterion( predictions.view(-1, predictions.size(-1)), labels.view(-1))
			
			loss = criterion(predictions.reshape(-1, predictions.size(-1)), labels.reshape(-1))
			
			train_loss += loss.item()
			# ·´Ïò´«²¥
			loss.backward()
			optimizer.step()
			# # ¼ÆËãÑµÁ·¼¯µÄ×¼È·ÂÊ
			# correct_train += (predictions == new_labels).sum().item()  # ÕýÈ·Ô¤²âµÄÊýÁ¿
			# total_train += new_labels.numel()  # Ñù±¾×ÜÊý
			# ¼ÆËãÑµÁ·¼¯µÄ×¼È·ÂÊ
			_, predicted = torch.max(predictions, dim=-1)
			
			# correct_train += (predicted == new_labels).sum().item()
			correct_train += (((predicted == labels) & (labels != -1)).sum()).item()
			correct_train2 += (((predicted == labels) & (labels != -1) & (labels != 0)).sum()).item()
			correct_train3 += (((predicted == labels) & (labels == 0)).sum()).item()
			
			total_train += (labels != -1).sum().item()
			total_train2 += ((labels != -1) & (labels != 0)).sum().item()
			total_train3 += (labels == 0).sum().item()
			# total_train += new_labels.numel()
			# ÊÕ¼¯Ô¤²âºÍÕæÊµ±êÇ©£¬ÓÃÓÚ¼ÆËãF1-score
			all_train_preds.extend(predicted.cpu().numpy().flatten())
			all_train_labels.extend(labels.cpu().numpy().flatten())



		# ¼ÆËãÑµÁ·¼¯µÄÆ½¾ùËðÊ§ºÍ×¼È·ÂÊ
		train_loss = train_loss / len(train_loader)
		train_accuracy = correct_train / total_train
		train_accuracy2 = correct_train2 / total_train2
		train_accuracy3 = correct_train3 / total_train3

		# ¼ÆËãÑµÁ·¼¯µÄF1-score precision_score, recall_score, f1_score
		train_f1_total = f1_score(all_train_labels, all_train_preds, labels=[0, 1, 2, 3], average='macro')
		train_precision_score_total = precision_score(all_train_labels, all_train_preds, labels=[0, 1, 2, 3], average='macro')
		train_recall_score_total = recall_score(all_train_labels, all_train_preds, labels=[0, 1, 2, 3], average='macro')
		train_f1_per_class = f1_score(all_train_labels, all_train_preds, labels=[0, 1, 2, 3], average=None)
		train_precision_score_per_class = precision_score(all_train_labels, all_train_preds, labels=[0, 1, 2, 3], average=None)
		train_recall_score_per_class = recall_score(all_train_labels, all_train_preds, labels=[0, 1, 2, 3], average=None)



		# ÑéÖ¤¹ý³Ì
		model.eval()  # ÉèÖÃÄ£ÐÍÎªÆÀ¹ÀÄ£Ê½
		val_loss = 0.0
		correct_val = 0
		correct_val2 = 0
		correct_val3 = 0
		total_val = 0
		total_val2 = 0
		total_val3 = 0
		all_val_preds = []
		all_val_labels = []
		
		with torch.no_grad():  # ½ûÓÃÌÝ¶È¼ÆËã 
			for sequences, labels in val_loader:
				labels=labels.to(device)
				predictions,max_seq_len = model(original_batch=sequences, device=device)

				if max_seq_len < labels.size(1):
					labels = labels[:, :max_seq_len]			
									
				# print(predictions)
				# print(new_labels.shape)
				# ¼ÆËãËðÊ§
				# loss = criterion(predictions, new_labels)
				loss = criterion( predictions.reshape(-1, predictions.size(-1)), labels.reshape(-1))
				val_loss += loss.item()

				# # ¼ÆËãÑµÁ·¼¯µÄ×¼È·ÂÊ
				# correct_train += (predictions == new_labels).sum().item()  # ÕýÈ·Ô¤²âµÄÊýÁ¿
				# total_train += new_labels.numel()  # Ñù±¾×ÜÊý
				# ¼ÆËãÑµÁ·¼¯µÄ×¼È·ÂÊ
				_, predicted = torch.max(predictions, dim=-1)
				
				# correct_train += (predicted == new_labels).sum().item()
				correct_val += (((predicted == labels) & (labels != -1)).sum()).item()
				correct_val2 += (((predicted == labels) & (labels != -1) & (labels != 0)).sum()).item()
				correct_val3 += (((predicted == labels) & (labels == 0)).sum()).item()
				
				total_val += (labels != -1).sum().item()
				total_val2 += ((labels != -1) & (labels != 0)).sum().item()
				total_val3 += (labels == 0).sum().item()
				# total_train += new_labels.numel()


				# ÊÕ¼¯Ô¤²âºÍÕæÊµ±êÇ©£¬ÓÃÓÚ¼ÆËãF1-score
				all_val_preds.extend(predicted.cpu().numpy().flatten())
				all_val_labels.extend(labels.cpu().numpy().flatten())




		# ¼ÆËãÑéÖ¤¼¯µÄÆ½¾ùËðÊ§ºÍ×¼È·ÂÊ
		val_loss = val_loss / len(val_loader)
		val_accuracy = correct_val / total_val
		val_accuracy2 = correct_val2 / total_val2
		val_accuracy3 = correct_val3 / total_val3
		# ¼ÆËãÑéÖ¤¼¯µÄF1-score  precision_score, recall_score, f1_score
		val_f1_total = f1_score(all_val_labels, all_val_preds, labels=[0, 1, 2, 3], average='macro')
		val_precision_score_total = precision_score(all_val_labels, all_val_preds, labels=[0, 1, 2, 3], average='macro')
		val_recall_score_total = recall_score(all_val_labels, all_val_preds, labels=[0, 1, 2, 3], average='macro')
		val_f1_per_class = f1_score(all_val_labels, all_val_preds, labels=[0, 1, 2, 3], average=None)
		val_precision_score_per_class = precision_score(all_val_labels, all_val_preds, labels=[0, 1, 2, 3], average=None)
		val_recall_score_per_class = recall_score(all_val_labels, all_val_preds, labels=[0, 1, 2, 3], average=None)

		# ´æ´¢Ã¿¸öepochµÄËðÊ§ºÍ×¼È·ÂÊ
		train_losses.append(train_loss)
		val_losses.append(val_loss)
		train_accuracies.append(train_accuracy)
		val_accuracies.append(val_accuracy)

		# ´òÓ¡ÑµÁ·¹ý³ÌÖÐµÄËðÊ§ºÍ×¼È·ÂÊ
		print(f"Epoch {epoch+1}/{epochs} - "
			  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy*100:.10f}% {train_accuracy2*100:.10f}% {train_accuracy3*100:.10f}% - "
			  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy*100:.10f}% {val_accuracy2*100:.10f}% {val_accuracy3*100:.10f}%"
			  f" Train F1 Total: {train_f1_total:.4f} - "
			  f" Train precision_score Total: {train_precision_score_total:.4f} - "
			  f" Train recall_score Total: {train_recall_score_total:.4f} - "
			  f"Train F1 per class: {train_f1_per_class} - "
			  f"Train precision_score per class: {train_precision_score_per_class} - "
			  f"Train recall_score per class: {train_recall_score_per_class} - "
			  f"Val F1 Total: {val_f1_total:.4f} - "
			  f"Val precision_score Total: {val_precision_score_total:.4f} - "
			  f"Val recall_score Total: {val_recall_score_total:.4f} - "
			  f"Val F1 per class: {val_f1_per_class}\n"
			  f"Val precision_score per class: {val_precision_score_per_class}\n"
			  f"Val recall_score per class: {val_recall_score_per_class}\n")
			  
			  
		train_log.write(f"Epoch {epoch+1}/{epochs} - "
			  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy*100:.10f}% {train_accuracy2*100:.10f}% {train_accuracy3*100:.10f}% - "
			  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy*100:.10f}% {val_accuracy2*100:.10f}% {val_accuracy3*100:.10f}%\n"
			f"Train F1 Total: {train_f1_total:.4f} - "
			f"Train precision_score Total: {train_precision_score_total:.4f} - "
			f"Train recall_score Total: {train_recall_score_total:.4f} - "
			f"Train F1 per class: {train_f1_per_class} - "
			f"Train precision_score per class: {train_precision_score_per_class} - "
			f"Train recall_score per class: {train_recall_score_per_class} - "
			f"Val F1 Total: {val_f1_total:.4f} - "
			f"Val precision_score Total: {val_precision_score_total:.4f} - "
			f"Val recall_score Total: {val_recall_score_total:.4f} - "
			f"Val F1 per class: {val_f1_per_class}\n\n"			  
			f"Val precision_score per class: {val_precision_score_per_class}\n\n"			  
			f"Val recall_score per class: {val_recall_score_per_class}\n\n"			  
			  )
			  
		torch.save(model, f'model_complete_{current_time}_{epoch}.pth')   
		train_log.close()
	torch.save(model, f'model_complete_{current_time}.pth') 



	return train_losses, val_losses, train_accuracies, val_accuracies

# ********************************************************************

def predict_and_get_domains(model, sequence, domain_labels = ["Non-structural","RUVC","HNH","HEPN"], device='cuda'):
	with torch.no_grad():
		predictions, _ = model([sequence],device=device)
	predicted_labels = torch.argmax(predictions, dim=-1).cpu().numpy()
	domain_positions = [sequence]
	current_domain = None
	start_pos = 0
	for i, label in enumerate(predicted_labels[0]):   
		domain_name = domain_labels[label]
		 
		if domain_name != current_domain:
			if current_domain is not None:
				 
				domain_positions.append({
					"domain": current_domain,
					"start": start_pos + 1,   
					"end": i                 
				})
			current_domain = domain_name
			start_pos = i
	if current_domain is not None:
		domain_positions.append({
			"domain": current_domain,
			"start": start_pos + 1,
			"end": len(predicted_labels[0])
		})
	return domain_positions



# domain_labels = {
    # "Non-structural": 0,
    # "RUVC": 1,
    # "HNH": 2,
    # "HEPN": 3,
    # "border":4,
# }




# ¼ÓÔØÊý¾Ý
with open("pdb_cas_train_data.pkl", "rb") as f:
    train_data_raw = pickle.load(f)
print(len(train_data_raw))
train_dataset = ProteinDataset(train_data_raw)
train_data_raw=[]
for sequence, label in train_dataset:	
	if not (label == label[0]).all():  
		train_data_raw.append((sequence,label))	
print(len(train_data_raw))

# ÌáÈ¡ËùÓÐ±êÇ©
all_labels = np.concatenate([label for _, label in train_data_raw], axis=0)


# È¥µô±êÇ©ÖÐµÄ -1
all_labels = all_labels[all_labels != -1]

# ¼ÆËã±êÇ©µÄÎ¨Ò»Öµ¼°Æä³öÏÖµÄÆµÂÊ
unique_labels, counts = np.unique(all_labels, return_counts=True)
total_samples = len(all_labels)

# ¼ÆËã 0 ±êÇ©µÄÈ¨ÖØ
if 0 in unique_labels:
    zero_index = np.where(unique_labels == 0)[0][0]
    zero_weight = total_samples / counts[zero_index]
else:
    zero_weight = 0
# ¼ÆËã·Ç 0 ±êÇ©µÄ×ÜÆµÂÊ
non_zero_counts = counts[unique_labels != 0]
non_zero_total = np.sum(non_zero_counts)
non_zero_weight = total_samples / non_zero_total

# ¹¹½¨È¨ÖØÊý×é
weights = []
for label, count in zip(unique_labels, counts):
    if label == 0:
        weights.append(zero_weight)
    else:
        weights.append(non_zero_weight)

weights = torch.tensor(weights, dtype=torch.float)

print("Computed class weights:", weights)



# ²ð·ÖÊý¾Ý¼¯ÎªÑµÁ·¼¯ºÍÑéÖ¤¼¯£¨80% ÑµÁ·£¬20% ÑéÖ¤£©
train_size = int(0.8 * len(train_data_raw))
val_size = len(train_data_raw) - train_size


train_data_raw, val_data_raw = torch.utils.data.random_split(train_data_raw, [train_size, val_size])
batch_size=180
train_loader = DataLoader(train_data_raw, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data_raw, batch_size=batch_size, shuffle=False)

torch.save(train_loader, f'train_loader_ReLU_short.pth')   
torch.save(val_loader, f'val_loader_ReLU_short.pth')   
 

# Load ESM-2 model
# ±¾µØ¼ÓÔØÄ£ÐÍ

#model_path = "/home/miniconda3/Downloads/esm/esm2_t36_3B_UR50D.pt"
model_path = "/home/miniconda3/Downloads/esm/esm2_t33_650M_UR50D.pt"
esm_model, alphabet,device=local_load_model(model_path)
print(device)
# ¼ÙÉèÒÑÓÐÄ£ÐÍºÍÊý¾Ý
# model = ESM_MultiInput(esm_model,alphabet, num_labels=len(np.unique(all_labels)),freeze_layers=6)


model = ESM_MultiInput_MLP(esm_model,alphabet, num_labels=len(np.unique(all_labels)),freeze_layers=3)

# model = ESM_MultiInput(esm_model,alphabet, hidden_dim=2048, num_labels=len(np.unique(all_labels)))

train_model(model,train_loader,val_loader, epochs=1000, learning_rate=1e-4, device=device,weights=weights)
















