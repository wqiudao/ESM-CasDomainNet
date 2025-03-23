import torch,argparse,os,datetime
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages



class ESM_MultiInput_MLP(nn.Module):
	def __init__(self, esm_model, alphabet, num_labels=4, freeze_layers=3, dropout_rate=0.1):
		super(ESM_MultiInput_MLP, self).__init__()
		self.esm_model = esm_model
		self.num_layers = esm_model.num_layers
		self.batch_converter = alphabet.get_batch_converter()
		self.num_labels = num_labels

		print(f'embed_dim: {esm_model.embed_dim}\nfreeze_layers: {freeze_layers}\nnum_layers: {self.num_layers}')

		for param in self.esm_model.parameters():
			param.requires_grad = False

		for i in range(self.num_layers - freeze_layers, self.num_layers):
			for param in self.esm_model.layers[i].parameters():
				param.requires_grad = True


		self.mlp = nn.Sequential(
			nn.Linear(esm_model.embed_dim, 1024),     
			nn.ReLU(),                                 
			nn.Dropout(dropout_rate),                

			nn.Linear(1024, 1024),                   
			nn.ReLU(),                                
			nn.Dropout(dropout_rate),                

			nn.Linear(1024, 1024),                   
			nn.ReLU(),                                
			nn.Dropout(dropout_rate),                

			nn.Linear(1024, num_labels)               
		)

	def forward(self, original_batch, device='cpu'):
		
		if not isinstance(original_batch[0], tuple):
			original_batch = [(f"seq_{i}", seq.strip()) for i, seq in enumerate(original_batch)]

		
		_, _, tokenized_batch = self.batch_converter(original_batch)
		tokenized_batch = tokenized_batch[:, 1:-1]  # del <cls> ºÍ <eos>
		tokenized_batch = tokenized_batch.to(device)

		#   ESM  
		with torch.no_grad():
			x_original = self.esm_model(
				tokenized_batch, repr_layers=[self.num_layers], return_contacts=False
			)['representations'][self.num_layers]  # [batch_size, seq_length, feature_dim]

		#   MLP  
		predictions = self.mlp(x_original)  # [batch_size, seq_length, num_labels]

		max_seq_len = max(len(seq.strip()) for _, seq in original_batch)

		return predictions, max_seq_len


def plot_multiple_protein_domains(data_list, output_prefix='protein_domain_diagram', plots_per_pdf=2,body_height=0.5):

	total_plots = len(data_list)
	if total_plots<plots_per_pdf:
		plots_per_pdf=total_plots
	
	num_pdfs = (total_plots + plots_per_pdf - 1) // plots_per_pdf
	print(num_pdfs)
	for pdf_idx in range(num_pdfs):
		with PdfPages(f'{output_prefix}_{pdf_idx + 1}.pdf') as pdf:
			start_idx = pdf_idx * plots_per_pdf
			end_idx = min(start_idx + plots_per_pdf, total_plots)
			# fig, axes = plt.subplots(nrows=end_idx - start_idx, ncols=1, figsize=(10 * (end_idx - start_idx), 2))
			fig, axes = plt.subplots(nrows=end_idx - start_idx, ncols=1, figsize=(10, 2 * (end_idx - start_idx)))
			if end_idx - start_idx == 1:
				axes = [axes] 
			for plot_idx, data in enumerate(data_list[start_idx:end_idx]):
				ax = axes[plot_idx]
				total_length = data[0]['length']
				ax.add_patch(patches.Rectangle((0, 0), total_length, body_height, color='lightgray'))
				colors = {'RUVC': 'lightgreen', 'HNH': 'cyan', 'HEPN': 'lightblue'}
				patches_list = [] 
				labels = []       
				for domain_data in data:
					start = domain_data['start']
					end = domain_data['end']
					domain = domain_data['domain']
					reliability = domain_data['reliability']
					name = domain_data['name']
					color = colors.get(domain, 'gray')
					ax.add_patch(patches.Rectangle((start, 0), end - start, body_height, color=color, alpha=reliability))
					
					label = f'{domain}: {start}, {end}, {reliability:.2f}'
					patches_list.append(patches.Patch(color=color,linewidth=0, alpha=reliability))
					labels.append(label)
					
					
				ax.text(0, body_height*2, name, ha='left', va='center', fontsize=10, color='black')	
				ax.text(0, body_height*1.25, 0, ha='right', va='center', fontsize=10, color='black')	
				ax.text(total_length, body_height*1.25, total_length, ha='left', va='center', fontsize=10, color='black')	
				ax.set_xlim(0, total_length)
				ax.set_ylim(0, 2)
				ax.set_yticks([])
				ax.axis('off')
				# ax.set_title(f'Protein Domain Diagram {start_idx + plot_idx + 1}')
				ax.legend(patches_list, labels, loc='center left', bbox_to_anchor=(1.05, body_height/2))
			pdf.savefig(fig, bbox_inches='tight')
			plt.close(fig)




# def process_fasta(fasta_file):
    # # sequences = []
	
    # sequences = set()
    # with open(fasta_file, 'r') as f:
        # sequence = ''
        # for line in f:
            # line = line.strip()
            # if line.startswith('>'):
                # if sequence:
                    # sequences.add(sequence)
                    # sequence = ''
            # else:
                # sequence += line
        # if sequence:  # Add the last sequence
            # sequences.add(sequence)
    # return sequences



def process_fasta(fasta_file):
	# sequences = set()
	fid=''
	sequence2id=defaultdict(str)
	valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')  # Set of valid amino acid characters
	with open(fasta_file, 'r') as f:
		sequence = ''
		for line in f:
			line = line.strip()
			if line.startswith('>'):
				if sequence:
					# sequences.add(sequence)
					sequence2id[sequence]=sequence2id[sequence]+fid
					sequence = ''
				fid=line[1:]
			else:
				# Remove any non-amino acid characters
				sequence += ''.join([char for char in line if char in valid_amino_acids])

		if sequence:  # Add the last sequence
			# sequences.add(sequence)
			sequence2id[sequence]=sequence2id[sequence]+fid+' '
	# print(sequence2id)
	return sequence2id









def predict_and_get_domains(model, sequence, domain_labels=["Non-structural", "RUVC", "HNH", "HEPN"], device='cuda', min_domain_len=25,overlap=2):
	with torch.no_grad():
		predictions, _ = model([sequence], device=device)
	predicted_labels = torch.argmax(predictions, dim=-1).cpu().numpy()
	# print(predictions)
	# Extract domain positions
	domain_positions = []
	domain_positions2 = []
	current_domain = None
	start_pos = 0
	# print(predicted_labels[0])
	for i, label in enumerate(predicted_labels[0]):
		domain_name = domain_labels[label]
		# print(domain_name)
		if domain_name != current_domain:
			if current_domain is not None:
				domain_positions.append({
					"domain": current_domain,
					"start": start_pos + 1,
					"end": i
				})
			current_domain = domain_name
			start_pos = i

	# Add the last domain
	if current_domain is not None:
		domain_positions.append({
			"domain": current_domain,
			"start": start_pos + 1,
			"end": len(predicted_labels[0])
		})

	for domain in domain_positions:
		if domain["end"] == domain["start"]:
			domain["domain"]='Non-structural'

	# Calculate amino acid count for each domain
	for domain in domain_positions:
		if domain["domain"]!='Non-structural':
			domain_positions2.append(domain)
	# print(domain_positions)	
	# print(domain_positions2)	
	if len(domain_positions2) < 1:
		return domain_positions2

	merged_domains = []
	current_domain = domain_positions2[0]   

	for domain in domain_positions2[1:]:             #domain
		 
		if domain['start'] <= current_domain['end'] + overlap and domain['domain'] == current_domain['domain']:
			 
			current_domain['end'] = max(current_domain['end'], domain['end'])
			current_domain['length'] = current_domain['end'] - current_domain['start'] + 1
		else:
			 
			merged_domains.append(current_domain)
			current_domain = domain

	 
	merged_domains.append(current_domain)
	domain_positions=merged_domains

	# print(domain_positions2)
	for domain in domain_positions:
		domain["length"] = domain["end"] - domain["start"] + 1

	# Filter domains based on Z-score threshold
	filtered_domains = [d for d in domain_positions if d["length"] >= min_domain_len]

	return filtered_domains









def predict_and_get_domains(model, sequence, domain_labels=["Non-structural", "RUVC", "HNH", "HEPN"], device='cuda', min_domain_len=25, overlap=2):
    with torch.no_grad():
        predictions, _ = model([sequence], device=device)
    
    # Calculate the confidence (probability)
    probabilities = F.softmax(predictions, dim=-1)  # Softmax to get probabilities
    predicted_labels = torch.argmax(probabilities, dim=-1).cpu().numpy()

    # Initialize lists to hold domain positions and confidences
    domain_positions = []
    domain_positions2 = []
    current_domain = None
    start_pos = 0
    domain_confidence = []  # List to store the confidence for each domain

    # Extract domain positions and confidences
    for i, label in enumerate(predicted_labels[0]):
        domain_name = domain_labels[label]
        confidence = probabilities[0, i, label].item()  # Confidence for the current label
        
        if domain_name != current_domain:
            if current_domain is not None:
                domain_positions.append({
                    "domain": current_domain,
                    "start": start_pos + 1,
                    "end": i,
                    "confidence": domain_confidence  # Add the confidence list to the domain
                })
            current_domain = domain_name
            start_pos = i
            domain_confidence = [confidence]  # Start a new confidence list for the new domain
        else:
            domain_confidence.append(confidence)  # Append confidence to the current domain

    # Add the last domain
    if current_domain is not None:
        domain_positions.append({
            "domain": current_domain,
            "start": start_pos + 1,
            "end": len(predicted_labels[0]),
            "confidence": domain_confidence
        })
    #print(domain_positions)
    # Replace domains with single position length as 'Non-structural'
    for domain in domain_positions:
        if domain["end"] == domain["start"]:
            domain["domain"] = 'Non-structural'

    # Calculate amino acid count for each domain
    for domain in domain_positions:
        if domain["domain"] != 'Non-structural':
            domain_positions2.append(domain)

    if len(domain_positions2) < 1:
        return domain_positions2
    #print(domain_positions2)
    # Merge domains that overlap (or are of the same type)
    merged_domains = []
    current_domain = domain_positions2[0]
    for domain in domain_positions2[1:]:
        if domain['start'] <= current_domain['end'] + overlap and domain['domain'] == current_domain['domain']:
            current_domain['end'] = max(current_domain['end'], domain['end'])
            current_domain['confidence'].extend(domain['confidence'])  # Merge confidence values
        else:
            merged_domains.append(current_domain)
            current_domain = domain

    merged_domains.append(current_domain)
    domain_positions = merged_domains

    for domain in domain_positions:
        domain["length"] = domain["end"] - domain["start"] + 1

    # Filter domains based on length threshold
    filtered_domains = [d for d in domain_positions if d["length"] >= min_domain_len]

    # Add reliability score (mean confidence) for each domain
    for domain in filtered_domains:
        domain["reliability"] = sum(domain["confidence"]) / len(domain["confidence"])

    return filtered_domains




def load_model(model_path, device='cpu'):
	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Model file not found: {model_path}")
	model = torch.load(model_path, map_location=device,weights_only=False)
	return model

def write_domains_to_csv(sequences, output_file, model, domain_labels=["Non-structural", "RUVC", "HNH", "HEPN"], device='cpu', min_domain_len=15, overlap=2, reliability=0.9,output_pdf="output_pdf"):
	with open(output_file, 'w') as f:
		# Write header manually
		f.write('Sequence_ID,fasta_index,Sequence,Domain,Start,End,Domain_Length,Length,Reliability\n')
		# for seq_id, seq in enumerate(sequences):
		seq_id=0
		data_lists=[]
		for seq,f_id in sequences.items():
			seq_id+=1
			print(f"Processing sequence {seq_id}: {f_id[:30]}...")  # Print the first 30 characters as a preview
			domains = predict_and_get_domains(model, seq, domain_labels, device, min_domain_len, overlap)
			# print(domains)
			data_list=[]
			for domain in domains:
				# print(domain)

				if domain['reliability']>=reliability:
					domain['name']=f"seq_{seq_id} {f_id}"
					domain['length']=len(seq)
					f.write(f"seq_{seq_id},{f_id},{seq},{domain['domain']},{domain['start']},{domain['end']},{domain['length']},{domain['length']},{domain['reliability']}\n")
					data_list.append(domain)
			if data_list:
				data_lists.append(data_list)
		# print(data_lists)
	

	plot_multiple_protein_domains(data_lists, output_pdf, plots_per_pdf=20)


def main():
	parser = argparse.ArgumentParser(description="Predict protein domain structures using ESM model.")
	parser.add_argument("model_path", help="Path to the pre-trained model (must exist)")
	parser.add_argument("fasta_file", help="Path to the input FASTA file containing protein sequences")
	parser.add_argument("--use_gpu", action='store_true', help="Flag to use GPU, if available")
	parser.add_argument("--min_domain_len", type=int, default=10, help="Minimum length of the predicted domain")
	parser.add_argument("--overlap", type=int, default=2, help="Allowable overlap between domains")
	parser.add_argument("--reliability", type=float, default=0.9, help="Reliability threshold for domain prediction (default: 0.9)")
	args = parser.parse_args()
	# Check if GPU is available, otherwise default to CPU
	device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
	print(f"Using device: {device}")
	# Load model
	model = load_model(args.model_path, device=device)
	# Process the sequences from the FASTA file
	sequences = process_fasta(args.fasta_file)
	# Output file name based on the input fasta file
	current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

	output_file = f'{os.path.splitext(args.fasta_file)[0]}_{current_time}_domains.csv'
	
	# Predict and write results to CSV
	write_domains_to_csv(sequences, output_file, model, device=device, min_domain_len=args.min_domain_len, overlap=args.overlap, reliability=args.reliability,output_pdf=f'{os.path.splitext(args.fasta_file)[0]}_{current_time}_domains')
	print(f"Results have been written to {output_file}")


if __name__ == "__main__":
    main()

