import streamlit as st
import re,datetime,base64
from collections import defaultdict
import os
 
from uuid import uuid4
 
import torch,argparse,os,datetime,re
import torch.nn as nn
 
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

		# print(f'embed_dim: {esm_model.embed_dim}\nfreeze_layers: {freeze_layers}\nnum_layers: {self.num_layers}')

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
		tokenized_batch = tokenized_batch[:, 1:-1]  # del <cls> ?? <eos>
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


def plot_multiple_protein_domains(data_list, output_prefix='protein_domain_diagram', plots_per_pdf=2,body_height=0.5,domain_names = ['RUVC', 'HNH', 'HEPN']):
	output_pdfs=[]
	total_plots = len(data_list)
	if total_plots<plots_per_pdf:
		plots_per_pdf=total_plots
	
	date2csv=open(f'{output_prefix}.csv','w')
	date2fasta=open(f'{output_prefix}.fasta','w')
	
	seq_unique_set=set()
	
	date2csv.write('Sequence_ID,fasta_index,Sequence,Domain,Start,End,Domain_Length,Length,Reliability\n')
	 
	
	colors = ["#76C893", "#4DD0E1", "#FFB4B4", "#D3D3D3", "#F4A261", "#A78BFA", "#B2DF8A", "#FDB462"]
	
	# domain_names = ['RUVC', 'HNH', 'HEPN']  # ??????????
	
	rcolors = {name.upper(): colors[i % len(colors)] for i, name in enumerate(domain_names)}

	
	# domain['info'] domain_data['name'] domain['seq']
	
	
	num_pdfs = (total_plots + plots_per_pdf - 1) // plots_per_pdf
	# print(num_pdfs)
	for pdf_idx in range(num_pdfs):
		
		with PdfPages(f'{output_prefix}_{pdf_idx + 1}.pdf') as pdf:
			output_pdfs.append(f'{output_prefix}_{pdf_idx + 1}.pdf')
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
				# colors = {'RUVC': 'lightgreen', 'HNH': 'cyan', 'HEPN': 'lightblue'}
				patches_list = [] 
				labels = []       
				for domain_data in data:
					start = domain_data['start']
					end = domain_data['end']
					domain = domain_data['domain']
					reliability = domain_data['reliability']
					name = domain_data['name']
					if domain_data['seq'] not in seq_unique_set:
						seq_unique_set.add(domain_data['seq'])
						date2fasta.write(f">{name}\n{domain_data['seq']}\n")
						
					color = rcolors.get(domain, 'gray')
					# name2=re.sub(r"\s+", ",", name)
					date2csv.write(f"{domain_data['info']}\n")
					
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


	date2csv.close()
	date2fasta.close()
	return output_pdfs
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

def write_domains_to_csv(sequences,fasta_path, output_file, model, domain_labels=["Non-structural", "RUVC", "HNH", "HEPN"], device='cpu', min_domain_len=15, overlap=2, reliability=0.9,output_pdf="output_pdf"):
	data_lists=[]
	data_lists2=[]
	data_lists3=[]
	with open(output_file, 'w') as f:
		# Write header manually
		f.write('Sequence_ID,fasta_index,Sequence,Domain,Start,End,Domain_Length,Length,Reliability\n')
		# for seq_id, seq in enumerate(sequences):
		seq_id=0
		data_lists=[]
		data_lists2=[]
		data_lists3=[]
		for seq,f_id in sequences.items():
			seq_id+=1
			# print(f"Processing sequence {seq_id}: {f_id[:30]}...")  # Print the first 30 characters as a preview
			domains = predict_and_get_domains(model, seq, domain_labels, device, min_domain_len, overlap)
			# print(domains)
			data_list=[]
			for domain in domains:
				# print(domain)

				if domain['reliability']>=reliability:
					domain['name']=f"seq_{seq_id} {f_id}"
					domain['length']=len(seq)
					domain['seq']=seq
					domain['info']=f"seq_{seq_id},{f_id},{seq},{domain['domain']},{domain['start']},{domain['end']},{domain['end']-domain['start']+1},{domain['length']},{domain['reliability']}"
					f.write(f"{domain['info']}\n")
	
					data_list.append(domain)
			if data_list:
				data_lists.append(data_list)

			if len(data_list)>1:
				data_lists2.append(data_list)
			if len(data_list)>2:
				data_lists3.append(data_list)


		# print(data_lists)
	
	output_pdf=f'{fasta_path}/{output_pdf}'
	output_pdfs1=False
	if data_lists:
		output_pdfs1=plot_multiple_protein_domains(data_lists, f'{output_pdf}_ones_domain', plots_per_pdf=20,domain_names=domain_labels[1:])
	# plot_multiple_protein_domains(data_lists2, f'{output_pdf}_double_domain', plots_per_pdf=20,domain_names=domain_labels[1:])
	# plot_multiple_protein_domains(data_lists3, f'{output_pdf}_triple_domain', plots_per_pdf=20,domain_names=domain_labels[1:])
	return output_pdfs1
	
	
	
	
	


def predict_domains_from_fasta(
	fasta_path: str,
	model_path: str='/home/miniconda3/paper/esmcascas_web/models/ESMCasDomainNet_v0.1.pth',
	fasta_file: str='cleaned.fasta',
	use_gpu: bool = True,
	min_domain_len: int = 10,
	overlap: int = 2,
	reliability: float = 0.9
):
	"""
	Predict domains from a FASTA file using a pre-trained model.

	Parameters:
		model_path (str): Path to the pre-trained .pth model.
		fasta_file (str): Path to the input FASTA file.
		use_gpu (bool): Whether to use GPU if available.
		min_domain_len (int): Minimum domain length.
		overlap (int): Overlap threshold for domain merging.
		reliability (float): Minimum reliability for domain inclusion.
	"""

	# fasta_file=f'{fasta_path}/{fasta_file}'

	device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
	# print(f"[INFO] Using device: {device}")

	model = load_model(model_path, device=device)
	# os.chdir(fasta_path)	
	sequences = process_fasta(f'{fasta_path}/{fasta_file}')

	current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	base = os.path.splitext(os.path.basename(fasta_file))[0]
	output_prefix = f"{base}_{current_time}_domains"
	output_csv = f"{fasta_path}/{output_prefix}.csv"

	opt_pdfs1=write_domains_to_csv(
		sequences,
		fasta_path,
		output_file=output_csv,
		model=model,
		device=device,
		min_domain_len=min_domain_len,
		overlap=overlap,
		reliability=reliability,
		output_pdf=output_prefix
	)

	# print(f"[DONE] Results saved to: {output_csv}")
	# print(f"[DONE] Results saved to: {opt_pdfs1}")
	
	# os.chdir('..')
	
	if not opt_pdfs1:
		return False,False
	
	
	return (opt_pdfs1[0],output_csv)




st.set_page_config(page_title="ESMCasDomainNet", layout="centered")
st.title("ESMCasDomainNet: CRISPR-Cas Domain Annotation Web Tool")

# ---- Section 1: Intro ----
st.markdown(
    """
	[GitHub Repository](https://github.com/wqiudao/ESM-CasDomainNet) https://github.com/wqiudao/ESM-CasDomainNet
    
	Paste your FASTA-formatted sequences below.  
    - Maximum: **20 unique sequences**    
    """
)


DEFAULT_FASTA = ">example1\nMWYASLMSAHHLRVGIDVGTHSVGLATLRVDDHGTPIELLSALSHIHDSGVGKEGKKDHDTRKKLSGIARRARRLLHHRRTQLQQLDEVLRDLGFPIPTPGEFLDLNEQTDPYRVWRVRARLVEEKLPEELRGPAISMAVRHIARHRGWRNPYSKVESLLSPAEESPFMKALRERILATTGEVLDDGITPGQAMAQVALTHNISMRGPEGILGKLHQSDNANEIRKICARQGVSPDVCKQLLRAVFKADSPRGSAVSRVAPDPLPGQGSFRRAPKCDPEFQRFRIISIVANLRISETKGENRPLTADERRHVVTFLTEDSQADLTWVDVAEKLGVHRRDLRGTAVHTDDGERSAARPPIDATDRIMRQTKISSLKTWWEEADSEQRGAMIRYLYEDPTDSECAEIIAELPEEDQAKLDSLHLPAGRAAYSRESLTALSDHMLATTDDLHEARKRLFGVDDSWAPPAEAINAPVGNPSVDRTLKIVGRYLSAVESMWGTPEVIHVEHVRDGFTSERMADERDKANRRRYNDNQEAMKKIQRDYGKEGYISRGDIVRLDALELQGCACLYCGTTIGYHTCQLDHIVPQAGPGSNNRRGNLVAVCERCNRSKSNTPFAVWAQKCGIPHVGVKEAIGRVRGWRKQTPNTSSEDLTRLKKEVIARLRRTQEDPEIDERSMESVAWMANELHHRIAAAYPETTVMVYRGSITAAARKAAGIDSRINLIGEKGRKDRIDRRHHAVDASVVALMEASVAKTLAERSSLRGEQRLTGKEQTWKQYTGSTVGAREHFEMWRGHMLHLTELFNERLAEDKVYVTQNIRLRLSDGNAHTVNPSKLVSHRLGDGLTVQQIDRACTPALWCALTREKDFDEKNGLPAREDRAIRVHGHEIKSSDYIQVFSKRKKTDSDRDETPFGAIAVRGGFVEIGPSIHHARIYRVEGKKPVYAMLRVFTHDLLSQRHGDLFSAVIPPQSISMRCAEPKLRKAITTGNATYLGWVVVGDELEINVDSFTKYAIGRFLEDFPNTTRWRICGYDTNSKLTLKPIVLAAEGLENPSSAVNEIVELKGWRVAINVLTKVHPTVVRRDALGRPRYSSRSNLPTSWTIE\n>example1\nMDMVYVLNKDGKPLMATTRGGRVRYLLKEKKARVVSSTPFTIQLNYDTPDATQDLILGIDPGRTNIGVAVVKEDGQCVFSAHLETRNKEVPLLMKKRAAFRRQHRTQDRRRKRQRRAIAAGTTVESNTIERLLPGYEKPIVCHHIRNKEARFNNRSRPAGWLTPTANHLLQAHINLIAKAAKFLPITKVVVELNRFAFMAMDNPNIRRWEYQQGPLYGLGSVKDAVYAQQDGHCLFCKKPIDHYHHVVPRHKGGSETLANRCGLCEKHHALVHKDKAWAEKLVTRKGGMNKKYHALSVLNQIIPHLMEYIGSETLYDAYATDGRSTKGFRIAKNVPKEHYTDAYCIACSILDADTKVSTPVEPFELKQFRRHDRQACIRQMVDRKYLLHGKVVAANRHKAIEQKSDSLEEFREAHGNAAVSQLTVKPHHSQCKDMARIMPGAMMDFDGIVGVFKGSSGRNNGTPNYYNSTKGERFLTRSCVLLAQNAGMVFIPA"






st.markdown("""
    <style>
    div.stButton > button {
        padding: 6px 28px;
        font-size: 20px;
		font-weight: bold; 
        border-radius: 8px;
        color: white;
        background-color: #4CAF50;  /* Green for Run */
        border: none;
        margin-right: 20px;
    }
    div.stButton > button:hover {
        background-color: #45a049;
    }

    /* Reset button (assumes it's the second button rendered) */
    div.stButton:nth-child(2) > button {
        background-color: #f44336;  /* Red for Reset */
    }
    div.stButton:nth-child(2) > button:hover {
        background-color: #da190b;
    }
    </style>
""", unsafe_allow_html=True)




# ---- Section 2: Textarea Input ----
fasta_input = st.text_area(
    "Paste FASTA sequences here:",
    value=DEFAULT_FASTA,
    height=200,
    key="fasta_textarea"
)

# ---- Section 3: Buttons ----
col1, col2 = st.columns([1, 1])
run_clicked = col1.button("Run")
reset_clicked = col2.button("Reset")

if reset_clicked:
    st.session_state.fasta_input = DEFAULT_FASTA

# ---- FASTA Parsing + Cleaning ----
def parse_and_clean_fasta(fasta_text):
    sequences = defaultdict(list)
    current_id = None
    current_seq = []
    id_counter = defaultdict(int)

    for line in fasta_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id and current_seq:
                seq_str = ''.join(current_seq).upper()
                sequences[seq_str].append(current_id)
            raw_id = line[1:].strip()
            base_id = re.sub(r"[^\w]", "_", raw_id)
            id_counter[base_id] += 1
            if id_counter[base_id] == 1:
                final_id = base_id
            else:
                final_id = f"{base_id}_{id_counter[base_id]}"
            current_id = final_id
            current_seq = []
        else:
            current_seq.append(line)

    if current_id and current_seq:
        seq_str = ''.join(current_seq).upper()
        sequences[seq_str].append(current_id)

    return sequences

# ---- Save to cleaned.fasta ----
def save_to_fasta(sequences_dict):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    uid = str(uuid4())[:8]
    output_dir = f"output_{now}_{uid}"
    os.makedirs(output_dir, exist_ok=True)
    fasta_path = os.path.join(output_dir, "cleaned.fasta")

    with open(fasta_path, "w") as f:
        for seq, id_list in sequences_dict.items():
            for sid in id_list:
                f.write(f">{sid}\n{seq}\n")

    return fasta_path, output_dir


def load_model(model_path, device='cpu'):
	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Model file not found: {model_path}")
	model = torch.load(model_path, map_location=device,weights_only=False)
	return model

 
def get_download_button(file_path, button_label="Download File"):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    file_name = os.path.basename(file_path)
    href = f'''
        <a href="data:application/octet-stream;base64,{b64}" download="{file_name}">
            <button style='padding:10px;font-size:16px;'>{button_label}</button>
        </a>
    '''
    return href
 
def get_pdf_preview_html(file_path):
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()
    b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{b64_pdf}"
            width="700"
            height="500"
            type="application/pdf"
        ></iframe>
    """
    return pdf_display

 
# ---- Run Logic ----
if run_clicked and fasta_input.strip():
	parsed = parse_and_clean_fasta(fasta_input)
	num_seqs = len(parsed)

	if num_seqs > 20:
		st.error(f"You submitted {num_seqs} unique sequences. Maximum allowed is 20.")
	elif num_seqs == 0:
		st.warning("No valid sequences found.")
	else:
		fasta_path, out_dir = save_to_fasta(parsed)

		with st.spinner("Running domain prediction..."):
			pdf_file, csv_file = predict_domains_from_fasta(out_dir)

		# st.success(f"Prediction completed.")
		if pdf_file:
			# ---- Move files to static dir ----
			static_dir = "static"
			os.makedirs(static_dir, exist_ok=True)
			pdf_name = os.path.basename(pdf_file)
			csv_name = os.path.basename(csv_file)
			
			
			pdf_name = f'{out_dir}_{pdf_name}'
			csv_name =f'{out_dir}_{csv_name}' 
			
			
			static_pdf = os.path.join(static_dir, pdf_name)
			static_csv = os.path.join(static_dir, csv_name)

			import shutil
			shutil.copy(pdf_file, static_pdf)
			shutil.copy(csv_file, static_csv)
			# st.success(f"Prediction completed.\n{pdf_name}\n{csv_name}")
			# ---- Download buttons (HTML) ----
			st.markdown("### Download Results")

		 
			st.markdown(get_download_button(f"static/{pdf_name}", "Download PDF")+get_download_button(f"static/{csv_name}", "Download CSV"), unsafe_allow_html=True)

			# ---- Inline PDF Preview ----

			st.subheader("Domain Annotation PDF Preview")
			st.markdown(get_pdf_preview_html(f"static/{pdf_name}"), unsafe_allow_html=True)
		else:
			st.markdown("###No Valid Domains Detected")
			st.markdown(
				"""
				We could not identify any valid protein domains from your input.  
				Please make sure the sequence is correct and in FASTA format.  
				Try using a longer or different sequence for better results.
				"""
			)
