
def calculate_precision_recall(true_positions, predicted_positions):
	predicted_aa = []
	true_aa = []
	for start, end in predicted_positions:
		predicted_aa.extend(range(start, end + 1))   
	for start, end in true_positions:
		true_aa.extend(range(start, end + 1))   
	true_positive = len(set(predicted_aa).intersection(true_aa))   
	# print(true_positive)
	# print(true_aa)
	# print(len(true_aa))
	# print(len(predicted_aa))
	precision = true_positive / len(predicted_aa) if len(predicted_aa) > 0 else 0
	recall = true_positive / len(true_aa) if len(true_aa) > 0 else 0
	print(f"Precision: {precision}\nRecall: {recall}")	
	# return precision, recall



true_positions = [(457, 563), (647, 705), (742, 767)]   
predicted_positions = [(498, 555), (611, 707), (744, 762)]   

calculate_precision_recall(true_positions, predicted_positions)



true_positions = [(636, 685), (888, 942), (1072, 1093)]   
predicted_positions = [(640, 692), (847, 945), (1070,1089)]   

calculate_precision_recall(true_positions, predicted_positions)



true_positions = [(363, 436), (578, 666), (694, 757)]   
predicted_positions = [(375, 435), (556, 663), (694, 710)]   

calculate_precision_recall(true_positions, predicted_positions)



true_positions = [(1, 62), (718, 764), (765, 924), (925,1102)]   
predicted_positions = [(1, 63), (724, 769), (770, 919), (920, 1097)]   

calculate_precision_recall(true_positions, predicted_positions)



true_positions = [(1, 38), (271, 314), (347, 399), (494,548)]   
predicted_positions = [(1, 45), (282, 319), (320, 447), (459, 562)]   

calculate_precision_recall(true_positions, predicted_positions)

