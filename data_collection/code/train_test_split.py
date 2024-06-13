import random
cat_test = ['Toilet', 'Scissors','Table', 'Stapler','USB','WashingMachine', 'Oven','Faucet', 'Phone','Kettle','Window']
cat_train = ['Safe', 'Door','Display','Refrigerator' ,'Laptop','Lighter','Microwave','Mouse','Box','TrashCan','KitchenPot','Suitcase','Pliers','StorageFurniture','Remote','Bottle'
    , 'FoldingChair','Toaster','Lamp','Dispenser', 'Cart', 'Globe','Eyeglasses','Pen','Switch','Printer','Keyboard','Fan','Knife','Dishwaher']
file_path = '../stats/train_46cats_all_data_list.txt'


test_lines = []
with open(file_path, 'r') as file:
    for line in file:
        cleaned_line = line.strip()
        for cat in cat_test:
            if cat in cleaned_line:
                test_lines.append(cleaned_line)
                break
train_lines = []
with open(file_path, 'r') as file:
    for line in file:
        cleaned_line = line.strip()
        for cat in cat_train:
            if cat in cleaned_line:
                train_lines.append(cleaned_line)
                break
random.shuffle(train_lines)
length = len(train_lines)


train_lines_output = train_lines[:int((4*length)/5)]
test_lines.extend(train_lines[int((4*length)/5):])

file_path1 = '../stats/test_id.txt'

with open(file_path1, 'w') as file:
    for item in test_lines:
        file.write(f"{item}\n")

file_path2 = '../stats/train_id.txt'
with open(file_path2, 'w') as file:
    for item in train_lines_output:
        file.write(f"{item}\n")

