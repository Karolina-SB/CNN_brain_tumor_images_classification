import csv
import glob


f="./balanced_test_labels.csv"
#f=open("./balenced_testing_labels.csv",'w')
#f=open("./unbalenced_training_labels.csv",'w')
#f=open("./unbalenced_testing_labels.csv",'w')


path = "C:/Dane/balanced_kfold_test/healthy/*.*"
h=glob.glob(path)
path = "C:/Dane/balanced_kfold_test/tumor/*.*"
t=glob.glob(path)
with open(f, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "label"])
    for element in h:
        writer.writerow([element, "healthy"])
    for element in t:
        writer.writerow([element, "tumor"])

