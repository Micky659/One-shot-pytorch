import os

notin = []

class Trainset:

    def __init__(self, path):
        self.path = path
        self.images = []
        self.count = 1

        for subFolder in os.listdir(self.path):
            print("-----------------------------------------------------------------------------------------------")
            print(subFolder)
            if subFolder in ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0012", "0014"]:
                continue
            for employee in os.listdir(os.path.join(self.path, subFolder)):
                notin.append(employee)
                if employee == "0013_0002001":
                    break

Trainset('/trainset')
print(notin)
