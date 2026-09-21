import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
import numpy as np
from leaf.dataloader import FEMNISTDataset
from torch.utils.data import DataLoader

class Client:
    def __init__(self, dataset, model, device, lr=0.01, batch_size=32):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size)

    def learn(self):
        self.model.train()
        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            self.optimizer.step()

    def send_model(self):
        return self.model.state_dict()

    def receive_model(self, new_state_dict):
        self.model.load_state_dict(new_state_dict)

class Server:
    def __init__(self, num_clients, model, device):
        self.num_clients = num_clients
        self.model = model
        self.device = device
        self.client_models = [model.state_dict() for _ in range(self.num_clients)]

    def calculate_optimized_weights(self, client_models):
        K = len(client_models)
        n = 0
        client_weights_norms = [torch.norm(torch.tensor([torch.norm(w).item() for w in model.values()])).item() for model in client_models]

        S_w = np.sum(client_weights_norms)
        A_w = S_w / K
        SD_w = np.sqrt(np.sum((client_weights_norms - A_w) ** 2) / K)

        wsd = []
        for w_k in client_weights_norms:
            if A_w - SD_w <= w_k <= A_w + SD_w:
                wsd.append(w_k)
                n += 1

        SDA_w = np.sum(wsd) / n
        optimized_weights = np.array(client_weights_norms) / SDA_w
        return optimized_weights

    def federated_averaging(self, clients):
        client_models = [client.send_model() for client in clients]
        optimized_weights = self.calculate_optimized_weights(client_models)

        weighted_avg = client_models[0].copy()
        for key in weighted_avg.keys():
            weighted_avg[key] *= 0

        for i, model in enumerate(client_models):
            for key in model.keys():
                weighted_avg[key] += optimized_weights[i] * model[key]

        for client in clients:
            client.receive_model(weighted_avg)

    def evaluate(self, clients):
        self.model.eval()
        correct = 0
        total = 0
        for client in clients:
            for data, target in client.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        return correct / total


def load_femnist_data(num_clients):
    femnist_data = FEMNISTDataset("data", num_clients)
    return [DataLoader(femnist_data.clients[i], batch_size=32, shuffle=True) for i in range(num_clients)]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_clients = 100
    num_rounds = 200
    performance_interval = 10

    # Load FEMNIST data
    client_dataloaders = load_femnist_data(num_clients)

    # Define the model architecture
    # Replace GraphNeuralNetwork with the appropriate model for FEMNIST, e.g., CNN or MLP
    model = CNNModel().to(device)

    # Initialize clients
    clients = [Client(client_dataloaders[i], model, device) for i in range(num_clients)]

    # Initialize the server
    server = Server(num_clients, model, device, num_communities)

    # Federated learning process
    for round in range(1, num_rounds + 1):
        print(f"Round {round}")

        # Clients learn locally
        for client in clients:
            client.learn()

        # Output performance every performance_interval rounds
        if round % performance_interval == 0:
            accuracy = server.evaluate(clients)
            print(f"Performance at round {round}: {accuracy:.4f}")

if __name__ == "__main__":
    main()
