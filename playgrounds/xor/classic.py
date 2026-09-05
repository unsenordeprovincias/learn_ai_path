from mnist.domain.models import NeuralNet, Layer, Vector
from mnist.domain.functions import mse, sigmoid
from dataclasses import dataclass
from random import choices

@dataclass
class Sample:
    x: Vector
    y_true: Vector

def set_infrastructure():
    l1 = Layer(2, 2, sigmoid)
    loutput = Layer(2, 1, sigmoid)

    nnXOR = NeuralNet([l1, loutput], mse)

    dataset = [
        Sample(x=Vector[0, 0], y_true=Vector[0]),
        Sample(x=Vector[0, 1], y_true=Vector[1]),
        Sample(x=Vector[1, 0], y_true=Vector[1]),
        Sample(x=Vector[1, 1], y_true=Vector[0])
    ]

    return nnXOR, dataset

def train(network: NeuralNet, 
          dataset: list[Sample], 
          n_epochs: int = 5000, 
          learning_rate: float = 0.5, 
          num_registers: int = 10) -> NeuralNet:

    k = n_epochs // num_registers

    for epoch in range(n_epochs):
        loss_epoch = 0
        for sample in (dataset):
            y_pred = network.forward(sample.x)
            loss_epoch += network.floss(y_pred, sample.y_true)
            network.backward(sample.y_true, learning_rate)

        if epoch % k == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:>5}/{n_epochs}  loss={loss_epoch / len(dataset):.6f}")    

    return network

if __name__ == '__main__':
    n_epochs = int(input("Num casos: "))
    network, dataset = set_infrastructure()
    network = train(network, dataset, n_epochs)

    for sample in dataset:
        print(f"{sample.x} -> {network.forward(sample.x)} vs {sample.y_true}")
