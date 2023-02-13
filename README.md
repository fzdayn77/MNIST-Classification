This is a basic Deep Learning project that uses a **CNN** (**C**onvolutional **N**eural **N**etwork), the **[LeNet5](https://www.datasciencecentral.com/lenet-5-a-classic-cnn-architecture/)** architecture, in order to classify the **[MNIST](https://en.wikipedia.org/wiki/MNIST_database)** Dataset.

> Used Framework : **[PyTorch](https://pytorch.org/)**  
> Used Programming Language : **Python3**

## LeNet5 architecture :
- First Convolutional Layer : _(5x5)-Filter_ + _ReLU_
- First Pooling Layer : _Max pooling with (2x2)-Filter_
- Second Convolutional Layer : _(5x5)-Filter_ + _ReLU_
- Second Pooling Layer : _Max pooling with (2x2)-Filter_
- Two Linear Layers + activation function [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
- Output Layer : _Output function [Softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)_
