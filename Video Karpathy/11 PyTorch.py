import torch

 #Torch si basa su Tensors che sono array n-dimensionali di scalari
d = torch.Tensor([[1,2,3], [4,5,6]])
print(d)


x1 = torch.Tensor([2.0]).double() ; x1.requires_grad = True #.double perchè così casto gli elementi nello steso, ossia float32 che con double diventa float64
x2 = torch.Tensor([0.0]).double() ; x2.requires_grad = True #essendo poi foglie PyTorch dà per scontato che non abbiano bisogno di gratdienti, quindi bisogna specificarlo
w1 = torch.Tensor([-3.0]).double() ; w1.requires_grad = True
w2 = torch.Tensor([1.0]).double() ; w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double() ; b.requires_grad = True
n = x1*w1 + x2*w2 + b #neuron
o = torch.tanh(n)

print(o.data.item()) #si usa .item prende un singolo Tensor di un elemento e lo restituisce togliendo il Tensor
o.backward()

#o.item e o.data.item producono lo stesso risultato in PyTorch

print("----------")
print("x2", x2.grad.item())
print("w2", w2.grad.item())
print("x1", x1.grad.item())
print("w1", w1.grad.item())
#e ottengo gli stessi risultati dei punti precedenti
#pythorch può praticamente fare le cose simili che abbiamo sviluppato con Micrograd