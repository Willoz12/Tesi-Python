from graphviz import Digraph
import math
import matplotlib.pyplot as plt
import numpy as np



#plt.plot(np.arange(-5,5,0.2), np.tanh(np.arange(-5,5,0.2))); plt.grid() #squash function, compresa tra -1 e 1
#plt.show()

class Value: #definizione classe Value e tutte le operazioni

    def __init__(self, data, _children=(), _op='', label=''):  #label crea un'etichetta al singolo valore      
        self.data = data
        self.grad = 0.0 #0 significa no effetto, ovvero che ogni valore non ha impatto sull'output, perché se il gradient è 0 significa che cambiare
#la variabile non fa cambiare la funzione di perdita L        
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data})"
      
    def __add__(self, other):
        out =Value(self.data + other.data, (self, other), '+')
        return out
    
    def __mul__(self, other):
        out =Value(self.data * other.data, (self, other), '*')
        return out
    
    def __tanh__(self): #implementiamo con tanh, ovvero [(e^2x)-1]/[(e^2x)+1]
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1) #l'espressione di tanh
        out = Value(t, (self, ), 'tanh') #t ha solo un "children" ed è una tupla composta solo da self, 'tanh' è il nome dell'operazione 
        return out

#####DRAW DOT GRAFICO######
def trace(root): #funzione "helper" che numera tutti i nodi e bordi
#costruisce un set con tutti i nodi e bordi in un grafico    
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child,v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) #LR -> Left to right
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        #per ogni valore nel grafico creami un nodo rettangolare ('record') per esso
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            #se questo è il risultato per un'operazione creami un nodo op per esso
            dot.node(name = uid + n._op, label = n._op)
            #e collegalo ad esso
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        #collega n1 al nodo di n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

#Struttura di un neurone: somma wi*xi + b
#input di x1 e x2, quindi un neurone bidimensionali
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
#i pesi dei neuroni w1 w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
#Bias del neurone
b = Value(6.8813735870195432, label='b') #preso questo bias specifico per avere dei numeri semplici negli output
#Vogliamo fare quindi come detto sopra: x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.__tanh__(); o.label = 'o'#output e funziona ora che ho definito tanh sopra
#di nuovo, non potendolo visualizzare in modo grafico non so se l'output corrisponda, però mi sembra coincida con i risultati del video
#tutto questo quindi è un neurone, che fa parte di una rete neurale con una funzione di perdita L che misura l'accuratezza di questa rete

#INIZIAMO BACKPROPAGATION
o.grad = 1.0 #Derivata di o con sè stesso
n.grad = 0.5 #do/dn. la derivata di una tangente è 1-tanh(n)**2 = 1 - o**2 poiché abbiamo già tanh(n) che è o e ciò viene 0.5
b.grad = 0.5 #come nell'esempio prima la derivata di un'addizione rimane la stessa, quindi sarà sempre 0.5
x1w1x2w2.grad = 0.5 #uguale a b
x1w1.grad = 0.5 #essendo un membro della somma di (x1*w1) + (x2*w2) lo 0.5 rimane anche qua
x2w2.grad = 0.5 #stesso ragionamento di x1w1
x2.grad = 0.5 #per la chain rule sarà w2.data * x2w2.grad, quindi 1.0*0.5 cioè 0.5
w2.grad = 0.0 #per la chain rule sarà x2.data * x2w2.grad, quindi 0.0*0.5 cioè 0. Interessante notare come w2 non abbia alcun peso nell'espressione finale
#perché si va a moltiplicare per x2 che è 0, quindi non cambiando la sua derivata sarà 0
x1.grad = -1.5 #w1.data * x1w1.grad come i calcoli sopra, perciò -3*0.5
w1.grad = 1 #x1.data * x1w1.grad come i calcoli sopra, perciò 2*0.5



print(draw_dot(o))