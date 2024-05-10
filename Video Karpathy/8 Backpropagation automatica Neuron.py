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
        self._backward = lambda: None #farà la backpropagation con la chain rule in modo automatico. Di default non fa niente
        self._prev = set(_children)
        self._op = _op
        self.label = label

#repr è importante in modo da avere un risultato leggibile       
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward(): #dobbiamo definire quel che succede quando andiamo a fare la backpropagation per un'addizione.
#bisogna prendere il gradient di "out" e metterlo anche ai grad di self e other, i due dati dell'addizione visto che in derivata si distribuiscono
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad           
        out._backward = _backward 

        return out
    
    def __mul__(self, other): #bisogna farlo anche nella moltiplicazione
        out =Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad = other.data * out.grad #derivata pezzo successivo (out) * valore altro membro moltiplicazione
            other.grad = self.data * out.grad
        out._backward = _backward

        return out
    
    def __tanh__(self): #implementiamo con tanh, ovvero [(e^2x)-1]/[(e^2x)+1]
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1) #l'espressione di tanh
        out = Value(t, (self, ), 'tanh') #t ha solo un "children" ed è una tupla composta solo da self, 'tanh' è il nome dell'operazione

        def _backward():
            self.grad = (1 - t**2) * out.grad #t è l'espressione della tangente e 1 - tanh(x)**2 è la definizione di derivata
        out._backward = _backward

        return out
    
    def backward(self):
        #costruzione di un grafico topologico
        topo = []
        visited = set() #set of visited nodes
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)    #partendo da o si va verso tutti i children tirandoli fuori da destra verso sinistra
#partendo da o, se non è visitato segnalo come visitato e iteralo per tutti i children e costruisce un altro elemento
#che si aggiungerà da solo alla lista dopo che tutti i children sono stati processati
        o.grad = 1.0
        for node in reversed(topo):
            node._backward()

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

#ora bisogna farlo in modo automatico senza calcolare
#Per prima cosa bisogna notare come out.grad è settato a 0, quindi bisogna modificarlo
o.grad = 1.0
o._backward() #questo comando dovrebbe servire a far sì che il gradient si propaghi indietro
n._backward() #porta il gradient a b e x1w1x2w2 visto che è un'addizione
b._backward() #non succede nulla perché essendo un nodo foglia non ha nessun processo sopra come definito a riga 17 (self._backward = lambda: None )
x1w1x2w2._backward() #
x1w1._backward() #
x2w2._backward() #

print(draw_dot(o))

#sarà un problema se ci dovessero essere corrispondenze multiple, come b = a+a e così via, ma vediamo nel file 9