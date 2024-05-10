from graphviz import Digraph
import math
import matplotlib.pyplot as plt
import numpy as np


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
            self.grad += 1.0 * out.grad #+= Perché in caso di più variabili i gradienti si devono accumulare
            other.grad += 1.0 * out.grad           
        out._backward = _backward 

        return out
    
    def __mul__(self, other): #bisogna farlo anche nella moltiplicazione
        out =Value(self.data * other.data, (self, other), '*')
    
        def _backward():
            self.grad += other.data * out.grad #derivata pezzo successivo (out) * valore altro membro moltiplicazione
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __tanh__(self): #implementiamo con tanh, ovvero [(e^2x)-1]/[(e^2x)+1]
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1) #l'espressione di tanh
        out = Value(t,(self, ), 'tanh') #t ha solo un "children" ed è una tupla composta solo da self, 'tanh' è il nome dell'operazione

        def _backward():
            self.grad += (1 - t**2) * out.grad #t è l'espressione della tangente e 1 - tanh(x)**2 è la definizione di derivata
        out.backward = _backward    
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
        self.grad = 1.0 #o al momento non è definito, ma perché questo è il modello senza esempi ma solo dei comandi
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

a = Value(3.0, label = 'a')
b = a + a; b.label = 'b'
b.backward()
print(draw_dot(b))

