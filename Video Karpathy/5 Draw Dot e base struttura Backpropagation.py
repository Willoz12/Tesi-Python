from graphviz import Digraph
import matplotlib.pyplot as plt 

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

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label = 'e'
f = Value(-2.0, label='f')
d = e + c; d.label = 'd'
L = d*f; L.label = 'L'#L è l'output finale stavolta
print(L)

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

print(draw_dot(L)) #PROBLEMA: NON MI DA IL GRAFICO IN CONSOLE. O MEGLIO, ME LO DA' MA NON ME LO VISUALIZZA COME GRAFICO, COME SE GRAPHVIZ NON FUNZIONASSE

#Ora servirà fare la "backpropagation", quindi partire da L e arrivare ad a e b. Calcolare le derivate di L rispetto a f, d, c...
#Quindi servirà calcolare che peso avranno le variabili a,b,c ed e sulla funzione di perdita L
#tutti i "grad" al momento sono 0, nel file 6 saranno completi
#grad indica la derivata dell'output rispetto a quel valore, quindi la derivata di L rispetto ad ogni singolo elemento:
# la derivata di L rispetto a L sarà 1 perché ogni cambio h di L, L essendo sé stesso è proporzionale
#possiamo calcolari o stimarli