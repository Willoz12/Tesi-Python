from graphviz import Digraph

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

#facciamo la backpropagation manualmente

L.grad = 1.0
d.grad = -2.0
f.grad =  4.0
#derivare L secondo c, quindi dL/dc. Per farlo dobbiamo prima considerare l'impatto che ha c su d visto che sappiamo d su L
#Sapendo che d = c + e, la derivata in c o in e è sempre = 1 visto che è composta da solo un'addizione, quindi l'aggiunta è proporzionale
#dd/dc = dd/de = 1.0. Quindi le derivate locali sono queste, ma vogliamo la derivata di L su c, quindi come si fa a mettere insieme le informazioni?
#Si fa con la chain rule:
#Se una variabile 'z' dipende da una 'y' che dipende a sua volta da una 'x', 'z' dipende da 'x' a sua volta, la chain rule è:
#volendo dz/dx = dz/dy * dy/dx, perciò noi vogliamo dL/dc ma ABBIAMO dL/dd e dd/dc, perciò
#dL/dc = (dL/dd)*(dd/dc) = dL/dd visto che dd/dc = 1
c.grad = -2.0
e.grad = -2.0 #visto che è aggiunto a c la situazione non cambia
#ora mi mancano 'b' e 'a'. Si procede come prima:
#ora so che dL/de = 2.0 come calcolato sopra
#VOGLIO dL/da = (dL/de) * (de/da)
#quindi ottengo dL/da = -2 * (-3)
a.grad = 6
b.grad = -4
#in sostanza quindi backpropagation è un'applicazione ricorsiva della chain rule



print(draw_dot(L)) #PROBLEMA: NON MI DA IL GRAFICO IN CONSOLE. O MEGLIO, ME LO DA' MA NON ME LO VISUALIZZA COME GRAFICO, COME SE GRAPHVIZ NON FUNZIONASSE



