
install micrograd

from micrograd.engine import Value

#tecnicamente dovrei fare a = Value(-4.0) e b = Value(2.0), ovvero trasformare questi in un valore oggetto chiamato "Value".
#Non potendo importare il pacchetto non so come impostarlo. I numeri dovrebbero comunque ricondurre a sé stessi
#Costruiamo un'equazione dove 'a' e 'b' sono trasformati in c,d,e...
#Micrograd supporta diversi tipi di operazioni:

#Sommare e moltiplicare valori, elevarli a potenza, operatore di incremento (+=), porre in negativo (-a), dividere per una costante (f/2)

#L'idea è di creare un'espressione che, dati i valori 'a' e 'b', restituisca un output 'g' che in background Micrograd costruirà questa espressione

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu() #"Squash at Zero" detto da Karpathy
#ReLU accelera la convergenza della discesa del gradiente verso il minimo globale
#della funzione di perdita grazie alla sua proprietà lineare e non saturante.
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f/2.0
g += 10.0/f
print(f'{g.data:.4f}') #Viene vuori essere 24.7041 #mi dice però che non può farlo perché non ho importato il pacchetto
#g.backward() #Parte da 'g' e poi inizia ad andare indietro e applica la catena di calcolo. Quindi valuta la derivata di g rispetto ai nodi interni, come 'c' e 'd',
#ma anche rispetto agli input come 'a' e 'b'. Ovviamente non mi funziona, in questo caso perché "'float' object has no attribute 'backward'"
print(f'{a.grad:.4f}') #viene 138.8338, ovvero la derivata di dg/da
print(f'{b.grad:.4f}') #viene 645.5773, ovvero la derivata di dg/db






