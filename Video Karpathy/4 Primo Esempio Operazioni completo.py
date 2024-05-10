class Value: #definizione classe Value

    def __init__(self, data, _children=(), _op=''): #a differenza del 3 aggiungo una variabile children che sarà una tupla vuota. Una Tupla è una sequenza immutabile di valori tipo una lista ma non modificabile
#aggiungo anche _op        
        self.data = data
        self._prev = set(_children)
        self._op = _op

#repr è importante in modo da avere un risultato leggibile    
    def __repr__(self):
        return f"Value(data={self.data})"
    
#attualmente non si possono sommare perché Python non sa come trattare questi elementi "Value", quindi bisogna aggiungerlo con:    
    def __add__(self, other):
        out =Value(self.data + other.data, (self, other), '+')
        return out
    
# internamente a Python scrivere a+b comporta che a.__add__(b)
#aggiungiamo anche la moltiplicazione
    def __mul__(self, other):
        out =Value(self.data * other.data, (self, other), '*')
        return out


a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a*b + c
print(d) # si legge come (a.__mul__(b)).__add__(c)
print(d._prev) #questo mi dice i "children" di d, ovvero a*b e c
print(d._op) #mi restituisce il segno, + nell'addizione e * nella moltiplicazione. In questo caso mi darà + perché 'd' è formato dall'addizione di 2 valori