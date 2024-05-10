class Value: #definizione classe Value

    def __init__(self, data): 
        self.data = data

#repr è importante in modo da avere un risultato leggibile    
    def __repr__(self):
        return f"Value(data={self.data})"
    
#attualmente i valori non si possono sommare perché Python non sa come trattare questi elementi "Value", quindi bisogna aggiungerlo con:    
    def __add__(self, other):
        out =Value(self.data + other.data)
        return out
    
# internamente a Python scrivere a+b comporta che a.__add__(b)
#aggiungiamo anche la moltiplicazione
    def __mul__(self, other):
        out =Value(self.data * other.data)
        return out


a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
print(a*b + c) # si legge come (a.__mul__(b)).__add__(c)