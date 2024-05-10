import math
import numpy as np
import matplotlib.pyplot as plt

a = 2.0
b = -3.0
c = 10.0
d = a*b + c
print(d)

h = 0.00001 #di nuovo prendo h vicino a 0

#inputs
a = 2.0
b = -3.0
c = 10.0
#vorrò quindi la derivata di d in base ad a,b, e c
d1 = a*b + c
a +=h
d2 = a*b + c
slope = (d2-d1)/h #quindi la differenza tra d1 e d2 che la variazione di a, ossia h

print("d1",d1)
print("d2",d2)
print("slope", slope) #ottengo b come risultato perché differenziando otterrei appunto il risultato di b, stesso discorso
#ovviamente se mettessi b+=h invece di a+=h perché differenziando avrei a per lo stesso motivo inverso
#discorso diverso per c perché essendo esterno in un'addizione, la sua pendenza sarà il valore di h

