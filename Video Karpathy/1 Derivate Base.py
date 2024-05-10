import math
import numpy as np
import matplotlib.pyplot as plt

#crea un grafico
#plt.plot([1,2,3,4],[4,7,2,77])
#Visualizza grafico
#plt.show()


#Definizione di Derivata, introduzione:
def f(x):
    return 3*x**2 - 4*x + 5 #potrebbe essere una parabola
print(f(3.0))

xs = np.arange(-5,5,0.25) #valori consecuitivi
print(xs)
ys =f(xs) #metto la funzione definita sopra
print(ys)
plt.plot(xs,ys)
plt.show() #e si ottiene una parabola

#voglio sapere quale sia la derivata di questa funzione in ogni punto di x
#DEF di derivata: limite per h che tende a 0 di di [f(x+h) - f(x)]/h ovvero che:
#se incrementi di pochissimo, ovvero h, la funzione x, come risponde la funzione? Quanto è la pendenza, quanto sale, scende...

h = 0.00001
x = 3.0
print(f(x)) #mi darà 20 come abbiamo visto sopra
print(f(x + h)) #quindi + la h vicina a 0, quindi aggiungendo un numero h in base al grafico visto prima incrementerà di poco
slope = (f(x+h) - f(x))/h #la pendenza viene 14, che è corretto facendo la derivata di f, perché sarebbe 6*3-4, quindi proprio 14
print(slope)

x_minimo = 2/3
print((f(x_minimo + h)-f(x_minimo))/h) #come vediamo la derivata sarà 0