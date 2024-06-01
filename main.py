import matplotlib.pyplot as plt
import numpy as np
import Insideout
import os
from scipy.stats import pearsonr
import Turbulence
import dataLoader
import FC
import BOLDFilters
import p_values
import phFCD
import swFCD
import PhaseInteractionMatrix
BOLDFilters.TR = 3
BOLDFilters.flp = 0.01
BOLDFilters.fhi = 0.09

t=dataLoader.computeAvgSC_HC_Matrix(dataLoader.getClassifications(), dataLoader.base_folder)
t=dataLoader.correctSC(t)
print(dataLoader.characterizeConnectivityMatrix(t))
plt.imshow(t)
plt.show()
NPARCELLS = 360




def calculate_Tauwinner(FowRev):
    max_means = []

    # Iterar sobre els vectorss de FowRev
    for vector in FowRev:
        max_means=np.max(np.mean(vector))

    # Calcular el Tauwinner com l'index del maxim valor mitja
    Tauwinner = np.argmax(max_means)

    return Tauwinner








# diccionari per guardar fowrev de cada subjecte, i declaració de variables
Fr = {}
Fr2 = {}
subject_list = []
listafowrevs=[]
index2 = []
rspatime_lista = []
rspa_lista = []
rspatime_su_lista = []
rspa_su_lista = []
Rtime_lista = []
Rtime_su_lista = []
acfspa_lista = []
acfspa_su_lista = []
acftime_lista = []
acftime_su_lista = []

P=dataLoader.checkClassifications(subject_list,fileName="/subjects.csv")


# Ruta de la carpeta principal
carpeta_principal = dataLoader.base_folder+"/DADES fMRI"+"/fMRI"

for indice, carpeta in enumerate(os.listdir(carpeta_principal)):
    if indice >= 37:
        break
    SC,Abeta,Tau,TimeSeries = dataLoader.loadSubjectData(str(carpeta), correcSCMatrix=True, normalizeBurden=True)
    listafowrevs.append(Insideout.from_fMRI(TimeSeries)[0])
    Fr[str(carpeta)], _ , _ = Insideout.from_fMRI(TimeSeries)
    NOVA = TimeSeries[:360, :197]  # Retall
    Fr2[str(carpeta)] = Turbulence.from_fMRI(NOVA,applyFilters=False)[0]
    Rspatime,Rspatime_su,Rspa,Rtime,Rspa_su,Rtime_su,acfspa,acftime,acfspa_su,acftime_su=Turbulence.from_fMRI(NOVA,applyFilters=False)
    rspatime_lista.append(Rspatime)
    rspatime_su_lista.append(Rspatime_su)
    rspa_lista.append(Rspa)
    rspa_su_lista.append(Rspa_su)
    Rtime_lista.append(Rtime)
    Rtime_su_lista.append(Rtime_su)
    acfspa_lista.append(acfspa)
    acfspa_su_lista.append(acfspa_su)
    acftime_lista.append(acftime)
    acftime_su_lista.append(acftime_su)


########SHADE ERROR BARS##########
def calculate_stats(data):
    means = np.nanmean(data, axis=0)
    stds = np.nanstd(data, axis=0)
    return means, stds

# Calcular mitjana i desviació estandar per Rspa y Rspa_su
mean_Rspa, std_Rspa = calculate_stats(rspa_lista)
mean_Rspa_su, std_Rspa_su = calculate_stats(rspa_su_lista)



# Plot RSPA
plt.figure(1)
plt.plot(np.arange(0, NPARCELLS), mean_Rspa, '-r')
plt.fill_between(np.arange(0, NPARCELLS), mean_Rspa - std_Rspa, mean_Rspa + std_Rspa,
                 color='r', alpha=0.7)
plt.plot(np.arange(0, NPARCELLS), mean_Rspa_su, '-k')
plt.fill_between(np.arange(0, NPARCELLS), mean_Rspa_su - std_Rspa_su, mean_Rspa_su + std_Rspa_su,
                 color='k', alpha=0.7)
plt.xlabel('Index')
plt.ylabel('Rspa')
plt.title('Rspa')
plt.show()




mean_Rtime, std_Rtime = calculate_stats(Rtime_lista)
mean_Rtime_su, std_Rtime_su = calculate_stats(Rtime_su_lista)
# Plot Rtime
plt.figure(2)
plt.plot(np.arange(0, 197), mean_Rtime, '-r')
plt.fill_between(np.arange(0, 197), mean_Rtime - std_Rtime, mean_Rtime + std_Rtime,
                 color='r', alpha=0.7)
plt.plot(np.arange(0, 197), mean_Rtime_su, '-k')
plt.fill_between(np.arange(0, 197), mean_Rtime_su - std_Rtime_su, mean_Rtime_su + std_Rtime_su,
                 color='k', alpha=0.7)
plt.xlabel('Index')
plt.ylabel('Rtime')
plt.title('Rtime')
plt.show()





mean_acfspa, std_acfspa = calculate_stats(acfspa_lista)
mean_acfspa_su, std_acfspa_su = calculate_stats(acfspa_su_lista)
# Plot acfspa
plt.figure(3)
plt.plot(np.arange(0, 360), mean_acfspa, '-r')
plt.fill_between(np.arange(0, 360), mean_acfspa - std_acfspa, mean_acfspa + std_acfspa,
                 color='r', alpha=0.7)
plt.plot(np.arange(0, 360), mean_acfspa_su, '-k')
plt.fill_between(np.arange(0, 360), mean_acfspa_su - std_acfspa_su, mean_acfspa_su + std_acfspa_su,
                 color='k', alpha=0.7)
plt.xlabel('Index')
plt.ylabel('acfspa')
plt.title('acfspa')
plt.show()






mean_acftime, std_acftime = calculate_stats(acftime_lista)
mean_acftime_su, std_acftime_su = calculate_stats(acftime_su_lista)
# Plot acftime
plt.figure(4)
plt.plot(np.arange(0, 197), mean_acftime, '-r')
plt.fill_between(np.arange(0, 197), mean_acftime - std_acftime, mean_acftime + std_acftime,
                 color='r', alpha=0.7)
plt.plot(np.arange(0, 197), mean_acftime_su, '-k')
plt.fill_between(np.arange(0, 197), mean_acftime_su - std_acftime_su, mean_acftime_su + std_acftime_su,
                 color='k', alpha=0.7)
plt.xlabel('Index')
plt.ylabel('acftime')
plt.title('acftime')
plt.show()









#BOXPLOT RSPATIME

BOXRSPA={'RspaTIME':rspatime_lista , 'RspaTIME_su': rspatime_su_lista}
p_values.plotComparisonAcrossLabels2(BOXRSPA)








############################DICC TURBULENCIA##################################################
# Creem un nou diccionari per guardar la informació fusionada
fusion_dict2 = {}

# Iterem sobre les claus de un dels diccionaris (suposem Fr)
for id_sujeto, Rspatime in Fr2.items():
    # Verifiquem si el ID del subjete esta present en els dos diccionaris
    if id_sujeto in P:
        # Obtenim el tipo del subjecte del segon diccionari
        tipo = P[id_sujeto]
        # Guardem la informació fusionada en el nou diccionari
        fusion_dict2[id_sujeto] = {'Rspatime': Rspatime, 'tipo': tipo}
        # Ordenar les claus del diccionari fusion_dict per tipo i despues per ID
fusion_ordenado2 = sorted(fusion_dict2.items(), key=lambda x: (x[1]['tipo'], x[0]))

        # Convertir la llista de tuples novament en un diccionari
fusion_ordenado2_dict = dict(fusion_ordenado2)

# Crear tres diccionaris buits per posar subjectes i fowrev de cada tipus
tipo_HC = {}
tipo_MCI = {}
tipo_AD = {}

# Iterar sobre els elements del diccionari fusionat ordenat
for id_sujeto, info_sujeto in fusion_ordenado2_dict.items():
    tipo = info_sujeto['tipo']
    Rspatime = info_sujeto['Rspatime']
    # Agregar el subjete al diccionari corresponent según el seu tipo
    if tipo == 'HC':
        tipo_HC[id_sujeto] = Rspatime
    elif tipo == 'MCI':
        tipo_MCI[id_sujeto] = Rspatime
    elif tipo == 'AD':
        tipo_AD[id_sujeto] = Rspatime
    elif tipo=='SMC':
        print("ee")


# Convertir els diccionaris en arrays per facilitar l'acces.
array_HC = np.array(list(tipo_HC.values()))
array_MCI = np.array(list(tipo_MCI.values()))
array_AD = np.array(list(tipo_AD.values()))



DICCFINAL2 = {'HC': array_HC, 'MCI': array_MCI, 'AD': array_AD}


p_values.plotComparisonAcrossLabels2(DICCFINAL2)


#CALCULEM TAUWINNER
Tauwinner = calculate_Tauwinner(listafowrevs)



# Creem un nou diccionari para guardar la informació fusionada
fusion_dict = {}

# Iterem sobre las claus de un dels diccionaris
for id_sujeto, fowrev in Fr.items():
    # Verificamos si el ID del subjete esta present en els dos diccionarios
    if id_sujeto in P:
        # Obtenim el tipo del subjete del segon diccionari
        tipo = P[id_sujeto]
        # Guardem la informació fusionada en el nou diccionari
        fusion_dict[id_sujeto] = {'fowrev': fowrev, 'tipo': tipo}
        # Ordenar las claus del diccionari fusion_dict per tipo i después por ID
fusion_ordenado = sorted(fusion_dict.items(), key=lambda x: (x[1]['tipo'], x[0]))

        # Convertir la llista de tuplas novament en un diccionari
fusion_ordenado_dict = dict(fusion_ordenado)






# Crear tres diccionaris buits per posar subjectes i fowrev de cada tipu
tipo_HC = {}
tipo_MCI = {}
tipo_AD = {}

# Iterar sobre els elements del diccionari fusionat ordenat
for id_sujeto, info_sujeto in fusion_ordenado_dict.items():
    tipo = info_sujeto['tipo']
    fowrev = info_sujeto['fowrev']
    # Agregar el subjete al diccionari corresponent segons el seu tipu
    if tipo == 'HC':
        tipo_HC[id_sujeto] = fowrev[Tauwinner]
    elif tipo == 'MCI':
        tipo_MCI[id_sujeto] = fowrev[Tauwinner]
    elif tipo == 'AD':
        tipo_AD[id_sujeto] = fowrev[Tauwinner]
    elif tipo=='SMC':
        print("ee")


# Convertir els diccionaris en arrays per facilitar l'acces.
array_HC = np.array(list(tipo_HC.values()))
array_MCI = np.array(list(tipo_MCI.values()))
array_AD = np.array(list(tipo_AD.values()))



DICCFINAL = {'HC': array_HC, 'MCI': array_MCI, 'AD': array_AD}


p_values.plotComparisonAcrossLabels2(DICCFINAL)




#Fem la matriu i a dins obtenim la classificacio de les dades dels subjects

#Calculem FC

FCsubject = FC.from_fMRI(TimeSeries)
plt.imshow(FCsubject)
plt.show()

#Calculem phIntMat, swIntMat
phIntMatr = phFCD.from_fMRI(TimeSeries, True, True)
matrixFull = phFCD.buildFullMatrix(phIntMatr)
plt.imshow(matrixFull)
plt.show()

swIntMatr = swFCD.from_fMRI(TimeSeries, True, True)
matrixFull = swFCD.buildFullMatrix(swIntMatr)
plt.imshow(matrixFull)
plt.show()



print("done")
