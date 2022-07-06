import numpy as np
import skfuzzy as fuzz
from skfuzzy import control
import pandas as pd

distance = control.Antecedent(np.arange(0,1000,200),'distance')
velocity= control.Antecedent(np.arange(0,1000,200),'velocity')
result=control.Consequent(np.arange(0,7000,500),'result')

distance.automf(3)
velocity.automf(3)


#threshold for classification [starting point, peak point, end point]
#you'd want the 'good' part to be narrow, make it strict enough for user to login properly, but not general enough
#that any other value will allow login
result['poor']=fuzz.trimf(result.universe,[0,1500,2500])
result['good']=fuzz.trimf(result.universe,[2500,3500,5000])


rule1=control.Rule(distance['poor'] | velocity['poor'], result['poor'])
rule2 = control.Rule(distance['good'] | velocity['good'], result['good'])

result_ctrl=control.ControlSystem([rule1, rule2])

resultant=control.ControlSystemSimulation(result_ctrl)

#col=["Avg Dist", "Avg velo"]
col=["Distance", "velocity"]
df = pd.read_csv("D:/source/repos/PythonApplication1/mousetest.csv", usecols=col)
resultant.input['distance']=df.iloc[-1,0]
resultant.input['velocity']=df.iloc[-1,1]

#find out how to read at the last row values specifically, it can't work if we read directly

resultant.compute()
print(resultant.output['result'])
result.view(sim=resultant)
