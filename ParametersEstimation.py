import numpy as np
import matplotlib.pyplot as plt
import DynamicsPrograming as DP
from DynamicsPrograming import OptimizationSolver

class DynamicsSystem:
    def __init__(self,p):
        # FIXED, DO NOT CHANGE THE ATTRIBUTES
        self.Initial=0
        self.TimeInterval=0
        self.Parameters=p
    def StateEquation(self,x):
        # Define state equation here
        # VanderPol equation with 2 parameters
        State = np.zeros((2, 1))
        State[0] = self.Parameters[0] * x[1]
        State[1] = self.Parameters[1] * (1 - x[0]**2) * x[1] - x[0]
        return State
# Generate data for fitting
time=np.linspace(0,10,1000)
ic=np.array([[2],[0]])
pTrue=np.array([[3],[2]])
System=DynamicsSystem(pTrue)
System.Initial=ic
System.TimeInterval=time

Measure=DP.OdeSolver(pTrue,System)
Measure=Measure+np.random.normal(0,0.3,Measure.shape)

#--------------------------------
# MaxIter=125; s0=5e-3
p0=np.array([[2],[1]])
p, CostList = OptimizationSolver(p0,Measure,System,Solver='BFGS',MaxIter=12)
EstimatedSystem=System
EstimatedSystem.Parameters=p
EstimatedDynamics=DP.OdeSolver(p,System)


s1=np.transpose(Measure[0,:])
st1=np.transpose(EstimatedDynamics[0,:])
s2=np.transpose(Measure[1,:])
st2=np.transpose(EstimatedDynamics[1,:])
plt.figure

fig, axs = plt.subplots(2)
fig.suptitle('Fitting Results')
axs[0].plot(time, s1)
axs[0].plot(time,st1)
plt.xlabel('time'); plt.ylabel('x_1(t)')
axs[1].plot(time, s2)
axs[1].plot(time,st2)
plt.xlabel('time'); plt.ylabel('x_2(t)')
plt.show()
logCost=np.log10(CostList)
plt.figure
plt.plot(logCost[0,:])
plt.title('Cost v.s Iteration, Log Scale')
plt.xlabel('Iteration'); plt.ylabel('Cost')
