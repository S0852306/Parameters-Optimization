import numpy as np
def OdeSolver(p,Dynamics):
    # Initial is a column vector
    # time is a row vector, time coordinate
    Dynamics.Parameters=p
    Time=Dynamics.TimeInterval
    Initial=Dynamics.Initial
    # State=Dynamics.StateEquation
    Solution=np.zeros([Initial.size,Time.size])
    Solution[:,0]=Initial[:, 0]
    dt=Time[1]-Time[0]
    # Initialize 2-nd Adam-Bashforth Method using Euler's Method
    dx=Dynamics.StateEquation(Solution[:,0])
    PrevDx=dx[:,0]
    Solution[:,1]=Solution[:,1]+dt*dx[:,0]
    # End Euler's Method
    for i in range(Time.size-1):
        dx=Dynamics.StateEquation(Solution[:,i])
        Solution[:,i+1]=Solution[:,i]+dt*(1.5*dx[:,0]-0.5*PrevDx)
        PrevDx=dx[:,0]
    return Solution


def NumericalGrad(Observe,p,Dynamics):
    NumOfVariable=p.size
    Gradient=np.zeros((NumOfVariable,1))
    predict=OdeSolver(p,Dynamics)

    h=1e-3
    disturb=p
    J0=CostFunction(Observe,predict)
    for i in range(NumOfVariable):
        
        disturb[i]=p[i]+h
        DistCurve=OdeSolver(disturb,Dynamics)
        JD=CostFunction(Observe,DistCurve)
        Gradient[i,0]=(JD-J0)/h

    return Gradient,J0

def CostFunction(Observe,Predict):
    E=Observe-Predict
    MSE=np.mean(E**2)
    return MSE
#--------------------------------


def ADAM(p,MaxIter,s0,Dynamics,Observe):
    # Modified ADAM for deterministic optimization
    # Remove bias correction term v/(1-m^Iter)
    g0=np.zeros((p.size,1)); g1=g0
    m0=0.2; m1=0.999; epsilon=1e-8
    CostList=np.zeros((MaxIter,1))
    DispNum=np.floor(MaxIter/10)
    for i in range(MaxIter):
        dp, Cost=NumericalGrad(Observe,p,Dynamics)
        g0=m0*g0+(1-m0)*dp
        g1=m1*g1+(1-m1)*(dp**2)
        p=p-s0*(g0/(np.sqrt(g1)+epsilon))
        Dynamics.Parameters=p
        CostList[i]=Cost
        if np.mod(i+1,DispNum)==0:
            print("Iteration: {}, Cost: {:4.4f}".format(i+1, Cost))
    FinalCost=CostList[-1][0]; #FinalCost=FinalCost[0]
    print("Max Iteration: {}, Cost: {:4.4f}".format(MaxIter, FinalCost))
    pNew=p
    return pNew, CostList

def LineSearch(x,p,Dynamics,SearchVector,dp):
        # Simple Backtracking Line Search
        y=OdeSolver(p,Dynamics)
        C0=CostFunction(x,y)
        MaxIterLS=30; c=1e-4; Decay=0.5

        Scalar=np.transpose(SearchVector) @ dp
        for j in range(MaxIterLS):
            step=np.power(Decay,j)
            pstar=p+step*SearchVector
            yj=OdeSolver(pstar,Dynamics)
            Cj=CostFunction(x,yj)
            LHS=Cj; RHS=C0+c*Scalar*step
            WolfeCondition=LHS<=RHS
            if WolfeCondition==1:
                DescentVector=step*SearchVector
                break
        if WolfeCondition == 0:
                print('Warning, Line Search Fail')
                DescentVector=step*SearchVector 
        return DescentVector,C0

def BFGS(s,y,H):
    # Quasi-Newton method for CNN, Yi-Ren, Goldfarb, 2022
    # Modify s,y update rule makes Quasi-Newton more stable for noisy gradient.
    mu1=0.2; mu2=0.001
    Quad=np.transpose(y) @ H @ y
    InvRho=np.transpose(s) @ y
    if InvRho<mu1*Quad:
        theta=(1-mu1)*Quad/(Quad-InvRho)
    else:
        theta=1
    s=theta*s+(1-theta)*(H @ y); y=y+mu2*s
    Rho=1/(np.transpose(s) @ y); st=np.transpose(s); yt=np.transpose(y);
    H = H+(Rho**2)*(st @ y+ yt @ H @ y)*(np.outer(s, s))-Rho*(H @ np.outer(y, s) + np.outer(s, y) @ H)
    return H

def QuasiNewton(p,MaxIter,Dynamics,Observe):
    dp, Cost=NumericalGrad(Observe,p,Dynamics)
    delta=1e-2
    H=delta*np.eye(p.size)
    CostList=np.zeros((1,MaxIter))
    DispNum=np.floor(MaxIter/10)

    for i in range(MaxIter):
        SearchVector = -H @ dp
        DescentVector, C0 = LineSearch(Observe,p,Dynamics,SearchVector=SearchVector,dp=dp)
        s = DescentVector
        p = p+s
        dpNew, C0 = NumericalGrad(Observe,p,Dynamics)
        yi = dpNew-dp
        dp = dpNew
        H = BFGS(s,yi,H)
        if np.mod(i+1,DispNum)==0:
            print("Iteration: {}, Cost: {:4.4f}".format(i+1, C0))
        CostList[0][i]=C0
    return p, CostList

def OptimizationSolver(p0,Observe,Dynamics,Solver,MaxIter,**kwargs):
    s0 = kwargs.get('s0')
    m1 = kwargs.get('m1')
    m2 = kwargs.get('m2')

    if s0 is None and Solver=='ADAM':
        s0 = 3e-3
    if Solver=='ADAM' and m1 is None:
        m1 = 0.9
    if Solver=='ADAM' and m2 is None:
        m2 = 0.999
    if Solver=='ADAM':
        p, CostList = ADAM(p0,MaxIter,s0,Dynamics,Observe)
    if Solver=='BFGS':
        p, CostList = QuasiNewton(p0,MaxIter,Dynamics,Observe)
    if Solver!='ADAM' and Solver!='BFGS':
        print('Non-exist solver name, Solver options: ADAM, BFGS ')
    return p, CostList
