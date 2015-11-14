class GeometricMean():
    def __init__(self,initial,target,beta):
        self.initial = initial
        self.target = target
        self.beta = beta

    def __call__(self,x):
        f1_x = self.initial(x)
        fT_x = self.target(x)
        return f1_x**(1-self.beta) * fT_x**self.beta
