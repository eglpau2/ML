class Paket:
    def __init__(self):
        import pandas as pd
        import matplotlib.pyplot as pyl
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score

        self.pd  = pd
        self.pyl = pyl
        self.train_test_split = train_test_split
        self.LinearRegression = LinearRegression
        self.mean_squared_error = mean_squared_error
        self.r2_score = r2_score
