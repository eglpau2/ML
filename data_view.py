class Data_v:
    def __init__(self,data):
        self.data = data

    def aprasas(self):
        a = print(self.data.isnull().sum())
        b = print(self.data.dtypes)
        dtype_counts = self.data.dtypes.value_counts()
        k  = print("how much diffrent types is:", dtype_counts)

        return a, b, k

    def aspras_st(self):
        for col in self.data.columns:
           a =  print(f"---{col}---/n {self.data[col].describe()}")
        return a


