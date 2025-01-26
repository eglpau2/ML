class duomenu_tvarkymas():
    def __init__(self, duomenys):
        self.duomenys = duomenys

#ismetame stulpeli
# uzpildome praleistas reiksmes stulpelyje age
    def stulpeliu_tvark(self):
        self.duomenys.drop(columns="Cabin")
        self.duomenys["Age"] = self.duomenys["Age"].fillna(0)
        self.duomenys['Is_Female'] = (self.duomenys['Sex'] == 'female').astype(int)
        self.duomenys['Is_male'] = (self.duomenys['Sex'] == 'male').astype(int)

        self.duomenys['Is_S'] = (self.duomenys['Embarked'] == 'S').astype(int)
        self.duomenys['Is_C'] = (self.duomenys['Embarked'] == 'C').astype(int)
        self.duomenys['Is_Q'] = (self.duomenys['Embarked'] == 'Q').astype(int)

        return self.duomenys

    def Ticket(self):
        all_letters = self.duomenys['Ticket'].str.isalpha()
        # 2. Remove rows with only letters in 'col1'
        self.duomenys = self.duomenys[~all_letters]

        return self.duomenys
