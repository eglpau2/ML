class Nustatymai:
    def MainPath(self):
        return "C:/Users/egle0/Documents/DUOMENYS/Predict Podcast Listening Time/"

    def getT(self):
        return self.MainPath()+"train.csv"

    def getTest(self):
        return self.MainPath()+"test.csv"


    #gal but padaryti kad nuskaitytu tam tikros vietos direktorija ir iskaidytu i skirtingus failus viskas
