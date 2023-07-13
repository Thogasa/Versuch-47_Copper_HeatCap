import string
import numpy as np

def generatetable(data ,name: string, cap: string, lab: string, titles: list, rotate: bool):
    f = open("build/table_" + name + ".tex", "w")
    f.write(r"\begin{table}" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\caption{" + cap + r"}" + "\n")
    f.write(r"\label{tab:" + lab + r"}" + "\n")
    if (rotate):
        f.write(r"\rotatebox{90}{" + "\n")
    temp = "c"
    for i in range(1, len(titles)):
        temp += " c"
    f.write(r"\begin{tabular}[t]{" + temp + r"}" + "\n")
    f.write(r"\toprule" + "\n")
    temp = ""
    for i in titles:
        temp += i + " & "
    temp = temp[0:len(temp)-3]
    f.write(temp + r"\\" + "\n")
    f.write(r"\midrule" +"\n")
    for i in data:
        temp = ""
        for j in i:
            if(j<10**-10):
                j = 0
            if(j > 0):
                temp += str(j)
            if(j==0):
                temp += "0"
            if(np.isnan(j)):
                temp += "-"
            temp += " & "
        temp = temp[0:len(temp)-3]
        f.write(temp.replace(".",",") + r"\\"+"\n")
    
    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}" + "\n")
    if (rotate):
        f.write(r"}" + "\n")
    f.write(r"\end{table}")
    f.close()

from math import log10 , floor

def round_to_significant(x, sig):
    return round(x, sig-int(floor(log10(abs(x))))-1)

