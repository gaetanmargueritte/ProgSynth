from pysmt.smtlib.parser import ProgSmtLibParser
from pysmt.pslobject import PSLObject

if __name__ == "__main__":
    fname = "dataset/calculator1.psl"
    file = open(fname, 'r')
    parser = ProgSmtLibParser()
    pslobject: PSLObject = parser.get_script(file)

    print(pslobject.func_name)
    print("*"*50)
    print(pslobject.pbe)
    print("*"*50)
    print(pslobject.logics)