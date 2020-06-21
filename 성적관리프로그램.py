
class Grade:
    def __init__(self):
        self.hakbunlist=[]
        self.namelist=[]
        self.korlist=[]
        self.englist=[]
        self.mathlist=[]
        flag= True
        print("프로그램을 종료하려면 학번에 '0'을 입력하세요")
        while flag:
            hakbun=input("학번을 입력하시오:")
            if hakbun=='0':
                flag=False
            else:
                name=input("이름을 입력하세요:")
                kor=int(input("국어점수를 입력하세요:"))
                eng=int(input("영어점수를 입력하세요:"))
                math=int(input("수학점수를 입력하세요:"))

                self.hakbunlist.append(hakbun)
                self.namelist.append(name)
                self.korlist.append(kor)
                self.englist.append(eng)
                self.mathlist.append(math)

   
        self.totlist=[]
        self.avglist=[]
        self.hakjumlist=[]
        total=0
        avg=0.0
        for i in range(len(self.korlist)):
            total = self.korlist[i]+self.englist[i]+self.mathlist[i]
            avg=total/3.0
            self.totlist.append(total)
            self.avglist.append(avg)

            if avg>=90:
                grade='A'
            elif avg>= 80:
                grade='B'
            elif avg>= 70:
                grade='C'
            elif avg>= 60:
                grade='D'
            else:
                grade='F'
            self.hakjumlist.append(grade)


    def printList(self):
        print("="*70)
        print("번호\t\t이름\t국어\t영어\t수학\t총점\t평균\t학점")
        print("="*70)
        for i in range(len(self.hakbunlist)):
            print("%3s\t\t%s\t%3d\t%3d\t%3d\t%3d\t%.2f\t%s"
            %(self.hakbunlist[i],self.namelist[i],self.korlist[i],self.englist[i],self.mathlist[i],
            self.totlist[i],self.avglist[i],self.hakjumlist[i]))

myGrade=Grade()
myGrade.printList()